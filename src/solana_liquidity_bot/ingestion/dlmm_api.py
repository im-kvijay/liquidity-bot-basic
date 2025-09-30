"""Client for retrieving Meteora DLMM pool statistics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Iterable, List, Optional

import requests
from cachetools import TTLCache

from ..monitoring.event_bus import EVENT_BUS, EventSeverity, EventType
from ..monitoring.metrics import METRICS

from ..config.settings import DataSourceConfig, get_app_config
from ..datalake.schemas import DlmmPoolSnapshot, TokenMetadata
from ..monitoring.logger import get_logger


@dataclass(slots=True)
class _DlmmPoolPayload:
    address: str
    base_mint: str
    quote_mint: str
    base_amount: float
    quote_amount: float
    fee_bps: int
    bin_step: Optional[int]
    base_virtual_liquidity: Optional[float]
    quote_virtual_liquidity: Optional[float]
    tvl_usd: Optional[float]
    volume_24h_usd: Optional[float]
    price_usd: Optional[float]
    is_active: bool


class DlmmClient:
    """HTTP client that surfaces pool level data for Meteora DLMM venues."""

    def __init__(
        self,
        config: Optional[DataSourceConfig] = None,
        session: Optional[requests.Session] = None,
        *,
        app_config=None,
    ) -> None:
        self._app_config = app_config or get_app_config()
        self._config = config or self._app_config.data_sources
        self._session = session or requests.Session()
        self._cache: TTLCache[str, List[DlmmPoolSnapshot]] = TTLCache(
            maxsize=256, ttl=self._config.cache_ttl_seconds
        )
        self._logger = get_logger(__name__)
        self._cluster = (self._app_config.mode.cluster or "mainnet-beta").lower()

        # Circuit breaker state
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._circuit_breaker_threshold = 5  # Failures before opening circuit
        self._circuit_breaker_timeout = 300  # 5 minutes timeout
        self._is_circuit_open = False

    def _request(self, mint: str) -> List[_DlmmPoolPayload]:
        # Check circuit breaker
        self._check_circuit_breaker()
        if self._is_circuit_open:
            METRICS.increment("dlmm_circuit_breaker_open")
            self._logger.warning("DLMM circuit breaker open, skipping request for %s", mint)
            return []

        if self._cluster.startswith("devnet") or self._cluster == "testnet":
            base_url = str(getattr(self._config, "dlmm_devnet_base_url", self._config.dlmm_base_url)).rstrip("/")
        else:
            base_url = str(self._config.dlmm_base_url).rstrip("/")
        endpoint = str(self._config.dlmm_pool_endpoint)
        params = {
            "page": 1,
            "limit": getattr(self._config, "dlmm_page_limit", 50),
        }
        if "{mint" in endpoint:
            url = f"{base_url}{endpoint}".format(mint=mint)
            params.setdefault("mint", mint)
        else:
            endpoint_path = endpoint or ""
            if endpoint_path and not endpoint_path.startswith("/"):
                endpoint_path = "/" + endpoint_path
            url = f"{base_url}{endpoint_path}"
            # Prefer server-side filtering for pairs API; some backends ignore 'mint',
            # but accept 'mint_x'/'mint_y'. We include all to maximize compatibility.
            if "pairs" in endpoint_path:
                params["mint_x"] = mint
                params["mint_y"] = mint
            else:
                params["mint"] = mint
        headers = {"User-Agent": "solana-liquidity-bot/1.0"}
        try:
            response = self._session.get(
                url,
                headers=headers,
                timeout=self._config.http_timeout,
                params=params,
            )
            if response.status_code == 404:
                # The pairs endpoint returns 404 when no pools exist for the mint. Treat as empty
                # rather than emitting a warning every cycle so dry-runs stay quiet.
                self._logger.debug("No DLMM pools for mint %s (404)", mint)
                return []
            response.raise_for_status()
            self._record_success()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:  # pragma: no cover - network path
            self._logger.warning("Failed to fetch DLMM pools for %s: %s", mint, exc)
            self._record_failure()
            return []
        items = []
        for item in self._extract_items(payload):
            parsed = self._parse_item(item)
            if parsed is not None:
                items.append(parsed)
        self._record_success()
        return items

    def _extract_items(self, payload) -> Iterable[dict]:  # type: ignore[override]
        if isinstance(payload, dict):
            for key in ("pairs", "data", "pools", "items", "results"):
                if key in payload and isinstance(payload[key], list):
                    return payload[key]
            return [payload]
        if isinstance(payload, list):
            return payload
        return []

    def _parse_item(self, item: dict) -> Optional[_DlmmPoolPayload]:  # pragma: no cover - defensive
        try:
            address = (
                item.get("id")
                or item.get("address")
                or item.get("poolAddress")
                or item.get("pair_address")
            )
            base_mint = (
                item.get("baseMint")
                or item.get("baseMintAddress")
                or item.get("mint_x")
            )
            quote_mint = (
                item.get("quoteMint")
                or item.get("quoteMintAddress")
                or item.get("mint_y")
            )
            if not address or not base_mint or not quote_mint:
                return None
            base_amount = self._to_float(
                item.get("baseReserve")
                or item.get("baseLiquidity")
                or item.get("baseTokenAmount")
                or item.get("baseAmount")
                or item.get("reserve_x_amount")
            )
            quote_amount = self._to_float(
                item.get("quoteReserve")
                or item.get("quoteLiquidity")
                or item.get("quoteTokenAmount")
                or item.get("quoteAmount")
                or item.get("reserve_y_amount")
            )
            fee_bps = self._extract_fee_bps(item)
            bin_step = self._maybe_int(item.get("binStep") or item.get("bin_step"))
            base_virtual = self._maybe_float(
                item.get("baseVirtualLiquidity")
                or item.get("base_virtual_liquidity")
                or item.get("virtual_base_liquidity")
            )
            quote_virtual = self._maybe_float(
                item.get("quoteVirtualLiquidity")
                or item.get("quote_virtual_liquidity")
                or item.get("virtual_quote_liquidity")
            )
            tvl_usd = self._maybe_float(
                item.get("tvlUsd")
                or item.get("tvl_usd")
                or item.get("liquidity")
            )
            volume_24h = self._maybe_float(
                item.get("volume24hUsd")
                or item.get("volume24h_usd")
                or item.get("volumeUsd")
                or item.get("trade_volume_24h")
            )
            price = self._maybe_float(
                item.get("price")
                or item.get("priceUsd")
                or item.get("current_price")
            )
            is_active = bool(
                item.get("isActive", not item.get("is_blacklisted", False))
            )
            return _DlmmPoolPayload(
                address=str(address),
                base_mint=str(base_mint),
                quote_mint=str(quote_mint),
                base_amount=base_amount,
                quote_amount=quote_amount,
                fee_bps=fee_bps,
                bin_step=bin_step,
                base_virtual_liquidity=base_virtual,
                quote_virtual_liquidity=quote_virtual,
                tvl_usd=tvl_usd,
                volume_24h_usd=volume_24h,
                price_usd=price,
                is_active=is_active,
            )
        except Exception:  # noqa: BLE001
            return None

    def _to_float(self, value) -> float:
        try:
            if value is None:
                return 0.0
            if isinstance(value, (int, float)):
                return float(value)
            return float(str(value))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return 0.0

    def _maybe_float(self, value) -> Optional[float]:
        try:
            if value is None:
                return None
            if isinstance(value, (int, float)):
                return float(value)
            return float(str(value))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return None

    def _maybe_int(self, value) -> Optional[int]:
        try:
            if value is None:
                return None
            if isinstance(value, int):
                return value
            return int(str(value))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return None

    def _extract_fee_bps(self, item: dict) -> int:
        direct_keys = ("feeBps", "fee_bps", "base_fee_bps")
        for key in direct_keys:
            value = item.get(key)
            if value is None:
                continue
            numeric = self._maybe_float(value)
            if numeric is None:
                continue
            if 0 < numeric < 1:
                return int(round(numeric * 100))
            return int(round(numeric))

        percent_keys = ("base_fee_percentage", "base_fee_percent", "fee_percentage")
        for key in percent_keys:
            value = item.get(key)
            if value is None:
                continue
            numeric = self._maybe_float(value)
            if numeric is not None:
                return int(round(numeric * 100))

        fallback = self._maybe_float(item.get("base_fee"))
        if fallback is not None:
            if 0 < fallback < 1:
                return int(round(fallback * 100))
            return int(round(fallback))

        return 30

    def fetch_pools_for_mint(self, mint: str) -> List[DlmmPoolSnapshot]:
        if mint in self._cache:
            return self._cache[mint]
        payloads = self._request(mint)
        results = [
            DlmmPoolSnapshot(
                address=item.address,
                base_token_mint=item.base_mint,
                quote_token_mint=item.quote_mint,
                base_token_amount=item.base_amount,
                quote_token_amount=item.quote_amount,
                fee_bps=item.fee_bps,
                bin_step=item.bin_step,
                base_virtual_liquidity=item.base_virtual_liquidity,
                quote_virtual_liquidity=item.quote_virtual_liquidity,
                tvl_usd=item.tvl_usd,
                volume_24h_usd=item.volume_24h_usd,
                price_usd=item.price_usd,
                is_active=item.is_active,
            )
            for item in payloads
        ]
        if results:
            self._cache[mint] = results
        return results

    def _check_circuit_breaker(self) -> None:
        """Check if circuit breaker should be opened or closed."""
        if self._is_circuit_open:
            # Check if we should try to close the circuit
            if self._last_failure_time and datetime.now(timezone.utc) - self._last_failure_time > timedelta(seconds=self._circuit_breaker_timeout):
                self._is_circuit_open = False
                self._failure_count = 0
                self._logger.info("DLMM circuit breaker closed, resuming requests")

    def _record_failure(self) -> None:
        """Record a request failure for circuit breaker logic."""
        self._failure_count += 1
        self._last_failure_time = datetime.now(timezone.utc)

        if self._failure_count >= self._circuit_breaker_threshold:
            self._is_circuit_open = True
            METRICS.increment("dlmm_circuit_breaker_opened")
            EVENT_BUS.publish(
                EventType.HEALTH,
                {
                    "message": f"DLMM circuit breaker opened after {self._failure_count} failures",
                    "dlmm_failures": self._failure_count
                },
                severity=EventSeverity.WARNING
            )
            self._logger.warning("DLMM circuit breaker opened due to %d failures", self._failure_count)

    def _record_success(self) -> None:
        """Record a request success for circuit breaker logic."""
        if self._failure_count > 0:
            self._failure_count = 0
            self._last_failure_time = None
            if self._is_circuit_open:
                self._is_circuit_open = False
                METRICS.increment("dlmm_circuit_breaker_closed")
                self._logger.info("DLMM circuit breaker closed after successful request")

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if hasattr(self._session, 'close'):
                self._session.close()
        except Exception as exc:
            self._logger.debug("Error closing DLMM session: %s", exc)

    def fetch_pools_for_tokens(
        self, tokens: Iterable[TokenMetadata]
    ) -> dict[str, List[DlmmPoolSnapshot]]:
        mapping: dict[str, List[DlmmPoolSnapshot]] = {}
        for token in tokens:
            pools = self.fetch_pools_for_mint(token.mint_address)
            if pools:
                mapping[token.mint_address] = pools
        return mapping


__all__ = ["DlmmClient"]
