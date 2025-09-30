"""Client for retrieving DAMM v2 pool statistics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

import asyncio
import requests
from cachetools import TTLCache

from ..monitoring.event_bus import EVENT_BUS, EventSeverity, EventType
from ..monitoring.metrics import METRICS

from ..config.settings import DataSourceConfig, get_app_config

try:
    from contextlib import asynccontextmanager
except ImportError:
    from contextlib import contextmanager as asynccontextmanager
from ..datalake.schemas import DammPoolSnapshot, TokenMetadata
from ..monitoring.logger import get_logger


@dataclass(slots=True)
class _PoolPayload:
    address: str
    base_mint: str
    quote_mint: str
    base_amount: float
    quote_amount: float
    base_symbol: Optional[str]
    quote_symbol: Optional[str]
    fee_bps: int
    tvl_usd: Optional[float]
    volume_24h_usd: Optional[float]
    price_usd: Optional[float]
    is_active: bool
    created_at: Optional[datetime]
    fee_scheduler_mode: Optional[str]
    fee_scheduler_current_bps: Optional[int]
    fee_scheduler_min_bps: Optional[int]
    fee_scheduler_start_bps: Optional[int]
    fee_collection_token: Optional[str]
    launchpad: Optional[str]


class DammClient:
    """HTTP client that surfaces pool level data for DAMM v2."""

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
        self._cache: TTLCache[str, List[DammPoolSnapshot]] = TTLCache(
            maxsize=256, ttl=self._config.cache_ttl_seconds
        )
        # Request deduplication cache - prevents multiple identical requests
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._logger = get_logger(__name__)
        self._cluster = (self._app_config.mode.cluster or "mainnet-beta").lower()

        # Circuit breaker state
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._circuit_breaker_threshold = 5  # Failures before opening circuit
        self._circuit_breaker_timeout = 300  # 5 minutes timeout
        self._is_circuit_open = False

    def _request(self, mint: str) -> List[_PoolPayload]:
        # Check circuit breaker
        self._check_circuit_breaker()
        if self._is_circuit_open:
            METRICS.increment("damm_circuit_breaker_open")
            self._logger.warning("DAMM circuit breaker open, skipping request for %s", mint)
            return []

        if self._cluster.startswith("devnet") or self._cluster == "testnet":
            base_url = str(getattr(self._config, "damm_devnet_base_url", self._config.damm_base_url)).rstrip("/")
        else:
            base_url = str(self._config.damm_base_url).rstrip("/")
        endpoint = str(self._config.damm_pool_endpoint)
        params = {
            "page": 1,
            "limit": getattr(self._config, "damm_page_limit", 50),
        }
        if "{mint" in endpoint:
            url = f"{base_url}{endpoint}".format(mint=mint)
            params.setdefault("mint", mint)
        else:
            endpoint_path = endpoint or ""
            if endpoint_path and not endpoint_path.startswith("/"):
                endpoint_path = "/" + endpoint_path
            url = f"{base_url}{endpoint_path}"
            params["mint"] = mint
        headers = {"User-Agent": "solana-liquidity-bot/1.0"}
        if self._config.axiom_api_key:
            headers["Authorization"] = f"Bearer {self._config.axiom_api_key}"

        attempts = 0
        while True:
            attempts += 1
            try:
                response = self._session.get(
                    url,
                    headers=headers,
                    timeout=self._config.http_timeout,
                    params=params,
                )
            except requests.RequestException as exc:  # pragma: no cover - network
                retry_status = getattr(getattr(exc, "response", None), "status_code", None)
                if retry_status == 429 and attempts < 4:
                    backoff = 0.5 * attempts
                    self._logger.debug(
                        "DAMM rate limited for %s, retrying in %.2fs", mint, backoff
                    )
                    time.sleep(backoff)
                    continue
                self._logger.warning("Failed to fetch DAMM pools for %s: %s", mint, exc)
                self._record_failure()
                return []

            if response.status_code in {400, 404}:
                # Meteora returns 4xx when no DAMM v2 pools exist for the mint.
                # Treat as "no pools" to avoid noisy warnings.
                self._logger.debug("No DAMM pools for mint %s (%s)", mint, response.status_code)
                return []
            if response.status_code == 429 and attempts < 4:
                backoff = 0.5 * attempts
                self._logger.debug(
                    "DAMM rate limited for %s, retrying in %.2fs", mint, backoff
                )
                time.sleep(backoff)
                continue
            try:
                response.raise_for_status()
                self._record_success()
            except requests.HTTPError as exc:  # pragma: no cover - network
                if response.status_code == 429 and attempts < 4:
                    backoff = 0.5 * attempts
                    self._logger.debug(
                        "DAMM rate limited for %s, retrying in %.2fs", mint, backoff
                    )
                    time.sleep(backoff)
                    continue
                self._logger.warning("Failed to fetch DAMM pools for %s: %s", mint, exc)
                self._record_failure()
                return []
            try:
                payload = response.json()
            except ValueError as exc:  # pragma: no cover - network
                self._logger.warning("Failed to decode DAMM response for %s: %s", mint, exc)
                self._record_failure()
                return []
            break
        items = []
        raw_items = self._extract_items(payload)
        for item in raw_items:
            parsed = self._parse_item(item)
            if parsed is not None:
                items.append(parsed)
        self._record_success()
        return items

    def list_recent_pools(self, limit: int, page: int = 1) -> List[DammPoolSnapshot]:
        if limit <= 0:
            return []
        if self._cluster.startswith("devnet") or self._cluster == "testnet":
            base_url = str(getattr(self._config, "damm_devnet_base_url", self._config.damm_base_url)).rstrip("/")
        else:
            base_url = str(self._config.damm_base_url).rstrip("/")
        endpoint = str(getattr(self._config, "damm_pool_endpoint", "/pools")) or "/pools"
        if "{mint" in endpoint:
            endpoint_path = "/pools"
        else:
            endpoint_path = endpoint
        if endpoint_path and not endpoint_path.startswith("/"):
            endpoint_path = "/" + endpoint_path
        url = f"{base_url}{endpoint_path}"
        params = {"page": max(page, 1), "limit": max(1, limit), "order_by": "created_at_slot_timestamp", "order": "desc"}
        headers = {"User-Agent": "solana-liquidity-bot/1.0"}
        if self._config.axiom_api_key:
            headers["Authorization"] = f"Bearer {self._config.axiom_api_key}"

        attempts = 0
        while True:
            attempts += 1
            try:
                response = self._session.get(
                    url,
                    headers=headers,
                    timeout=self._config.http_timeout,
                    params=params,
                )
            except requests.RequestException as exc:  # pragma: no cover - network
                retry_status = getattr(getattr(exc, "response", None), "status_code", None)
                if retry_status == 429 and attempts < 4:
                    backoff = 0.5 * attempts
                    self._logger.debug(
                        "DAMM recent pools rate limited, retrying in %.2fs", backoff
                    )
                    time.sleep(backoff)
                    continue
                self._logger.warning("Failed to list recent DAMM pools: %s", exc)
                return []
            if response.status_code == 429 and attempts < 4:
                backoff = 0.5 * attempts
                self._logger.debug(
                    "DAMM recent pools rate limited, retrying in %.2fs", backoff
                )
                time.sleep(backoff)
                continue
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:  # pragma: no cover - network
                if response.status_code == 429 and attempts < 4:
                    backoff = 0.5 * attempts
                    self._logger.debug(
                        "DAMM recent pools rate limited, retrying in %.2fs", backoff
                    )
                    time.sleep(backoff)
                    continue
                self._logger.warning("Failed to list recent DAMM pools: %s", exc)
                return []
            try:
                payload = response.json()
            except ValueError as exc:  # pragma: no cover - network
                self._logger.warning("Failed to decode DAMM recent response: %s", exc)
                return []
            break

        items = []
        raw_items = self._extract_items(payload)
        for item in raw_items:
            parsed = self._parse_item(item)
            if parsed is not None:
                items.append(parsed)
        snapshots = [
            DammPoolSnapshot(
                address=item.address,
                base_token_mint=item.base_mint,
                quote_token_mint=item.quote_mint,
                base_token_amount=item.base_amount,
                quote_token_amount=item.quote_amount,
                fee_bps=item.fee_bps,
                base_symbol=item.base_symbol,
                quote_symbol=item.quote_symbol,
                tvl_usd=item.tvl_usd,
                volume_24h_usd=item.volume_24h_usd,
                price_usd=item.price_usd,
                is_active=item.is_active,
                created_at=item.created_at,
                fee_scheduler_mode=item.fee_scheduler_mode,
                fee_scheduler_current_bps=item.fee_scheduler_current_bps,
                fee_scheduler_min_bps=item.fee_scheduler_min_bps,
                fee_scheduler_start_bps=item.fee_scheduler_start_bps,
                fee_collection_token=item.fee_collection_token,
                launchpad=item.launchpad,
            )
            for item in items
        ]
        return snapshots

    def _extract_items(self, payload) -> Iterable[dict]:  # type: ignore[override]
        if isinstance(payload, dict):
            for key in ("pools", "data", "items", "results"):
                if key in payload and isinstance(payload[key], list):
                    return payload[key]
            return [payload]
        if isinstance(payload, list):
            return payload
        return []

    def _parse_item(self, item: dict) -> Optional[_PoolPayload]:  # pragma: no cover - defensive
        try:
            address = (
                item.get("address")
                or item.get("poolId")
                or item.get("id")
                or item.get("pool_address")
            )
            base_mint = (
                item.get("baseMint")
                or item.get("base_token")
                or item.get("baseMintAddress")
                or item.get("token_a_mint")
            )
            quote_mint = (
                item.get("quoteMint")
                or item.get("quote_token")
                or item.get("quoteMintAddress")
                or item.get("token_b_mint")
            )
            if not address or not base_mint or not quote_mint:
                return None
            base_amount = self._to_float(
                item.get("baseReserve")
                or item.get("baseLiquidity")
                or item.get("baseTokenAmount")
                or item.get("baseAmount")
                or item.get("token_a_amount")
            )
            quote_amount = self._to_float(
                item.get("quoteReserve")
                or item.get("quoteLiquidity")
                or item.get("quoteTokenAmount")
                or item.get("quoteAmount")
                or item.get("token_b_amount")
            )
            base_symbol = (
                item.get("baseSymbol")
                or item.get("token_a_symbol")
                or item.get("token_a_symbol")
            )
            quote_symbol = (
                item.get("quoteSymbol")
                or item.get("token_b_symbol")
                or item.get("token_b_symbol")
            )
            raw_fee = item.get("feeBps") or item.get("fee_bps") or item.get("base_fee") or item.get("baseFeeBps")
            fee_bps = self._to_fee_bps(raw_fee)
            tvl_usd = self._maybe_float(
                item.get("tvlUsd") or item.get("tvl_usd") or item.get("tvl")
            )
            volume_24h = self._maybe_float(
                item.get("volume24hUsd")
                or item.get("volume24h_usd")
                or item.get("volume_usd")
                or item.get("volume24h")
            )
            price = self._maybe_float(
                item.get("price") or item.get("priceUsd") or item.get("pool_price")
            )
            is_active = bool(
                item.get("isActive", item.get("farm_active", True))
            )
            created_at = self._parse_datetime(
                item.get("createdAt")
                or item.get("created_at")
                or item.get("created_at_unix")
                or item.get("created_at_slot_timestamp")
            )
            fee_scheduler = item.get("feeScheduler") or item.get("fee_scheduler") or {}
            if isinstance(fee_scheduler, dict):
                scheduler_mode = fee_scheduler.get("mode") or fee_scheduler.get("schedulerMode")
                current_fee = self._to_fee_bps(
                    fee_scheduler.get("currentFeeBps") or fee_scheduler.get("current_fee_bps")
                )
                min_fee = self._to_fee_bps(
                    fee_scheduler.get("minFeeBps") or fee_scheduler.get("min_fee_bps")
                )
                start_fee = self._to_fee_bps(
                    fee_scheduler.get("startFeeBps") or fee_scheduler.get("start_fee_bps")
                )
            else:
                scheduler_mode = None
                current_fee = 0
                min_fee = 0
                start_fee = 0
            fee_collection_token = (
                item.get("feeCollectionToken")
                or item.get("fee_collection_token")
                or fee_scheduler.get("collectionToken")
                if isinstance(fee_scheduler, dict)
                else None
            )
            launchpad = item.get("launchpad") or item.get("launch_pad")

            return _PoolPayload(
                address=address,
                base_mint=str(base_mint),
                quote_mint=str(quote_mint),
                base_amount=base_amount,
                quote_amount=quote_amount,
                base_symbol=str(base_symbol) if base_symbol else None,
                quote_symbol=str(quote_symbol) if quote_symbol else None,
                fee_bps=fee_bps,
                tvl_usd=tvl_usd,
                volume_24h_usd=volume_24h,
                price_usd=price,
                is_active=is_active,
                created_at=created_at,
                fee_scheduler_mode=scheduler_mode,
                fee_scheduler_current_bps=current_fee if current_fee else None,
                fee_scheduler_min_bps=min_fee if min_fee else None,
                fee_scheduler_start_bps=start_fee if start_fee else None,
                fee_collection_token=fee_collection_token,
                launchpad=str(launchpad) if launchpad else None,
            )
        except Exception:  # noqa: BLE001
            return None

    def _to_float(self, value) -> float:  # pragma: no cover - defensive
        try:
            if value is None:
                return 0.0
            if isinstance(value, (int, float)):
                return float(value)
            return float(str(value))
        except (TypeError, ValueError):
            return 0.0

    def _to_fee_bps(self, value) -> int:
        try:
            if value is None:
                return 0
            if isinstance(value, str) and not value.strip():
                return 0
            fee = float(value)
        except (TypeError, ValueError):
            return 0
        if fee <= 10:
            return int(round(fee * 100))
        return int(round(fee))

    def _maybe_float(self, value) -> Optional[float]:  # pragma: no cover - defensive
        try:
            if value is None:
                return None
            if isinstance(value, (int, float)):
                return float(value)
            return float(str(value))
        except (TypeError, ValueError):
            return None

    def _parse_datetime(self, value) -> Optional[datetime]:  # pragma: no cover - defensive
        if value is None:
            return None
        try:
            if isinstance(value, (int, float)):
                # Assume seconds
                return datetime.utcfromtimestamp(float(value))
            if isinstance(value, str):
                try:
                    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
                except ValueError:
                    return datetime.utcfromtimestamp(float(value))
        except Exception:
            return None
        return None

    def fetch_pools_for_mint(self, mint: str) -> List[DammPoolSnapshot]:
        if mint in self._cache:
            return self._cache[mint]

        # Request deduplication - prevent multiple identical requests
        request_key = f"request:{mint}"
        if request_key in self._pending_requests:
            self._logger.debug("Using existing request for mint %s", mint[:8])
            return self._pending_requests[request_key].result()

        # Create new request
        future = asyncio.Future()
        self._pending_requests[request_key] = future

        try:
            raw_payloads = self._request(mint)
            payloads = [
                item
                for item in raw_payloads
                if mint in {item.base_mint, item.quote_mint}
            ]
            if not payloads:
                payloads = raw_payloads
            results = [
                DammPoolSnapshot(
                    address=item.address,
                    base_token_mint=item.base_mint,
                    quote_token_mint=item.quote_mint,
                    base_token_amount=item.base_amount,
                    quote_token_amount=item.quote_amount,
                    fee_bps=item.fee_bps,
                    base_symbol=item.base_symbol,
                    quote_symbol=item.quote_symbol,
                    tvl_usd=item.tvl_usd,
                    volume_24h_usd=item.volume_24h_usd,
                    price_usd=item.price_usd,
                    is_active=item.is_active,
                    created_at=item.created_at,
                    fee_scheduler_mode=item.fee_scheduler_mode,
                    fee_scheduler_current_bps=item.fee_scheduler_current_bps,
                    fee_scheduler_min_bps=item.fee_scheduler_min_bps,
                    fee_scheduler_start_bps=item.fee_scheduler_start_bps,
                    fee_collection_token=item.fee_collection_token,
                    launchpad=item.launchpad,
                )
                for item in payloads
            ]
            if results:
                self._cache[mint] = results

            future.set_result(results)
            return results

        except Exception as e:
            future.set_exception(e)
            self._record_failure()
            raise
        finally:
            # Clean up pending request
            self._pending_requests.pop(request_key, None)

    def _check_circuit_breaker(self) -> None:
        """Check if circuit breaker should be opened or closed."""
        if self._is_circuit_open:
            # Check if we should try to close the circuit
            if self._last_failure_time and datetime.now(timezone.utc) - self._last_failure_time > timedelta(seconds=self._circuit_breaker_timeout):
                self._is_circuit_open = False
                self._failure_count = 0
                self._logger.info("DAMM circuit breaker closed, resuming requests")

    def _record_failure(self) -> None:
        """Record a request failure for circuit breaker logic."""
        self._failure_count += 1
        self._last_failure_time = datetime.now(timezone.utc)

        if self._failure_count >= self._circuit_breaker_threshold:
            self._is_circuit_open = True
            METRICS.increment("damm_circuit_breaker_opened")
            EVENT_BUS.publish(
                EventType.HEALTH,
                {
                    "message": f"DAMM circuit breaker opened after {self._failure_count} failures",
                    "damm_failures": self._failure_count
                },
                severity=EventSeverity.WARNING
            )
            self._logger.warning("DAMM circuit breaker opened due to %d failures", self._failure_count)

    def _record_success(self) -> None:
        """Record a request success for circuit breaker logic."""
        if self._failure_count > 0:
            self._failure_count = 0
            self._last_failure_time = None
            if self._is_circuit_open:
                self._is_circuit_open = False
                METRICS.increment("damm_circuit_breaker_closed")
                self._logger.info("DAMM circuit breaker closed after successful request")

    def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if hasattr(self._session, 'close'):
                self._session.close()
        except Exception as exc:
            self._logger.debug("Error closing DAMM session: %s", exc)

    def fetch_pools_for_tokens(
        self, tokens: Iterable[TokenMetadata]
    ) -> Dict[str, List[DammPoolSnapshot]]:
        mapping: Dict[str, List[DammPoolSnapshot]] = {}
        for token in tokens:
            pools = self.fetch_pools_for_mint(token.mint_address)
            if pools:
                mapping[token.mint_address] = pools
        return mapping


__all__ = ["DammClient"]
