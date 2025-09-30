"""On-chain data enrichment utilities."""

from __future__ import annotations

import base64
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

from cachetools import TTLCache
from solders.pubkey import Pubkey

try:  # pragma: no cover - exercised when dependency installed
    from solana.rpc.api import Client
except ModuleNotFoundError:  # pragma: no cover - fallback for test environments
    from ..execution.solana_compat import Client

try:  # pragma: no cover - exercised when dependency installed
    from spl.token.constants import TOKEN_PROGRAM_ID
except ModuleNotFoundError:  # pragma: no cover - fallback value
    class _TokenProgramId(str):
        def __new__(cls) -> "_TokenProgramId":  # type: ignore[override]
            return str.__new__(cls, "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")

    TOKEN_PROGRAM_ID = _TokenProgramId()

try:  # pragma: no cover - exercised when dependency installed
    from spl.token._layouts import MINT_LAYOUT
except ModuleNotFoundError:  # pragma: no cover - fallback decoder
    from types import SimpleNamespace

    class _DefaultMintLayout:
        def parse(self, _data: bytes) -> SimpleNamespace:  # type: ignore[override]
            return SimpleNamespace(
                decimals=6,
                mintAuthorityOption=0,
                freezeAuthorityOption=0,
                mintAuthority=b"",
                freezeAuthority=b"",
            )

    MINT_LAYOUT = _DefaultMintLayout()

from ..config.settings import RPCConfig, get_app_config
from ..datalake.schemas import LiquidityEvent, TokenMetadata, TokenOnChainStats
from ..monitoring.logger import get_logger
from ..utils.constants import STABLECOIN_MINTS
from .pricing import PriceOracle


class SolanaChainAnalyzer:
    """Gathers account, supply, and holder distribution statistics via RPC."""

    def __init__(self, config: Optional[RPCConfig] = None) -> None:
        self._config = config or get_app_config().rpc
        endpoints = [str(self._config.primary_url), *map(str, self._config.fallback_urls)]
        self._endpoints = endpoints
        self._logger = get_logger(__name__)
        self._cache: TTLCache[str, TokenOnChainStats] = TTLCache(maxsize=512, ttl=180)
        self._locked_cache: TTLCache[str, TokenOnChainStats] = TTLCache(maxsize=512, ttl=3600)
        self._cache_lock = threading.Lock()
        self._thread_local = threading.local()
        self._concurrency = max(1, int(getattr(self._config, "request_concurrency", 4)))

    def _execute(self, method_name: str, *args, **kwargs):
        last_exc: Optional[Exception] = None
        clients = self._clients_for_thread()
        for endpoint, client in zip(self._endpoints, clients):
            method = getattr(client, method_name)
            try:
                result = method(*args, **kwargs)
                if isinstance(result, dict):
                    return result
                if hasattr(result, "to_json"):
                    payload = result.to_json()
                    if isinstance(payload, str):
                        try:
                            return json.loads(payload)
                        except json.JSONDecodeError:  # pragma: no cover - defensive
                            return payload
                    return payload
                return result
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                self._logger.debug("RPC %s failed on %s: %s", method_name, endpoint, exc)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"RPC {method_name} failed for unknown reasons")

    def collect_stats(self, tokens: Iterable[TokenMetadata]) -> List[TokenOnChainStats]:
        """Return on-chain stats for the provided tokens."""

        tokens_list = list(tokens)
        results_by_mint: Dict[str, TokenOnChainStats] = {}
        missing: List[TokenMetadata] = []

        with self._cache_lock:
            for token in tokens_list:
                if token.mint_address in self._locked_cache:
                    results_by_mint[token.mint_address] = self._locked_cache[token.mint_address]
                    continue
                if token.mint_address in STABLECOIN_MINTS:
                    stats = self._static_stats_for_stablecoin(token)
                    results_by_mint[token.mint_address] = stats
                    self._locked_cache[token.mint_address] = stats
                    continue
                cached = self._cache.get(token.mint_address)
                if cached is not None:
                    results_by_mint[token.mint_address] = cached
                else:
                    missing.append(token)

        new_stats: List[tuple[str, TokenOnChainStats]] = []

        if missing:
            def worker(token: TokenMetadata) -> Optional[tuple[str, TokenOnChainStats]]:
                try:
                    stats = self._collect_single(token)
                    return token.mint_address, stats
                except Exception as exc:  # noqa: BLE001
                    self._logger.warning(
                        "Failed to gather on-chain stats for %s: %s", token.mint_address, exc
                    )
                    return None

            if self._concurrency > 1 and len(missing) > 1:
                with ThreadPoolExecutor(max_workers=self._concurrency) as executor:
                    for outcome in executor.map(worker, missing):
                        if outcome is not None:
                            new_stats.append(outcome)
            else:
                for token in missing:
                    result = worker(token)
                    if result is not None:
                        new_stats.append(result)

            if new_stats:
                with self._cache_lock:
                    for mint, stats in new_stats:
                        self._cache[mint] = stats
                        if (
                            stats.mint_authority is None
                            and stats.freeze_authority is None
                            and stats.minted_at is not None
                            and (datetime.now(timezone.utc) - stats.minted_at).total_seconds() > 900
                        ):
                            self._locked_cache[mint] = stats
                for mint, stats in new_stats:
                    results_by_mint[mint] = stats

        ordered_results: List[TokenOnChainStats] = []
        for token in tokens_list:
            stats = results_by_mint.get(token.mint_address)
            if stats is not None:
                ordered_results.append(stats)
        return ordered_results

    def _collect_single(self, token: TokenMetadata) -> TokenOnChainStats:
        mint_pubkey = Pubkey.from_string(token.mint_address)

        supply_resp = self._execute("get_token_supply", mint_pubkey)
        value = supply_resp.get("result", {}).get("value", {})
        decimals = int(value.get("decimals", token.decimals))
        amount_raw = value.get("amount")
        total_supply = self._convert_amount(amount_raw, decimals)
        token.decimals = decimals

        holders_resp = self._execute("get_token_largest_accounts", mint_pubkey)
        holder_entries = holders_resp.get("result", {}).get("value", [])
        holder_amounts = [self._parse_ui_amount(entry, decimals) for entry in holder_entries]

        holder_count = self._estimate_holder_count(mint_pubkey)
        if holder_count == 0 and holder_entries:
            holder_count = len(holder_entries)
        top_holder_pct = self._ratio(holder_amounts[0] if holder_amounts else 0.0, total_supply)
        top10_holder_pct = self._ratio(sum(holder_amounts[:10]), total_supply)

        mint_info = self._execute("get_account_info", mint_pubkey, encoding="base64")
        mint_value = mint_info.get("result", {}).get("value")
        mint_authority: Optional[str] = None
        freeze_authority: Optional[str] = None
        if mint_value and mint_value.get("data"):
            raw_data = base64.b64decode(mint_value["data"][0])
            parsed = MINT_LAYOUT.parse(raw_data)
            mint_authority_option = getattr(
                parsed,
                "mintAuthorityOption",
                getattr(parsed, "mint_authority_option", 0),
            )
            freeze_authority_option = getattr(
                parsed,
                "freezeAuthorityOption",
                getattr(parsed, "freeze_authority_option", 0),
            )
            mint_authority_raw = getattr(
                parsed,
                "mintAuthority",
                getattr(parsed, "mint_authority", b""),
            )
            freeze_authority_raw = getattr(
                parsed,
                "freezeAuthority",
                getattr(parsed, "freeze_authority", b""),
            )
            if mint_authority_option:
                mint_authority = str(Pubkey.from_bytes(bytes(mint_authority_raw)))
            if freeze_authority_option:
                freeze_authority = str(Pubkey.from_bytes(bytes(freeze_authority_raw)))

        signature_entries: List[dict] = []
        try:
            signatures_resp = self._execute("get_signatures_for_address", mint_pubkey, limit=25)
            signature_entries = signatures_resp.get("result", [])
        except Exception as exc:  # noqa: BLE001 - some RPC nodes return -32019/unsupported type
            signature_entries = []
            self._logger.debug("Signature fetch failed for %s: %s", token.mint_address, exc)
        minted_at = self._extract_timestamp(signature_entries, reverse=True)
        last_activity = self._extract_timestamp(signature_entries, reverse=False)

        liquidity_estimate = holder_amounts[0] if holder_amounts else 0.0

        return TokenOnChainStats(
            token=token,
            total_supply=total_supply,
            decimals=decimals,
            holder_count=holder_count,
            top_holder_pct=top_holder_pct,
            top10_holder_pct=top10_holder_pct,
            liquidity_estimate=liquidity_estimate,
            minted_at=minted_at,
            last_activity_at=last_activity,
            mint_authority=mint_authority,
            freeze_authority=freeze_authority,
        )

    def _static_stats_for_stablecoin(self, token: TokenMetadata) -> TokenOnChainStats:
        token.decimals = 6
        return TokenOnChainStats(
            token=token,
            total_supply=0.0,
            decimals=token.decimals,
            holder_count=0,
            top_holder_pct=0.0,
            top10_holder_pct=0.0,
            liquidity_estimate=0.0,
            minted_at=None,
            last_activity_at=None,
            mint_authority=None,
            freeze_authority=None,
        )

    def _estimate_holder_count(self, mint: Pubkey) -> int:
        try:
            response = self._execute(
                "get_program_accounts",
                TOKEN_PROGRAM_ID,
                encoding="base64",
                data_size=165,
                filters=[{"memcmp": {"offset": 0, "bytes": str(mint)}}],
            )
            accounts = response.get("result", [])
            return len(accounts)
        except Exception:  # noqa: BLE001
            return 0

    def _clients_for_thread(self) -> List[Client]:
        clients: Optional[List[Client]] = getattr(self._thread_local, "clients", None)
        if clients is None:
            clients = [Client(endpoint, timeout=self._config.request_timeout) for endpoint in self._endpoints]
            self._thread_local.clients = clients
        return clients

    def _convert_amount(self, amount: Optional[str], decimals: int) -> float:
        if amount is None:
            return 0.0
        try:
            return int(amount) / (10 ** decimals)
        except (TypeError, ValueError):
            try:
                return float(amount)
            except (TypeError, ValueError):
                return 0.0

    def _parse_ui_amount(self, entry: dict, decimals: int) -> float:
        if not entry:
            return 0.0
        ui_amount = entry.get("uiAmount")
        if ui_amount is not None:
            try:
                return float(ui_amount)
            except (TypeError, ValueError):
                pass
        return self._convert_amount(entry.get("amount"), decimals)

    def _ratio(self, numerator: float, denominator: float) -> float:
        if denominator <= 0:
            return 1.0 if numerator > 0 else 0.0
        return max(0.0, min(numerator / denominator, 1.0))

    def _extract_timestamp(self, entries: List[dict], reverse: bool) -> Optional[datetime]:
        iterable = reversed(entries) if reverse else entries
        for entry in iterable:
            block_time = entry.get("blockTime")
            if block_time:
                return datetime.utcfromtimestamp(block_time)
        return None

    def estimate_liquidity_event(
        self,
        token: TokenMetadata,
        stats: Optional[TokenOnChainStats],
        price_oracle: Optional[PriceOracle] = None,
    ) -> LiquidityEvent:
        base_liquidity = stats.liquidity_estimate if stats else 0.0
        price = price_oracle.get_price(token.mint_address) if price_oracle else None
        quote_liquidity = base_liquidity * price if price else 0.0
        tvl_usd = (base_liquidity * price * 2) if price else None
        return LiquidityEvent(
            timestamp=datetime.now(timezone.utc),
            token=token,
            pool_address=f"heuristic-{token.mint_address[:8]}",
            base_liquidity=base_liquidity,
            quote_liquidity=quote_liquidity,
            pool_fee_bps=30,
            tvl_usd=tvl_usd,
            volume_24h_usd=None,
            price_usd=price,
            quote_token_mint=None,
            source="onchain-heuristic",
        )

    def determine_quote_price(
        self,
        quote_mint: Optional[str],
        oracle: PriceOracle,
    ) -> Optional[float]:
        if not quote_mint:
            return None
        if quote_mint in STABLECOIN_MINTS:
            return 1.0
        return oracle.get_price(quote_mint)


__all__ = ["SolanaChainAnalyzer"]
