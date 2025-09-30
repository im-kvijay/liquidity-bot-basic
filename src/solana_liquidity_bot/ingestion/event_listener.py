"""Ingestion orchestrator that combines on-chain and off-chain discovery sources."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

from ..config.settings import AppConfig, DataSourceConfig, get_app_config
from ..datalake.schemas import (
    DammPoolSnapshot,
    LiquidityEvent,
    TokenMetadata,
    TokenOnChainStats,
    TokenUniverseEntry,
)
from ..datalake.storage import SQLiteStorage
from ..monitoring.logger import get_logger
from ..utils.constants import STABLECOIN_MINTS
from .axiom_api import AxiomClient
from .damm_api import DammClient
from .dlmm_api import DlmmClient
from .onchain import SolanaChainAnalyzer
from .pricing import PriceOracle
from .pumpfun_api import PumpFunClient
from .solana_token_list import SolanaTokenListClient
from .token_registry import TokenRegistryAggregator, build_liquidity_event
from .launch_filters import evaluate_launch_candidate


class DiscoveryService:
    """Coordinates discovery across Axiom, PumpFun, and on-chain log listeners."""

    def __init__(
        self,
        storage: Optional[SQLiteStorage] = None,
        app_config: Optional[AppConfig] = None,
        config: Optional[DataSourceConfig] = None,
        logger: Optional[logging.Logger] = None,
        token_registry: Optional[TokenRegistryAggregator] = None,
    ) -> None:
        self._app_config = app_config or get_app_config()
        self._config = config or self._app_config.data_sources
        self._storage = storage or SQLiteStorage(self._app_config.storage.database_path)
        self._axiom_client = AxiomClient(self._config)
        self._pumpfun_client = PumpFunClient(self._config)
        self._damm_client = None
        if self._app_config.venues.damm.enabled:
            self._damm_client = DammClient(self._config, app_config=self._app_config)
        self._dlmm_client = None
        if self._app_config.venues.dlmm.enabled:
            self._dlmm_client = DlmmClient(self._config, app_config=self._app_config)
        self._price_oracle = PriceOracle(self._config)
        self._onchain = SolanaChainAnalyzer(self._app_config.rpc)
        self._token_list_client = (
            SolanaTokenListClient(self._config)
            if getattr(self._config, "enable_solana_token_list", True)
            else None
        )
        self._registry = token_registry or TokenRegistryAggregator(
            storage=self._storage,
            app_config=self._app_config,
            axiom_client=self._axiom_client,
            pumpfun_client=self._pumpfun_client,
            token_list_client=self._token_list_client,
            damm_client=self._damm_client,
            dlmm_client=self._dlmm_client,
            price_oracle=self._price_oracle,
            onchain_analyzer=self._onchain,
        )
        self._logger = logger or get_logger(__name__)
        self._universe_cache: Dict[str, TokenUniverseEntry] = {}

    def discover_tokens(self, limit: int = 20) -> List[TokenMetadata]:
        """Fetch recent listings and apply deduplication."""

        universe = self.discover_universe(limit)
        allowed: List[TokenMetadata] = []
        for entry in universe:
            control_status = entry.control.status if entry.control else "allow"
            if control_status == "allow":
                allowed.append(entry.token)
        self._logger.debug(
            "Discovered %d tokens (%d allowed)", len(universe), len(allowed)
        )
        return allowed

    def discover_universe(self, limit: int) -> List[TokenUniverseEntry]:
        entries = self._registry.build_universe(limit)
        self._capture_launch_candidates(entries)
        self._universe_cache = {entry.token.mint_address: entry for entry in entries}
        return entries

    def fetch_onchain_stats(self, tokens: Iterable[TokenMetadata]) -> List[TokenOnChainStats]:
        stats: List[TokenOnChainStats] = []
        missing: List[TokenMetadata] = []
        for token in tokens:
            cached = self._universe_cache.get(token.mint_address)
            if cached and cached.stats:
                stats.append(cached.stats)
            else:
                missing.append(token)
        if missing:
            stats.extend(self._onchain.collect_stats(missing))
        return stats

    def detect_liquidity_events(
        self,
        tokens: Iterable[TokenMetadata],
        stats: Optional[Iterable[TokenOnChainStats]] = None,
    ) -> List[LiquidityEvent]:
        tokens_list = list(tokens)
        stats_list = list(stats) if stats is not None else self.fetch_onchain_stats(tokens_list)
        stats_by_mint = {stat.token.mint_address: stat for stat in stats_list}

        events: List[LiquidityEvent] = []
        for token in tokens_list:
            entry = self._universe_cache.get(token.mint_address)
            stat = stats_by_mint.get(token.mint_address)
            if entry:
                event = build_liquidity_event(
                    token,
                    entry.stats,
                    entry.damm_pools,
                    entry.dlmm_pools,
                    self._price_oracle,
                )
            else:
                pools = []
                if self._damm_client is not None:
                    pools = self._damm_client.fetch_pools_for_mint(token.mint_address)
                if pools:
                    event = self._build_event_from_pool(token, pools, stat)
                else:
                    event = self._onchain.estimate_liquidity_event(token, stat, self._price_oracle)
            events.append(event)

        self._logger.debug("Generated %d liquidity events", len(events))
        return events

    def _capture_launch_candidates(self, entries: Iterable[TokenUniverseEntry]) -> None:
        config = self._app_config.strategy.launch
        if not config.enabled:
            return
        records = []
        reference_time = datetime.now(timezone.utc)
        for entry in entries:
            record = evaluate_launch_candidate(entry, config, now=reference_time)
            if record is not None:
                records.append(record)
        if records:
            try:
                self._storage.record_damm_launches(records)
            except Exception as exc:  # noqa: BLE001 - defensive logging
                self._logger.debug("Unable to persist DAMM launch records: %s", exc)

    def _build_event_from_pool(
        self,
        token: TokenMetadata,
        pools: List[DammPoolSnapshot],
        stats: Optional[TokenOnChainStats],
    ) -> LiquidityEvent:
        pool = self._select_pool(pools)
        quote_price = self._onchain.determine_quote_price(pool.quote_token_mint, self._price_oracle)
        price_usd = pool.price_usd
        if price_usd is None and pool.base_token_amount > 0:
            ratio = self._safe_ratio(pool.quote_token_amount, pool.base_token_amount)
            if quote_price is None and pool.quote_token_mint in STABLECOIN_MINTS:
                quote_price = 1.0
            if quote_price is None:
                quote_price = self._price_oracle.get_price(pool.quote_token_mint)
            if quote_price is not None:
                price_usd = ratio * quote_price

        tvl_usd = pool.tvl_usd
        if tvl_usd is None and price_usd is not None:
            quote_value = pool.quote_token_amount * (quote_price if quote_price is not None else 1.0)
            base_value = pool.base_token_amount * price_usd
            tvl_usd = base_value + quote_value
        if tvl_usd is None and stats and stats.liquidity_estimate:
            tvl_usd = stats.liquidity_estimate * (price_usd or 0.0) * 2
        if tvl_usd is None:
            tvl_usd = pool.base_token_amount + pool.quote_token_amount

        return LiquidityEvent(
            timestamp=datetime.now(timezone.utc),
            token=token,
            pool_address=pool.address,
            base_liquidity=pool.base_token_amount,
            quote_liquidity=pool.quote_token_amount,
            pool_fee_bps=pool.fee_bps,
            tvl_usd=tvl_usd,
            volume_24h_usd=pool.volume_24h_usd,
            price_usd=price_usd,
            quote_token_mint=pool.quote_token_mint,
            source="damm" if pool.is_active else "damm-inactive",
            launchpad=pool.launchpad,
        )

    def _select_pool(self, pools: List[DammPoolSnapshot]) -> DammPoolSnapshot:
        return max(
            pools,
            key=lambda pool: (
                1 if pool.is_active else 0,
                pool.tvl_usd or (pool.base_token_amount + pool.quote_token_amount),
            ),
        )

    def _safe_ratio(self, numerator: float, denominator: float) -> float:
        if denominator <= 0:
            return 0.0
        return numerator / denominator


__all__ = ["DiscoveryService"]
