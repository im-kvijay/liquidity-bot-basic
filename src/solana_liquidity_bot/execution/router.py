"""Venue routing engine that selects the optimal execution path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import time

from ..config.settings import AppConfig, AppMode, RouterConfig, get_app_config
from ..datalake.schemas import StrategyDecision, TokenUniverseEntry
from ..monitoring.metrics import METRICS
from ..monitoring.logger import get_logger
from ..utils.constants import STABLECOIN_MINTS
from .venues import PoolContext, QuoteRequest, VenueAdapter, VenueQuote


@dataclass(slots=True)
class RoutingDecision:
    adapter: VenueAdapter
    request: QuoteRequest
    quote: VenueQuote
    score: float


class OrderRouter:
    """Scores venue quotes and selects the most attractive route."""

    def __init__(
        self,
        adapters: Sequence[VenueAdapter],
        app_config: Optional[AppConfig] = None,
    ) -> None:
        if not adapters:
            raise ValueError("At least one venue adapter must be supplied")
        self._app_config = app_config or get_app_config()
        self._config: RouterConfig = self._app_config.router
        self._adapters = list(adapters)
        self._logger = get_logger(__name__)

    def evaluate_routes(
        self,
        decision: StrategyDecision,
        entry: TokenUniverseEntry,
        mode: AppMode,
        *,
        limit: Optional[int] = None,
    ) -> list[RoutingDecision]:
        candidates: list[RoutingDecision] = []
        target_pool = decision.pool_address if decision.action == "exit" else None
        target_venue = decision.venue if decision.action == "exit" else None
        for adapter in self._adapters:
            if target_venue and adapter.name != target_venue:
                continue
            for pool in adapter.pools(entry):
                if target_pool and pool.address != target_pool:
                    continue
                request = self._build_request(decision, entry, pool, mode)
                try:
                    quote = adapter.quote(request)
                except Exception as exc:  # noqa: BLE001 - defensive logging
                    self._logger.warning(
                        "Adapter %s failed to quote pool %s: %s",
                        adapter.name,
                        pool.address,
                        exc,
                    )
                    continue
                if quote is None:
                    continue
                score = self._score_quote(decision, quote)
                if score <= 0:
                    continue
                candidates.append(
                    RoutingDecision(adapter=adapter, request=request, quote=quote, score=score)
                )
        candidates.sort(key=lambda item: item.score, reverse=True)
        if limit is not None:
            return candidates[:limit]
        return candidates

    def route(
        self,
        decision: StrategyDecision,
        entry: TokenUniverseEntry,
        mode: AppMode,
    ) -> Optional[RoutingDecision]:
        start = time.perf_counter()
        candidates = self.evaluate_routes(decision, entry, mode, limit=1)
        METRICS.observe("router_latency_ms", (time.perf_counter() - start) * 1000.0)
        return candidates[0] if candidates else None

    def _build_request(
        self,
        decision: StrategyDecision,
        entry: TokenUniverseEntry,
        pool: PoolContext,
        mode: AppMode,
    ) -> QuoteRequest:
        base_price_hint = decision.price_usd or pool.price_usd
        quote_price_hint = 1.0 if pool.quote_mint in STABLECOIN_MINTS else None
        return QuoteRequest(
            decision=decision,
            pool=pool,
            risk=entry.risk,
            mode=mode,
            allocation_usd=decision.allocation,
            base_price_hint=base_price_hint,
            quote_price_hint=quote_price_hint,
            universe_entry=entry,
            position=decision.position_snapshot,
        )

    def _score_quote(self, decision: StrategyDecision, quote: VenueQuote) -> float:
        if decision.action == "exit":
            return max(decision.allocation, 1.0)

        config = self._config
        allocation = max(decision.allocation, 1.0)

        # Hard cost cap: block routes with excessive costs
        # Get base fee from decision metadata or quote
        base_fee_bps = 0
        if hasattr(decision, 'metadata') and decision.metadata:
            base_fee_bps = decision.metadata.get("launch_fee_bps", 0)
        
        # Fallback to quote fee if metadata missing
        if base_fee_bps == 0:
            base_fee_bps = quote.fee_bps

        # Estimate priority fee (5 bps conservative estimate)
        priority_fee_bps = 5

        # Total costs = base fee + slippage + priority fee
        total_costs_bps = base_fee_bps + quote.expected_slippage_bps + priority_fee_bps

        # Hard cost gate: reject if total costs exceed 100 bps (1%)
        if total_costs_bps >= 100:
            return 0.0

        liquidity_ratio = quote.pool_liquidity_usd / max(allocation * 3.0, 1.0)
        liquidity_component = min(max(liquidity_ratio, 0.0), 1.5)
        fee_component = max(0.0, 1.0 - (quote.fee_bps / 150.0))
        volatility_component = max(0.0, 1.0 - min(quote.volatility_penalty, 1.0))
        depth_component = min(max(quote.depth_score, 0.0), 1.0)
        rebate_component = min(max(quote.rebate_bps / 50.0, 0.0), 1.0)
        fill_component = min(max(quote.liquidity_score, 0.0), 1.0)
        if quote.expected_slippage_bps > config.slippage_safety_bps * 1.6:
            return 0.0
        slippage_penalty = min(
            (quote.expected_slippage_bps + config.slippage_safety_bps) / 1000.0,
            1.5,
        )
        raw_score = (
            (config.liquidity_weight * min(liquidity_component, 1.0))
            + (config.fee_weight * fee_component)
            + (config.volatility_weight * volatility_component)
            + (config.depth_weight * depth_component)
            + (config.rebate_weight * rebate_component)
            + (config.fill_weight * fill_component)
        )
        if quote.pool_liquidity_usd < allocation * 1.2:
            raw_score -= 0.2
        adjusted = raw_score - (0.22 * slippage_penalty)
        return max(adjusted, 0.0)


__all__ = ["OrderRouter", "RoutingDecision"]
