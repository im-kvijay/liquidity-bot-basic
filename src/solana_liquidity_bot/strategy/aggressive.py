"""Aggressive market making strategy that prioritises high-liquidity pools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from ..datalake.schemas import StrategyDecision, TokenUniverseEntry
from ..monitoring.logger import get_logger
from ..strategy.allocator import Allocator
from .base import Strategy, StrategyContext


@dataclass(slots=True)
class _Candidate:
    entry: TokenUniverseEntry
    score: float


class AggressiveMakerStrategy(Strategy):
    """Creates maker positions on the most liquid pools with relaxed heuristics."""

    name = "agg_maker"

    def __init__(self, allocator: Allocator, *, max_entries: int | None = None) -> None:
        self._allocator = allocator
        self._logger = get_logger(__name__)
        self._max_entries = max_entries

    def generate(self, context: StrategyContext) -> List[StrategyDecision]:
        config = context.strategy_config
        limit = self._max_entries or config.max_positions
        candidates = self._select_candidates(context, limit)
        decisions: List[StrategyDecision] = []
        for item in candidates:
            token = item.entry.token
            position = context.positions.get(token.mint_address)
            score_obj = context.scores.get(token.mint_address)
            if score_obj is not None:
                allocation = self._allocator.determine_allocation(
                    score_obj,
                    position,
                    strategy="maker",
                )
            else:
                allocation = 0.0
            if allocation <= 0:
                allocation = max(config.allocation_per_position, 0.0)
            if allocation <= 0:
                continue
            event = item.entry.liquidity_event
            decision = StrategyDecision(
                token=token,
                action="rebalance" if position else "enter",
                allocation=allocation,
                priority=item.score,
                strategy=self.name,
                side="maker",
                price_usd=self._resolve_price(item.entry),
                pool_address=event.pool_address if event else None,
                venue=event.source if event else None,
                base_liquidity=item.entry.risk.liquidity_usd if item.entry.risk else None,
                quote_liquidity=item.entry.risk.volume_24h_usd if item.entry.risk else None,
                pool_fee_bps=event.pool_fee_bps if event else None,
                token_decimals=token.decimals,
                quote_token_decimals=None,
                correlation_id=f"{self.name}:{token.mint_address}",
                metadata={
                    "tvl_usd": item.entry.risk.liquidity_usd if item.entry.risk else 0.0,
                    "volume_usd": item.entry.risk.volume_24h_usd if item.entry.risk else 0.0,
                },
            )
            evaluations = context.router.evaluate_routes(decision, item.entry, context.mode, limit=1)
            if not evaluations:
                continue
            best = evaluations[0]
            decision.venue = best.quote.venue
            decision.pool_address = best.quote.pool_address
            decision.expected_value = best.score
            decision.expected_slippage_bps = best.quote.expected_slippage_bps
            decision.max_slippage_bps = min(
                context.risk_limits.max_slippage_bps,
                self._allocator.slippage_budget("maker"),
            )
            decision.expected_fees_usd = best.quote.expected_fees_usd
            decision.expected_rebate_usd = (best.quote.rebate_bps / 10_000) * allocation
            decision.metadata.update(best.quote.extras)
            decisions.append(decision)
        return decisions

    def _select_candidates(self, context: StrategyContext, limit: int) -> List[_Candidate]:
        entries: Iterable[TokenUniverseEntry] = context.universe.values()
        ranked: List[_Candidate] = []
        for entry in entries:
            if not entry.risk:
                continue
            control = entry.control
            if control is not None and control.status != "allow":
                continue
            liquidity = entry.risk.liquidity_usd or 0.0
            volume = entry.risk.volume_24h_usd or 0.0
            fee_bps = 0
            if entry.liquidity_event and entry.liquidity_event.pool_fee_bps:
                fee_bps = entry.liquidity_event.pool_fee_bps
            elif entry.risk and hasattr(entry.risk, "fee_bps") and entry.risk.fee_bps:
                fee_bps = entry.risk.fee_bps  # type: ignore[attr-defined]
            if fee_bps <= 0:
                fee_bps = 20
            fee_rate = fee_bps / 10_000
            daily_fee_revenue = volume * fee_rate
            fee_yield = daily_fee_revenue / max(liquidity, 1.0)
            if fee_yield < context.strategy_config.aggressive_min_fee_yield:
                continue
            score = daily_fee_revenue + (fee_yield * 10_000)
            ranked.append(_Candidate(entry=entry, score=score))
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:limit]

    def _resolve_price(self, entry: TokenUniverseEntry) -> float | None:
        event = entry.liquidity_event
        if event and event.price_usd:
            return event.price_usd
        if entry.risk and entry.risk.liquidity_usd:
            return None
        return None


__all__ = ["AggressiveMakerStrategy"]
