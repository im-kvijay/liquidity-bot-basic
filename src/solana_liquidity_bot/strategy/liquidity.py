"""Liquidity provisioning strategy with volatility-aware rebalancing."""

from __future__ import annotations

from typing import List

from ..datalake.schemas import StrategyDecision
from ..monitoring.logger import get_logger
from .allocator import Allocator
from .base import Strategy, StrategyContext


class LiquidityProvisionStrategy(Strategy):
    """Adjusts LP positions based on volatility and inventory risk."""

    name = "lp_rebalance"

    def __init__(self, allocator: Allocator) -> None:
        self._allocator = allocator
        self._logger = get_logger(__name__)

    def generate(self, context: StrategyContext) -> List[StrategyDecision]:
        decisions: List[StrategyDecision] = []
        for mint, position in context.positions.items():
            entry = context.universe.get(mint)
            if entry is None:
                continue
            volatility = entry.risk.volatility_score if entry and entry.risk else 0.5
            target_allocation = self._allocator.rebalance_allocation(position, volatility)
            if target_allocation <= 0:
                continue
            liquidity_usd = entry.risk.liquidity_usd if entry and entry.risk else None
            quote_liquidity = entry.risk.volume_24h_usd if entry and entry.risk else None
            fee_bps = (
                entry.liquidity_event.pool_fee_bps
                if entry and entry.liquidity_event
                else position.pool_fee_bps if hasattr(position, "pool_fee_bps") else 30
            )
            decision = StrategyDecision(
                token=position.token,
                action="rebalance",
                allocation=target_allocation,
                priority=1.0,
                strategy=self.name,
                side="lp",
                venue=position.venue,
                pool_address=position.pool_address,
                price_usd=(
                    entry.liquidity_event.price_usd
                    if entry and entry.liquidity_event
                    else position.entry_price
                ),
                base_liquidity=liquidity_usd,
                quote_liquidity=quote_liquidity,
                pool_fee_bps=fee_bps,
                token_decimals=position.token.decimals,
                correlation_id=f"{self.name}:{mint}",
                metadata={
                    "volatility_score": volatility,
                    "current_allocation": position.allocation,
                },
                notes=[f"volatility={volatility:.2f}"],
            )
            evaluations = context.router.evaluate_routes(
                decision, entry, context.mode, limit=1
            )
            if not evaluations:
                continue
            best = evaluations[0]
            decision.venue = best.quote.venue
            decision.pool_address = best.quote.pool_address
            decision.expected_value = best.score
            decision.expected_slippage_bps = best.quote.expected_slippage_bps
            decision.max_slippage_bps = min(
                context.risk_limits.max_slippage_bps,
                self._allocator.slippage_budget("lp"),
            )
            decision.expected_fees_usd = best.quote.expected_fees_usd
            decision.expected_rebate_usd = (best.quote.rebate_bps / 10_000) * target_allocation
            decision.metadata.update(best.quote.extras)
            decisions.append(decision)
        return decisions


__all__ = ["LiquidityProvisionStrategy"]
