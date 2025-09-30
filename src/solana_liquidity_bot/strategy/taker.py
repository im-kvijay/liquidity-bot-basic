"""Signal-driven taker strategy for opportunistic entries."""

from __future__ import annotations

from typing import List

from ..datalake.schemas import StrategyDecision
from ..monitoring.logger import get_logger
from ..config.settings import StrategyConfig, get_app_config
from .allocator import Allocator
from .base import Strategy, StrategyContext


class SignalTakerStrategy(Strategy):
    """Executes taker trades when short-term momentum and liquidity align."""

    name = "signal_taker"

    def __init__(
        self,
        allocator: Allocator,
        *,
        strategy_config: StrategyConfig | None = None,
        momentum_threshold: float | None = None,
    ) -> None:
        self._allocator = allocator
        self._config = strategy_config or get_app_config().strategy
        self._momentum_threshold = (
            momentum_threshold if momentum_threshold is not None else self._config.momentum_threshold
        )
        self._logger = get_logger(__name__)

    def generate(self, context: StrategyContext) -> List[StrategyDecision]:
        decisions: List[StrategyDecision] = []
        for score in context.scores.values():
            features = context.features.get(score.token.mint_address)
            entry = context.universe.get(score.token.mint_address)
            if features is None or entry is None:
                continue
            if (
                features.tvl_usd < self._config.min_liquidity_usd
                or features.volume_24h_usd < self._config.min_volume_24h_usd
                or features.fee_apr < self._config.min_fee_apr
                or features.holder_count < self._config.min_holder_count
                or features.price_impact_bps > self._config.price_impact_tolerance_bps
            ):
                continue
            momentum = self._compute_momentum(features)
            if momentum < self._momentum_threshold:
                continue
            if features.alpha_score < (self._config.alpha_score_target * 0.35):
                continue
            position = context.positions.get(score.token.mint_address)
            allocation = context.allocator.determine_allocation(score, position, strategy="taker")
            if allocation <= 0:
                continue
            risk_penalty = score.metrics.get("risk_penalty", 0.0)
            price_impact_penalty = score.metrics.get("price_impact_penalty", 0.0)
            alpha_boost = min(features.alpha_score / max(self._config.alpha_score_target, 1e-6), 2.0)
            priority = max(
                0.0,
                (score.score * momentum * alpha_boost)
                - risk_penalty
                - price_impact_penalty * 0.1,
            )
            decision = StrategyDecision(
                token=score.token,
                action="buy",
                allocation=allocation,
                priority=priority,
                strategy=self.name,
                side="taker",
                price_usd=features.price_usd,
                base_liquidity=features.base_liquidity,
                quote_liquidity=features.quote_liquidity,
                pool_fee_bps=features.pool_fee_bps,
                token_decimals=features.token_decimals or score.token.decimals,
                quote_token_decimals=features.quote_token_decimals or 0,
                correlation_id=f"{self.name}:{score.token.mint_address}",
                metadata={
                    "momentum": momentum,
                    "volume_24h_usd": features.volume_24h_usd,
                    "momentum_threshold": self._momentum_threshold,
                    "risk_penalty": risk_penalty,
                    "alpha_score": features.alpha_score,
                    "price_impact_bps": features.price_impact_bps,
                },
                notes=[
                    f"momentum={momentum:.2f}",
                    f"alpha={features.alpha_score:.2f}",
                    f"risk_pen={risk_penalty:.2f}",
                ],
            )
            evaluations = context.router.evaluate_routes(
                decision, entry, context.mode, limit=1
            )
            if not evaluations:
                continue
            best = evaluations[0]
            decision.venue = best.quote.venue
            decision.pool_address = best.quote.pool_address
            decision.quote_token_mint = best.quote.quote_mint
            decision.expected_value = best.score
            decision.expected_slippage_bps = best.quote.expected_slippage_bps
            decision.max_slippage_bps = min(
                context.risk_limits.max_slippage_bps,
                self._allocator.slippage_budget("taker"),
            )
            decision.expected_fees_usd = best.quote.expected_fees_usd
            decision.expected_rebate_usd = (best.quote.rebate_bps / 10_000) * allocation
            decision.expected_fill_probability = min(1.0, best.quote.depth_score)
            decision.metadata.update(best.quote.extras)
            decisions.append(decision)
        return decisions

    def _compute_momentum(self, features) -> float:
        liquidity_ratio = features.volume_24h_usd / max(features.tvl_usd, 1.0)
        fee_signal = features.fee_apr / max(self._config.fee_apr_target, 1e-6)
        velocity_signal = features.liquidity_velocity / max(self._config.liquidity_velocity_target, 1e-6)
        weight_sum = (
            self._config.momentum_liquidity_weight
            + self._config.momentum_fee_weight
            + self._config.momentum_velocity_weight
        )
        weight_sum = weight_sum or 1.0
        composite = (
            (liquidity_ratio * self._config.momentum_liquidity_weight)
            + (fee_signal * self._config.momentum_fee_weight)
            + (velocity_signal * self._config.momentum_velocity_weight)
        ) / weight_sum
        return min(max(composite, 0.0), 5.0)


__all__ = ["SignalTakerStrategy"]
