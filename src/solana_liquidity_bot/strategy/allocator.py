"""Capital allocation utilities shared across strategies."""

from __future__ import annotations

from typing import Iterable, List, Optional

from ..config.settings import StrategyConfig, get_app_config
from ..datalake.schemas import PortfolioPosition, StrategyDecision, TokenScore


class Allocator:
    """Capital allocator that adapts to inventory skew and strategy type."""

    def __init__(self, strategy_config: StrategyConfig | None = None) -> None:
        self._config = strategy_config or get_app_config().strategy

    def plan_entries(self, scores: Iterable[TokenScore]) -> List[StrategyDecision]:
        results: List[StrategyDecision] = []
        capital_remaining = self._config.portfolio_size
        for score in scores:
            allocation = self.determine_allocation(score)
            if allocation <= 0:
                continue
            if allocation > capital_remaining:
                continue
            metrics = score.metrics
            results.append(
                StrategyDecision(
                    token=score.token,
                    action="enter",
                    allocation=allocation,
                    priority=score.score,
                    pool_address=metrics.get("pool_address") or None,
                    quote_token_mint=metrics.get("quote_token_mint") or None,
                    price_usd=metrics.get("price_usd"),
                    token_decimals=int(metrics.get("token_decimals", score.token.decimals)),
                    quote_token_decimals=int(metrics.get("quote_token_decimals", 0)),
                    base_liquidity=metrics.get("base_liquidity"),
                    quote_liquidity=metrics.get("quote_liquidity"),
                    pool_fee_bps=int(metrics.get("pool_fee_bps", self._config.fee_apr_target * 10000)),
                )
            )
            capital_remaining -= allocation
            if len(results) >= self._config.max_positions:
                break
        return results

    def determine_allocation(
        self,
        score: TokenScore,
        position: Optional[PortfolioPosition] = None,
        strategy: str = "maker",
    ) -> float:
        # Respect fixed allocation override if configured
        fixed = getattr(self._config, "fixed_allocation_usd", None)
        if fixed is not None:
            return max(0.0, min(float(fixed), self._config.max_allocation_per_position))
        base = self._baseline_allocation(score)
        if position is not None and position.base_quantity > 0:
            price_usd = score.metrics.get("price_usd")
            if price_usd is None or price_usd <= 0:
                last_mark = position.last_mark_price
                if last_mark is not None and last_mark > 0:
                    price_usd = last_mark
                else:
                    price_usd = position.entry_price
            inventory_ratio = min(
                position.base_quantity * price_usd / max(self._config.portfolio_size, 1.0),
                1.0,
            )
            inventory_multiplier = max(0.4, 1.0 - inventory_ratio)
            if position.unrealized_pnl_pct < 0:
                inventory_multiplier *= 0.9
            base *= inventory_multiplier
        if strategy == "taker":
            base = min(base, self._config.max_allocation_per_position * 0.35)
        elif strategy == "lp":
            base = min(base, self._config.max_allocation_per_position * 1.1)
        return max(base, 0.0)

    def rebalance_allocation(self, position: PortfolioPosition, volatility_score: float) -> float:
        volatility_factor = max(0.4, min(1.2 - (volatility_score * 0.25), 1.0))
        target = position.allocation * volatility_factor
        return max(min(target, self._config.max_allocation_per_position), 0.0)

    def slippage_budget(self, strategy: str) -> int:
        if strategy == "maker":
            return int(self._config.maker_slippage_bps)
        if strategy == "lp":
            return int(self._config.lp_slippage_bps)
        if strategy == "taker":
            return int(self._config.taker_slippage_bps)
        return int(self._config.default_slippage_bps)

    def plan_exits(self, positions: Iterable[PortfolioPosition]) -> List[StrategyDecision]:
        return [
            StrategyDecision(token=position.token, action="exit", allocation=0.0)
            for position in positions
        ]

    def _baseline_allocation(self, score: TokenScore) -> float:
        metrics = score.metrics
        base_allocation = self._config.allocation_per_position
        liquidity_component = metrics.get("liquidity_component")
        if liquidity_component is not None:
            liquidity_multiplier = max(
                0.5,
                min(
                    liquidity_component * self._config.max_allocation_multiplier,
                    self._config.max_allocation_multiplier,
                ),
            )
        else:
            liquidity_multiplier = metrics.get("tvl_usd", 0.0) / max(
                self._config.liquidity_target_usd, 1.0
            )
            liquidity_multiplier = max(
                0.5, min(liquidity_multiplier, self._config.max_allocation_multiplier)
            )
        fee_multiplier = metrics.get("fee_apr", 0.0) / max(self._config.fee_apr_target, 1e-6)
        fee_multiplier = 1.0 + min(max(fee_multiplier, 0.0), 1.0) * 0.5
        velocity = metrics.get("liquidity_velocity", 0.0)
        velocity_target = max(self._config.liquidity_velocity_target, 1e-6)
        velocity_score = min(max(velocity / velocity_target, 0.0), self._config.max_allocation_multiplier)
        velocity_multiplier = 1.0 + (velocity_score * 0.25)
        momentum_signal = metrics.get("momentum_signal", 0.0)
        momentum_boost = max(momentum_signal - self._config.momentum_threshold, 0.0)
        momentum_multiplier = 1.0 + min(momentum_boost, 1.0) * 0.2
        risk_penalty = min(metrics.get("risk_penalty", 0.0), 0.6)
        risk_multiplier = max(0.4, 1.0 - risk_penalty)
        confidence_multiplier = max(0.5, min(score.score, 1.2))
        allocation = (
            base_allocation
            * liquidity_multiplier
            * fee_multiplier
            * confidence_multiplier
            * velocity_multiplier
            * momentum_multiplier
            * risk_multiplier
        )
        allocation = min(allocation, self._config.max_allocation_per_position)
        return allocation


__all__ = ["Allocator"]
