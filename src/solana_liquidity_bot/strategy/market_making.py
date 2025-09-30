"""Spread-based market making strategy for Meteora venues."""

from __future__ import annotations

from datetime import timedelta
from typing import List, Optional

from ..config.settings import StrategyConfig, get_app_config
from ..datalake.schemas import StrategyDecision
from ..monitoring.logger import get_logger
from ..monitoring.metrics import METRICS
from .allocator import Allocator
from .base import Strategy, StrategyContext


class SpreadMarketMakingStrategy(Strategy):
    """Places adaptive maker liquidity across DAMM and DLMM pools."""

    name = "spread_mm"

    def __init__(
        self,
        allocator: Allocator,
        strategy_config: StrategyConfig | None = None,
    ) -> None:
        self._allocator = allocator
        self._config = strategy_config or get_app_config().strategy
        self._logger = get_logger(__name__)

    def generate(self, context: StrategyContext) -> List[StrategyDecision]:
        decisions: List[StrategyDecision] = []
        exit_candidates = self._plan_exit_decisions(context)
        decisions.extend(exit_candidates)
        exit_mints = {decision.token.mint_address for decision in exit_candidates}
        scores = sorted(context.scores.values(), key=lambda item: item.score, reverse=True)
        max_candidates = min(context.strategy_config.max_candidates, len(scores))
        for score in scores[:max_candidates]:
            entry = context.universe.get(score.token.mint_address)
            if entry is None:
                continue
            if score.token.mint_address in exit_mints:
                continue
            position = context.positions.get(score.token.mint_address)
            allocation = context.allocator.determine_allocation(score, position, strategy="maker")
            if allocation <= 0:
                continue
            volatility = entry.risk.volatility_score if entry.risk else 0.5
            inventory_skew = 0.0
            if position and position.base_quantity > 0:
                inventory_skew = min(max(position.unrealized_pnl_pct / 100.0, -0.5), 0.5)
            metrics = score.metrics
            velocity_component = metrics.get("liquidity_velocity_component", 0.0)
            dynamic_fee_component = metrics.get("dynamic_fee_component", 0.0)
            volatility_component = metrics.get("volatility_component", 0.0)
            alpha_component = metrics.get("alpha_component", 0.0)
            liquidity_edge = min(max(velocity_component, 0.0), 1.0) * self._config.maker_liquidity_edge_bps
            fee_edge = min(max(dynamic_fee_component, 0.0), 1.2) * (self._config.maker_liquidity_edge_bps * 0.6)
            spread_bps = (
                self._config.maker_min_spread_bps
                + (volatility * self._config.maker_volatility_spread_factor)
                + abs(inventory_skew) * self._config.maker_inventory_spread_factor
                - liquidity_edge
                - fee_edge
            )
            spread_bps = max(
                self._config.maker_min_spread_bps,
                min(spread_bps, self._config.maker_max_spread_bps),
            )
            momentum_signal = metrics.get("momentum_signal", 0.0)
            profit_component = metrics.get("profit_component", 0.0)
            price_impact_penalty = metrics.get("price_impact_penalty", 0.0)
            priority = max(
                0.0,
                score.score
                + (momentum_signal * self._config.momentum_priority_weight)
                + (alpha_component * 0.05)
                + (profit_component * 0.05)
                - score.metrics.get("risk_penalty", 0.0)
                - price_impact_penalty * 0.1,
            )
            decision = StrategyDecision(
                token=score.token,
                action="rebalance" if position else "enter",
                allocation=allocation,
                priority=priority,
                strategy=self.name,
                side="maker",
                price_usd=score.metrics.get("price_usd") or (
                    entry.liquidity_event.price_usd if entry.liquidity_event else None
                ),
                base_liquidity=score.metrics.get("base_liquidity"),
                quote_liquidity=score.metrics.get("quote_liquidity"),
                pool_fee_bps=int(score.metrics.get("pool_fee_bps", 30)),
                token_decimals=int(score.metrics.get("token_decimals", score.token.decimals)),
                quote_token_decimals=int(score.metrics.get("quote_token_decimals", 0)),
                correlation_id=f"{self.name}:{score.token.mint_address}",
                metadata={
                    "target_spread_bps": spread_bps,
                    "inventory_skew": inventory_skew,
                    "volatility_score": volatility,
                    "liquidity_edge_bps": liquidity_edge,
                    "dynamic_fee_edge_bps": fee_edge,
                    "momentum_signal": momentum_signal,
                    "momentum_bonus": score.metrics.get("momentum_bonus", 0.0),
                    "risk_penalty": score.metrics.get("risk_penalty", 0.0),
                    "alpha_component": alpha_component,
                    "profit_component": profit_component,
                    "price_impact_penalty": price_impact_penalty,
                },
                notes=[
                    f"score={score.score:.2f}",
                    f"spread={spread_bps:.1f}bps",
                    f"vol={volatility:.2f}",
                    f"dyn_edge={fee_edge:.1f}",
                    f"mom={momentum_signal:.2f}",
                    f"risk_pen={score.metrics.get('risk_penalty', 0.0):.2f}",
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
                self._allocator.slippage_budget("maker"),
            )
            decision.expected_fees_usd = best.quote.expected_fees_usd
            decision.expected_rebate_usd = (best.quote.rebate_bps / 10_000) * allocation
            decision.expected_fill_probability = min(1.0, best.quote.liquidity_score)
            decision.metadata.update(best.quote.extras)
            decisions.append(decision)
        return decisions

    def _plan_exit_decisions(self, context: StrategyContext) -> List[StrategyDecision]:
        config = self._config
        results: List[StrategyDecision] = []
        take_profit_pct = self._as_percent(config.take_profit_pct)
        stop_loss_pct = self._as_percent(config.stop_loss_pct)
        stale_profit_pct = self._as_percent(config.exit_stale_profit_pct)
        min_hold = timedelta(seconds=max(config.exit_min_hold_seconds, 0))
        time_stop = timedelta(seconds=max(config.exit_time_stop_seconds, 0))
        now = context.timestamp
        for mint, position in context.positions.items():
            if position.base_quantity <= 0:
                continue
            entry = context.universe.get(mint)
            score = context.scores.get(mint)
            price = self._resolve_mark(price_hint=score.metrics.get("price_usd") if score else None,
                                       entry=entry,
                                       position=position,
                                       context=context)
            if price is None or price <= 0:
                continue
            pnl_pct = ((price / position.entry_price) - 1.0) * 100.0 if position.entry_price > 0 else 0.0
            age = now - position.created_at
            exit_reason: Optional[str] = None
            if pnl_pct >= take_profit_pct:
                exit_reason = "take_profit"
            elif pnl_pct <= -stop_loss_pct:
                exit_reason = "stop_loss"
            elif time_stop.total_seconds() > 0 and age >= time_stop:
                exit_reason = "time_lock"
            elif age >= max(min_hold, timedelta(seconds=0)) and pnl_pct <= -stale_profit_pct:
                exit_reason = "stale_negative"
            elif age >= max(time_stop, min_hold) and pnl_pct > 0:
                exit_reason = "time_guard"
            if exit_reason is None:
                continue
            allocation = position.base_quantity * price
            if allocation <= 0:
                continue
            quote_decimals = self._resolve_quote_decimals(score, entry)
            pool_address = position.pool_address
            if not pool_address and entry and entry.liquidity_event:
                pool_address = entry.liquidity_event.pool_address
            if not pool_address:
                self._logger.debug("Skipping exit for %s; pool unknown", mint)
                continue
            venue = position.venue
            if not venue and entry and entry.liquidity_event:
                venue = entry.liquidity_event.source
            if not venue:
                venue = "damm"
            quote_mint = None
            if score is not None:
                quote_mint = score.metrics.get("quote_token_mint")
            if not quote_mint and entry and entry.liquidity_event:
                quote_mint = entry.liquidity_event.quote_token_mint
            decision = StrategyDecision(
                token=position.token,
                action="exit",
                allocation=allocation,
                priority=1_000.0 + abs(pnl_pct),
                strategy=self.name,
                side="exit",
                price_usd=price,
                token_decimals=position.token.decimals,
                quote_token_decimals=quote_decimals,
                pool_address=pool_address,
                venue=venue,
                quote_token_mint=quote_mint,
                correlation_id=f"{self.name}:exit:{mint}",
                expected_value=allocation,
                expected_slippage_bps=0.0,
                max_slippage_bps=self._allocator.slippage_budget("maker"),
                metadata={
                    "exit_reason": exit_reason,
                    "unrealized_pct": pnl_pct,
                    "exit_base_quantity": position.base_quantity,
                    "exit_quote_quantity": allocation,
                },
                notes=[f"exit={exit_reason}", f"pnl={pnl_pct:.2f}%", f"age={age.total_seconds():.0f}s"],
                cooldown_seconds=config.exit_reentry_cooldown_seconds,
                position_snapshot=position,
            )
            if position.lp_token_amount:
                decision.metadata["lp_token_amount"] = float(position.lp_token_amount)
            METRICS.increment("strategy.exit.decisions", 1)
            results.append(decision)
        return results

    def _resolve_mark(
        self,
        *,
        price_hint: Optional[float],
        entry,
        position,
        context: StrategyContext,
    ) -> Optional[float]:
        if price_hint and price_hint > 0:
            return price_hint
        if position.last_mark_price and position.last_mark_price > 0:
            return position.last_mark_price
        entry_price = None
        if entry and entry.liquidity_event and entry.liquidity_event.price_usd:
            entry_price = entry.liquidity_event.price_usd
        if entry_price and entry_price > 0:
            return entry_price
        oracle_price = context.price_oracle.get_price(position.token.mint_address)
        if oracle_price and oracle_price > 0:
            return oracle_price
        return position.entry_price if position.entry_price > 0 else None

    def _resolve_quote_decimals(self, score, entry) -> int:
        if score is not None:
            decimals = score.metrics.get("quote_token_decimals")
            if decimals:
                return int(decimals)
        if entry and entry.liquidity_event and entry.liquidity_event.quote_token_mint:
            return 6
        return 6

    @staticmethod
    def _as_percent(value: float) -> float:
        return value if value >= 1.0 else value * 100.0


__all__ = ["SpreadMarketMakingStrategy"]
