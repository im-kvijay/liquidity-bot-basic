"""Coordinator that runs multiple strategies and applies risk gating."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Sequence

from ..datalake.schemas import StrategyDecision
from ..monitoring.event_bus import EVENT_BUS, EventType
from ..monitoring.logger import get_logger
from .base import Strategy, StrategyContext


class StrategyCoordinator:
    """Executes configured strategies and consolidates their decisions."""

    def __init__(self, strategies: Sequence[Strategy]) -> None:
        self._strategies = list(strategies)
        self._cooldowns: Dict[str, datetime] = {}
        self._logger = get_logger(__name__)

    def plan(self, context: StrategyContext) -> List[StrategyDecision]:
        context.risk_engine.update_exposures(context.pnl_engine.exposures())
        aggregated: List[StrategyDecision] = []
        for strategy in self._strategies:
            try:
                generated = strategy.generate(context)
            except Exception as exc:  # noqa: BLE001 - defensive logging
                self._logger.warning("Strategy %s failed: %s", strategy.name, exc)
                continue
            for decision in generated:
                if decision.action != "exit" and self._is_on_cooldown(
                    decision.token.mint_address, context.timestamp
                ):
                    continue
                entry = context.universe.get(decision.token.mint_address)
                risk_result = context.risk_engine.evaluate(decision, entry)
                if not risk_result.approved:
                    decision.notes.append("risk_blocked")
                    continue
                decision.cooldown_seconds = max(decision.cooldown_seconds, risk_result.throttle_seconds)
                aggregated.append(decision)

        aggregated.sort(key=lambda d: d.priority, reverse=True)
        selected: List[StrategyDecision] = []
        seen_tokens: set[str] = set()
        for decision in aggregated:
            if decision.token.mint_address in seen_tokens:
                continue
            selected.append(decision)
            if decision.action == "rebalance":
                EVENT_BUS.publish(
                    EventType.REBALANCE,
                    {
                        "mint": decision.token.mint_address,
                        "strategy": decision.strategy,
                        "allocation": decision.allocation,
                        "venue": decision.venue,
                    },
                    correlation_id=decision.correlation_id,
                )
            seen_tokens.add(decision.token.mint_address)
            if decision.cooldown_seconds > 0:
                self._cooldowns[decision.token.mint_address] = (
                    context.timestamp + timedelta(seconds=decision.cooldown_seconds)
                )
            if decision.action != "exit" and len(selected) >= context.max_decisions:
                break
        return selected

    def _is_on_cooldown(self, mint: str, now: datetime) -> bool:
        expiry = self._cooldowns.get(mint)
        if expiry is None:
            return False
        if now >= expiry:
            self._cooldowns.pop(mint, None)
            return False
        return True


__all__ = ["StrategyCoordinator"]
