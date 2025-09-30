"""Simulation utilities for replaying historical data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from ..datalake.schemas import PortfolioPosition, StrategyDecision, TokenScore


@dataclass(slots=True)
class BacktestResult:
    """Represents the outcome of a strategy simulation."""

    total_return: float
    trades_executed: int
    positions_opened: int
    positions_closed: int


class Backtester:
    """Naive backtester that consumes historical scores and emits aggregated results."""

    def run(
        self,
        historical_scores: Iterable[List[TokenScore]],
        historical_positions: Iterable[List[PortfolioPosition]],
        decisions: Iterable[List[StrategyDecision]],
    ) -> BacktestResult:
        trades = sum(len(batch) for batch in decisions)
        entries = sum(1 for batch in decisions for decision in batch if decision.action == "enter")
        exits = sum(1 for batch in decisions for decision in batch if decision.action == "exit")
        total_return = sum(position.unrealized_pnl_pct for batch in historical_positions for position in batch)
        return BacktestResult(
            total_return=total_return,
            trades_executed=trades,
            positions_opened=entries,
            positions_closed=exits,
        )


__all__ = ["Backtester", "BacktestResult"]
