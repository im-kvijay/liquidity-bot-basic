"""Strategy interfaces and context objects."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Protocol, Sequence

from ..analysis.features import TokenFeatures
from ..analytics.pnl import PnLEngine
from ..config.settings import AppMode, RiskLimitsConfig, StrategyConfig
from ..datalake.schemas import PortfolioPosition, StrategyDecision, TokenScore, TokenUniverseEntry
from ..execution.router import OrderRouter
from ..ingestion.pricing import PriceOracle
from .allocator import Allocator
from .risk import RiskEngine


@dataclass(slots=True)
class StrategyContext:
    """Context supplied to each strategy during decision generation."""

    timestamp: datetime
    mode: AppMode
    scores: Mapping[str, TokenScore]
    features: Mapping[str, TokenFeatures]
    universe: Mapping[str, TokenUniverseEntry]
    positions: Mapping[str, PortfolioPosition]
    allocator: Allocator
    risk_engine: RiskEngine
    pnl_engine: PnLEngine
    router: OrderRouter
    price_oracle: PriceOracle
    strategy_config: StrategyConfig
    risk_limits: RiskLimitsConfig
    max_decisions: int


class Strategy(Protocol):
    """Protocol implemented by all concrete strategies."""

    name: str

    def generate(self, context: StrategyContext) -> Sequence[StrategyDecision]:
        """Produce strategy decisions for the supplied context."""


__all__ = ["Strategy", "StrategyContext"]
