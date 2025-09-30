"""Risk management and circuit breaker enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

from ..analytics.pnl import ExposureSummary, PnLEngine
from ..config.settings import CircuitBreakerConfig, RiskLimitsConfig, get_app_config
from ..datalake.schemas import PortfolioPosition, StrategyDecision, TokenUniverseEntry
from ..monitoring.event_bus import EVENT_BUS, EventSeverity, EventType
from ..monitoring.logger import get_logger
from ..monitoring.metrics import METRICS


@dataclass(slots=True)
class RiskCheckResult:
    """Outcome of a pre-trade risk assessment."""

    approved: bool
    reasons: List[str]
    throttle_seconds: int = 0


class RiskEngine:
    """Evaluates decisions against configured limits and circuit breakers."""

    def __init__(
        self,
        config: Optional[RiskLimitsConfig] = None,
        breaker: Optional[CircuitBreakerConfig] = None,
        *,
        pnl_engine: Optional[PnLEngine] = None,
    ) -> None:
        app_config = get_app_config()
        self._config = config or app_config.risk
        self._breaker = breaker or app_config.circuit_breakers
        self._pnl_engine = pnl_engine
        self._exposures: Dict[str, ExposureSummary] = {}
        self._rejections = 0
        self._decisions = 0
        self._kill_switch_reason: Optional[str] = None
        self._last_breaker_trip: Optional[datetime] = None
        self._logger = get_logger(__name__)

    def bind_pnl_engine(self, pnl_engine: PnLEngine) -> None:
        self._pnl_engine = pnl_engine

    def update_positions(self, positions: Iterable[PortfolioPosition]) -> None:
        self._exposures = {
            position.token.mint_address: ExposureSummary(
                mint_address=position.token.mint_address,
                notional_usd=position.base_quantity * (position.last_mark_price or position.entry_price),
                base_quantity=position.base_quantity,
                last_price=position.last_mark_price or position.entry_price,
            )
            for position in positions
        }

    def update_exposures(self, exposures: Dict[str, ExposureSummary]) -> None:
        self._exposures = exposures

    def evaluate(
        self,
        decision: StrategyDecision,
        entry: Optional[TokenUniverseEntry],
    ) -> RiskCheckResult:
        if decision.action == "exit":
            return RiskCheckResult(approved=True, reasons=[])
        reasons: List[str] = []
        allocation = max(decision.allocation, 0.0)
        global_notional = sum(abs(item.notional_usd) for item in self._exposures.values())
        if global_notional + allocation > self._config.max_global_notional_usd:
            reasons.append("global_notional_limit")
        exposure = self._exposures.get(decision.token.mint_address)
        market_notional = abs(exposure.notional_usd) if exposure else 0.0
        if market_notional + allocation > self._config.max_market_notional_usd:
            reasons.append("market_notional_limit")
        if allocation > self._config.max_position_notional_usd:
            reasons.append("position_notional_limit")
        if decision.max_slippage_bps > self._config.max_slippage_bps:
            reasons.append("slippage_limit")
        if entry and entry.risk:
            if entry.risk.volatility_score > 1.0 and decision.side not in {"lp", "maker"}:
                reasons.append("volatility_limit")
            if entry.risk.liquidity_usd < max(allocation, 0.0) * 2:
                reasons.append("insufficient_liquidity")
            if not entry.risk.has_oracle_price:
                reasons.append("oracle_unavailable")
        if self._pnl_engine and -self._pnl_engine.daily_loss() > self._config.daily_loss_limit_usd:
            reasons.append("daily_loss_limit")
        if self._kill_switch_reason:
            reasons.append(f"kill_switch:{self._kill_switch_reason}")

        approved = not reasons
        self._record_decision(approved)
        throttle = 0
        if not approved:
            throttle = self._compute_throttle()
            METRICS.increment("risk_rejections", 1)
            EVENT_BUS.publish(
                EventType.REJECT,
                {
                    "mint": decision.token.mint_address,
                    "reason": ",".join(reasons) if reasons else "unknown",
                    "allocation": allocation,
                    "strategy": decision.strategy,
                    "throttle_seconds": throttle,
                },
                severity=EventSeverity.WARNING,
                correlation_id=decision.correlation_id,
            )
        if approved:
            METRICS.increment("risk_approved", 1)
        else:
            if self._should_trip_breaker():
                reasons.append("circuit_breaker")
                approved = False
        return RiskCheckResult(approved=approved, reasons=reasons, throttle_seconds=throttle)

    def engage_kill_switch(self, reason: str) -> None:
        if not self._kill_switch_reason:
            self._kill_switch_reason = reason
            self._last_breaker_trip = datetime.now(timezone.utc)
            self._logger.error("Kill switch engaged: %s", reason)
            EVENT_BUS.publish(
                EventType.HEALTH,
                {"message": reason, "kill_switch": True},
                severity=EventSeverity.CRITICAL,
            )

    def reset_kill_switch(self) -> None:
        self._kill_switch_reason = None
        EVENT_BUS.publish(
            EventType.HEALTH,
            {"message": "kill_switch_reset"},
            severity=EventSeverity.INFO,
        )

    @property
    def kill_switch_engaged(self) -> bool:
        return self._kill_switch_reason is not None

    def _record_decision(self, approved: bool) -> None:
        self._decisions += 1
        if not approved:
            self._rejections += 1
        if self._decisions:
            ratio = self._rejections / self._decisions
            METRICS.gauge("risk_rejection_rate", ratio)

    def _compute_throttle(self) -> int:
        ratio = (self._rejections / self._decisions) if self._decisions else 0.0
        if ratio > self._breaker.max_reject_rate:
            return self._breaker.cooldown_seconds
        return 0

    def _should_trip_breaker(self) -> bool:
        if self._pnl_engine:
            snapshot = self._pnl_engine.snapshot(persist=False)
            if snapshot.drawdown_pct >= self._breaker.max_drawdown_pct:
                self.engage_kill_switch("max_drawdown")
                return True
        ratio = (self._rejections / self._decisions) if self._decisions else 0.0
        if ratio > self._breaker.max_reject_rate:
            self.engage_kill_switch("reject_rate")
            return True
        return False


__all__ = ["RiskEngine", "RiskCheckResult"]
