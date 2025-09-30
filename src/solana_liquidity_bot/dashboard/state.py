"""Shared dashboard state and data access helpers."""

from __future__ import annotations

import queue
from typing import Dict, List

from ..config.settings import AppConfig
from ..datalake.storage import SQLiteStorage
from ..monitoring.event_bus import EVENT_BUS, EventBus
from ..monitoring.metrics import METRICS, MetricsRegistry
from ..monitoring.logger import get_logger
from .utils import to_serializable


class DashboardState:
    """Lightweight wrapper around storage, metrics, and the event bus."""

    def __init__(
        self,
        *,
        config: AppConfig,
        storage: SQLiteStorage,
        metrics: MetricsRegistry = METRICS,
        event_bus: EventBus = EVENT_BUS,
    ) -> None:
        self.config = config
        self.storage = storage
        self.metrics = metrics
        self.event_bus = event_bus
        self._logger = get_logger(__name__)

    def metrics_snapshot(self) -> Dict[str, object]:
        snapshot = self.metrics.snapshot()
        pnl = self.storage.list_pnl_snapshots(limit=1)
        latest_pnl = to_serializable(pnl[0]) if pnl else None
        return {
            "metrics": snapshot,
            "latest_pnl": latest_pnl,
            "disclaimer": self.config.monitoring.risk_disclaimer,
        }

    def positions(self) -> List[Dict[str, object]]:
        return [to_serializable(position) for position in self.storage.list_positions()]

    def fills(self, limit: int = 100) -> List[Dict[str, object]]:
        return [to_serializable(fill) for fill in self.storage.list_fills(limit=limit)]

    def router_decisions(self, limit: int = 100) -> List[Dict[str, object]]:
        return [
            to_serializable(record)
            for record in self.storage.list_router_decisions(limit=limit)
        ]

    def pnl_history(self, limit: int = 200) -> List[Dict[str, object]]:
        return [to_serializable(item) for item in self.storage.list_pnl_snapshots(limit=limit)]

    def event_history(self, limit: int = 200) -> List[Dict[str, object]]:
        return [event.to_dict() for event in self.event_bus.history(limit)]

    def token_controls(self) -> List[Dict[str, object]]:
        return [
            to_serializable(decision) for decision in self.storage.list_token_controls()
        ]

    def subscribe_events(self) -> "queue.SimpleQueue":
        return self.event_bus.create_listener()

    def remove_listener(self, listener: "queue.SimpleQueue") -> None:
        try:
            self.event_bus.remove_listener(listener)
        except Exception:  # pragma: no cover - defensive cleanup
            self._logger.debug("Listener cleanup failed", exc_info=True)


__all__ = ["DashboardState"]
