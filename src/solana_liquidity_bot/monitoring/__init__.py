"""Monitoring package exports and helpers."""

from __future__ import annotations

from typing import Optional

from ..config.settings import AppConfig, get_app_config
from ..datalake.storage import SQLiteStorage
from .alerts import AlertManager
from .event_bus import EVENT_BUS
from .logger import configure_logging
from .metrics import METRICS


def bootstrap_observability(
    storage: SQLiteStorage,
    *,
    config: Optional[AppConfig] = None,
) -> AlertManager:
    """Configure logging, event bus persistence, and alert routing."""

    app_config = config or get_app_config()
    configure_logging(app_config.monitoring)
    manager = AlertManager(app_config.monitoring)
    EVENT_BUS.attach_metrics(METRICS)
    EVENT_BUS.attach_alert_manager(manager)
    EVENT_BUS.attach_storage(storage)
    return manager


__all__ = ["bootstrap_observability", "EVENT_BUS", "METRICS"]
