"""Structured logging helpers with correlation ID support."""

from __future__ import annotations

import json
import logging
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from ..config.settings import MonitoringConfig, get_app_config

_LOGGER_CACHE: Dict[str, logging.Logger] = {}
_CORRELATION_ID: ContextVar[str] = ContextVar("correlation_id", default="-")
_LOGGING_CONFIGURED = False

_STANDARD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())


class _CorrelationFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - simple accessor
        record.correlation_id = _CORRELATION_ID.get("-")
        return True


class StructuredFormatter(logging.Formatter):
    """Formatter that emits structured JSON logs."""

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - formatting logic
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", "-"),
        }
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_ATTRS and not key.startswith("_")
        }
        if extras:
            payload["extra"] = extras
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(config: Optional[MonitoringConfig] = None) -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    cfg = config or get_app_config().monitoring
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.addFilter(_CorrelationFilter())
    root.setLevel(getattr(logging, cfg.log_level.upper(), logging.INFO))
    logging.captureWarnings(True)
    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _LOGGING_CONFIGURED:
        configure_logging()
    if name not in _LOGGER_CACHE:
        _LOGGER_CACHE[name] = logging.getLogger(name)
    return _LOGGER_CACHE[name]


@contextmanager
def correlation_scope(correlation_id: Optional[str]):
    token = _CORRELATION_ID.set(correlation_id or "-")
    try:
        yield
    finally:
        _CORRELATION_ID.reset(token)


__all__ = ["get_logger", "configure_logging", "correlation_scope", "StructuredFormatter"]
