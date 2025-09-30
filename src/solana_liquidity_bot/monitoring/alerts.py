"""Alerting utilities for Slack, Sentry, and generic webhooks."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Dict, Optional

import requests

from ..config.settings import MonitoringConfig, get_app_config
from .logger import get_logger


class AlertSeverity(str, Enum):
    """Common severity levels recognised by the alert manager."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertManager:
    """Dispatch alerts to configured endpoints with throttling."""

    def __init__(
        self,
        config: Optional[MonitoringConfig] = None,
        *,
        session: Optional[requests.Session] = None,
    ) -> None:
        self._config = config or get_app_config().monitoring
        self._session = session or requests.Session()
        self._logger = get_logger(__name__)
        self._last_sent: Dict[str, float] = {}

    def send(
        self,
        message: str,
        *,
        severity: AlertSeverity = AlertSeverity.INFO,
        key: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        key = key or message
        now = time.monotonic()
        throttle = max(self._config.alert_throttle_seconds, 0)
        last = self._last_sent.get(key)
        if last is not None and now - last < throttle:
            return
        self._last_sent[key] = now
        payload = {
            "message": message,
            "severity": severity.value,
            "extra": extra or {},
        }
        if self._config.slack_webhook_url:
            self._post(self._config.slack_webhook_url, {"text": f"[{severity.value.upper()}] {message}"})
        for url in self._config.webhook_urls:
            self._post(url, payload)
        if self._config.sentry_dsn:
            self._post(
                self._config.sentry_dsn,
                {
                    "level": severity.value,
                    "message": message,
                    "timestamp": time.time(),
                    "environment": self._config.sentry_environment,
                    "extra": extra or {},
                },
            )

    def _post(self, url: str, payload: Dict[str, Any]) -> None:
        try:
            response = self._session.post(url, json=payload, timeout=5)
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network failures
            self._logger.warning("Failed to send alert to %s: %s", url, exc)


def send_slack_message(message: str, config: Optional[MonitoringConfig] = None) -> None:
    """Backward compatible helper to send a basic Slack notification."""

    manager = AlertManager(config)
    manager.send(message, severity=AlertSeverity.INFO)


__all__ = ["AlertManager", "AlertSeverity", "send_slack_message"]
