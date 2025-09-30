"""Internal event bus for structured observability events."""

from __future__ import annotations

import asyncio
import inspect
import logging
import queue
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Union

from .metrics import MetricsRegistry

try:  # Optional import used for type checking to avoid circulars at runtime
    from typing import TYPE_CHECKING
except ImportError:  # pragma: no cover - typing module always available in supported versions
    TYPE_CHECKING = False  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..datalake.storage import SQLiteStorage

from ..datalake.schemas import EventLogRecord
from .alerts import AlertManager, AlertSeverity


class EventType(str, Enum):
    """Supported event categories emitted by the bot."""

    FILL = "fill"
    CANCEL = "cancel"
    REJECT = "reject"
    REBALANCE = "rebalance"
    HEALTH = "health"
    ROUTER = "router"


class EventSeverity(str, Enum):
    """Severity levels associated with events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(slots=True)
class Event:
    """Normalized representation of an observability event."""

    type: EventType
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    severity: EventSeverity = EventSeverity.INFO
    correlation_id: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the event to a JSON-serialisable dictionary."""

        return {
            "type": self.type.value,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "correlation_id": self.correlation_id,
            "labels": self.labels,
        }


Subscriber = Callable[[Event], Union[None, Any]]


class EventBus:
    """Threaded event bus that fans out structured events to subscribers."""

    def __init__(self, history_size: int = 500) -> None:
        self._queue: "queue.Queue[Event]" = queue.Queue()
        self._subscribers: Dict[Optional[EventType], List[Subscriber]] = defaultdict(list)
        self._listeners: List["queue.SimpleQueue[Event]"] = []
        self._history: Deque[Event] = deque(maxlen=history_size)
        self._lock = threading.RLock()
        self._logger = logging.getLogger(__name__)
        self._metrics: Optional[MetricsRegistry] = None
        self._alerts: Optional[AlertManager] = None
        self._storage: Optional["SQLiteStorage"] = None
        self._worker = threading.Thread(target=self._run, name="event-bus", daemon=True)
        self._worker.start()

    def attach_metrics(self, registry: MetricsRegistry) -> None:
        self._metrics = registry

    def attach_alert_manager(self, manager: Optional[AlertManager]) -> None:
        self._alerts = manager

    def attach_storage(self, storage: Optional["SQLiteStorage"]) -> None:
        self._storage = storage

    def subscribe(self, event_type: Optional[EventType], handler: Subscriber) -> None:
        """Register a subscriber for a specific event type or all events."""

        with self._lock:
            self._subscribers[event_type].append(handler)

    def create_listener(self) -> "queue.SimpleQueue[Event]":
        """Create a queue listener that receives every dispatched event."""

        listener: "queue.SimpleQueue[Event]" = queue.SimpleQueue()
        with self._lock:
            self._listeners.append(listener)
        return listener

    def remove_listener(self, listener: "queue.SimpleQueue[Event]") -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def publish(
        self,
        event_type: Union[EventType, str],
        payload: Optional[Dict[str, Any]] = None,
        *,
        severity: EventSeverity = EventSeverity.INFO,
        correlation_id: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Publish a new event onto the bus."""

        if isinstance(event_type, str):
            try:
                event_type = EventType(event_type)
            except ValueError as exc:  # pragma: no cover - defensive branch
                raise ValueError(f"Unsupported event type: {event_type}") from exc
        event = Event(
            type=event_type,
            payload=dict(payload or {}),
            severity=severity,
            correlation_id=correlation_id,
            labels=dict(labels or {}),
        )
        self._queue.put(event)

    def history(self, limit: int = 100) -> List[Event]:
        with self._lock:
            return list(self._history)[-limit:]

    def flush(self, timeout: float = 1.0) -> bool:
        """Best-effort wait for the queue to drain."""

        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._queue.unfinished_tasks == 0:
                return True
            time.sleep(0.01)
        return self._queue.unfinished_tasks == 0

    def reset(self) -> None:
        """Clear subscribers and history. Intended for tests."""

        with self._lock:
            self._subscribers.clear()
            self._listeners.clear()
            self._history.clear()
        self._metrics = None
        self._alerts = None
        self._storage = None

    def _run(self) -> None:
        while True:
            event = self._queue.get()
            try:
                self._dispatch(event)
            except Exception:  # pragma: no cover - defensive logging
                self._logger.exception("Failed to dispatch event %s", event.type.value)
            finally:
                self._queue.task_done()

    def _dispatch(self, event: Event) -> None:
        with self._lock:
            self._history.append(event)
            handlers = list(self._subscribers.get(event.type, [])) + list(
                self._subscribers.get(None, [])
            )
            listeners = list(self._listeners)
        self._update_metrics(event)
        self._persist_event(event)
        self._trigger_alerts(event)
        for handler in handlers:
            try:
                result = handler(event)
                if inspect.isawaitable(result):
                    asyncio.run(result)
            except Exception:  # pragma: no cover - subscriber failures should never break dispatch
                self._logger.exception(
                    "Event handler %s failed for %s", getattr(handler, "__name__", handler), event.type.value
                )
        for listener in listeners:
            try:
                listener.put_nowait(event)
            except Exception:  # pragma: no cover - SimpleQueue.put_nowait rarely fails
                continue

    def _update_metrics(self, event: Event) -> None:
        if not self._metrics:
            return
        name = f"events.{event.type.value}"
        self._metrics.increment(name, 1.0)
        if event.type == EventType.FILL:
            slippage = float(event.payload.get("slippage_bps", 0.0) or 0.0)
            latency = event.payload.get("latency_seconds")
            self._metrics.observe("execution_slippage_bps", slippage)
            if latency is not None:
                self._metrics.observe("execution_latency_seconds", float(latency))
        if event.type == EventType.REJECT:
            reason = event.payload.get("reason")
            if isinstance(reason, str):
                self._metrics.increment(f"risk_reject_reason.{reason}", 1.0)
        if event.type == EventType.HEALTH and "queue_depth" in event.payload:
            try:
                self._metrics.gauge(
                    "transaction_queue_depth", float(event.payload.get("queue_depth", 0.0))
                )
            except (TypeError, ValueError):  # pragma: no cover - defensive
                pass

    def _persist_event(self, event: Event) -> None:
        if not self._storage:
            return
        record = EventLogRecord(
            timestamp=event.timestamp,
            event_type=event.type.value,
            severity=event.severity.value,
            payload=event.payload,
            correlation_id=event.correlation_id,
            labels=event.labels,
        )
        try:
            self._storage.record_event_log(record)
        except Exception:  # pragma: no cover - persistence failures should be non-fatal
            self._logger.exception("Failed to persist event log for %s", event.type.value)

    def _trigger_alerts(self, event: Event) -> None:
        if not self._alerts:
            return
        if event.severity in {EventSeverity.WARNING, EventSeverity.ERROR, EventSeverity.CRITICAL}:
            summary = event.payload.get("message") or event.payload
            message = f"{event.type.value.upper()}: {summary}"
            key = f"{event.type.value}:{event.payload.get('reason', '')}"
            self._alerts.send(
                message,
                severity=AlertSeverity(event.severity.value),
                key=key,
                extra=event.payload,
            )


EVENT_BUS = EventBus()


__all__ = [
    "EVENT_BUS",
    "EventBus",
    "Event",
    "EventType",
    "EventSeverity",
]
