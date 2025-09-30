from __future__ import annotations

import asyncio
from pathlib import Path

from httpx import ASGITransport, AsyncClient

from ..config.settings import get_app_config
from ..datalake.storage import SQLiteStorage
from ..monitoring import bootstrap_observability
from ..monitoring.event_bus import EVENT_BUS, EventSeverity, EventType
from ..monitoring.metrics import METRICS
from ..dashboard import DashboardState, create_dashboard_app


def _detach_event_bus() -> None:
    EVENT_BUS.attach_storage(None)
    EVENT_BUS.attach_alert_manager(None)


def test_event_bus_persists_events(tmp_path: Path) -> None:
    METRICS.reset()
    storage = SQLiteStorage(tmp_path / "state.sqlite3")
    bootstrap_observability(storage)
    EVENT_BUS.publish(
        EventType.HEALTH,
        {"message": "heartbeat"},
        severity=EventSeverity.INFO,
        correlation_id="test",
    )
    EVENT_BUS.flush()
    logs = storage.list_event_logs(limit=1)
    assert logs
    assert logs[0].payload["message"] == "heartbeat"
    assert METRICS.get("events.health") == 1
    _detach_event_bus()


def test_sqlite_storage_migrations(tmp_path: Path) -> None:
    storage = SQLiteStorage(tmp_path / "state.sqlite3")
    # Creating and reading from the new tables should not raise
    assert storage.list_router_decisions() == []
    assert storage.list_event_logs() == []
    assert storage.list_metrics_snapshots() == []


def test_prometheus_export_sanitizes_metric_names() -> None:
    METRICS.reset()
    METRICS.increment("events.health")
    METRICS.increment("router.selected.Meteora", 2)
    METRICS.gauge("queue.depth", 3)
    METRICS.observe("router.latency.ms", 0.5)
    output = METRICS.export_prometheus()
    lines = [line for line in output.splitlines() if line]
    assert any(line.startswith("# TYPE events_health counter") for line in lines)
    assert "events.health" not in output
    assert any("router_selected_Meteora" in line for line in lines)
    assert any("queue_depth" in line for line in lines)
    assert any("router_latency_ms" in line for line in lines)
    METRICS.reset()


def test_dashboard_app_basic_endpoints(tmp_path: Path) -> None:
    storage = SQLiteStorage(tmp_path / "state.sqlite3")
    bootstrap_observability(storage)
    config = get_app_config()
    state = DashboardState(config=config, storage=storage)
    app = create_dashboard_app(state)

    async def _exercise() -> None:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"
            metrics_resp = await client.get("/api/metrics")
            assert metrics_resp.status_code == 200
            payload = metrics_resp.json()
            assert "metrics" in payload
            controls_resp = await client.get("/api/token-controls")
            assert controls_resp.status_code == 200
            assert controls_resp.json() == []

    asyncio.run(_exercise())
    _detach_event_bus()
