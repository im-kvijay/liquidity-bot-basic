"""Entry point for launching the dashboard server."""

from __future__ import annotations

import argparse

import uvicorn

from ..config.settings import get_app_config
from ..datalake.storage import SQLiteStorage
from ..monitoring import bootstrap_observability
from ..monitoring.event_bus import EVENT_BUS
from ..monitoring.metrics import METRICS
from .app import create_dashboard_app
from .state import DashboardState


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Solana liquidity bot dashboard")
    parser.add_argument("--host", help="Override dashboard host")
    parser.add_argument("--port", type=int, help="Override dashboard port")
    args = parser.parse_args()

    config = get_app_config()
    storage = SQLiteStorage(config.storage.database_path)
    bootstrap_observability(storage, config=config)
    state = DashboardState(config=config, storage=storage, metrics=METRICS, event_bus=EVENT_BUS)
    app = create_dashboard_app(state)
    host = args.host or config.dashboard.host
    port = args.port or config.dashboard.port
    uvicorn.run(app, host=host, port=port, log_level=config.monitoring.log_level.lower())


if __name__ == "__main__":
    main()
