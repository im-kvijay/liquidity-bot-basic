# Dashboard Operations Guide

The dashboard runs alongside the trading engine and exposes a read-only view of the bot's health, PnL, routing decisions, and
compliance status. It is implemented as a FastAPI application with WebSocket streaming for live events.

## Launching the dashboard

```bash
python -m solana_liquidity_bot.dashboard --host 0.0.0.0 --port 8080
```

### Authentication

* **Read-only token** – set `DASHBOARD__READ_ONLY_TOKEN` or the corresponding environment variable to enforce a shared secret.
  The UI accepts the token via the `X-Auth-Token` header, query string (`?token=`), or the input field at the top of the page.
* **Auth modes** – `DashboardAuthMode.AUTHENTICATED` requires a token even when `read_only_token` is unset. The default
  `READ_ONLY` mode exposes metrics without authentication.

### API surface

| Endpoint | Method | Description |
| --- | --- | --- |
| `/api/metrics` | GET | Metrics snapshot, inventory exposure map, and the configured risk disclaimer. |
| `/api/fills` | GET | Recent fills (live and dry-run) with venue, slippage, and allocation metadata. |
| `/api/router-decisions` | GET | Router audit trail including venue scores and expected slippage. |
| `/api/pnl` | GET | Historical PnL snapshots for downstream analysis. |
| `/api/events` | GET | Structured event history (fills, rejects, rebalances, health). |
| `/api/token-controls` | GET | Current allow/deny/pause decisions with reasons, sources, and timestamps. |
| `/metrics` | GET | Prometheus-compatible exposition with counters, gauges, and percentile summaries. |
| `/ws/events` | WS | Pushes structured events in real time for the UI or external consumers. |

## UI overview

The single-page application renders the following sections:

1. **Overview** – headline PnL metrics, queue depth, latency percentiles, rejection rate, and the configured risk disclaimer.
2. **Per-market drill-down** – active positions with allocation, venue, and unrealised PnL percentage.
3. **Recent fills** – latest fills (dry and live) with notional sizes and venues.
4. **Inventory skew** – USD notional exposure per mint.
5. **Compliance controls** – allow/deny/pause decisions sourced from the token registry with reasons and provenance.
6. **Router decisions** – top recent routing choices including scores and slippage estimates.
7. **Log excerpts** – streaming event list pulled from the internal event bus.

## Operations checklist

* Refresh cadence is configurable via `dashboard.metrics_refresh_interval_seconds` (default 7s in the UI script).
* All endpoints respect authentication middleware; failed attempts return HTTP 401/4403 with no payload.
* Metrics snapshots are sourced from `MetricsRegistry.snapshot()` and the latest persisted PnL entry to keep the UI responsive
  even when the trading loop is temporarily idle.
* The dashboard serves the same Prometheus feed as `/metrics`, so external monitoring can reuse the HTTP server without an
  additional exporter.
* Event WebSockets leverage a thread-safe queue; ensure consumers close connections gracefully to avoid dangling listeners.

> **Reminder:** The dashboard exposes sensitive operational data. When deploying to production networks, route traffic through an
> authenticated reverse proxy and enable TLS termination.
