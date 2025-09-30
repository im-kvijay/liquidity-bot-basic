"""Dashboard application factory with optional FastAPI dependency."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional
from urllib.parse import parse_qs

from ..config.settings import DashboardAuthMode
from ..monitoring.metrics import METRICS
from .state import DashboardState

HTML_TEMPLATE = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Solana Liquidity Bot Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; padding: 0; background: #111; color: #f5f5f5; }
    header { padding: 16px 24px; background: #1f1f1f; box-shadow: 0 2px 8px rgba(0,0,0,0.5); }
    h1 { margin: 0; font-size: 24px; }
    main { padding: 24px; display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 24px; }
    section { background: #1a1a1a; border-radius: 12px; padding: 16px; box-shadow: 0 2px 10px rgba(0,0,0,0.3); }
    h2 { margin-top: 0; font-size: 20px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 6px 8px; border-bottom: 1px solid #333; text-align: left; }
    tr:nth-child(even) { background: #242424; }
    code { background: #222; padding: 2px 4px; border-radius: 4px; }
    #logs { max-height: 260px; overflow-y: auto; font-family: monospace; font-size: 13px; }
    #events { list-style: none; padding: 0; margin: 0; }
    .status-ok { color: #6adc6a; }
    .status-warn { color: #f0c674; }
    .status-err { color: #f2777a; }
    .token-input { margin-top: 8px; width: 100%; padding: 6px; border-radius: 6px; border: none; background: #202020; color: #f5f5f5; }
    .disclaimer { font-size: 12px; color: #aaa; margin-top: 12px; line-height: 1.4; }
  </style>
</head>
<body>
  <header>
    <h1>Solana Liquidity Bot Dashboard</h1>
    <label>Read-only token (optional):<br/><input id=\"auth-token\" class=\"token-input\" placeholder=\"Paste token to unlock\" /></label>
  </header>
  <main>
    <section>
      <h2>Overview</h2>
      <div id=\"overview\"></div>
    </section>
    <section>
      <h2>Per-Market Drill-down</h2>
      <div id=\"per-market\"></div>
    </section>
    <section>
      <h2>Recent Fills</h2>
      <div id=\"fills\"></div>
    </section>
    <section>
      <h2>Inventory Skew</h2>
      <div id=\"inventory\"></div>
    </section>
    <section>
      <h2>Compliance Controls</h2>
      <div id=\"controls\"></div>
    </section>
    <section>
      <h2>Router Decisions</h2>
      <div id=\"router\"></div>
    </section>
    <section>
      <h2>Log Excerpts</h2>
      <div id=\"logs\"><ul id=\"events\"></ul></div>
    </section>
  </main>
  <script>
    const overviewEl = document.getElementById('overview');
    const perMarketEl = document.getElementById('per-market');
    const fillsEl = document.getElementById('fills');
    const inventoryEl = document.getElementById('inventory');
    const controlsEl = document.getElementById('controls');
    const routerEl = document.getElementById('router');
    const eventsEl = document.getElementById('events');
    const tokenInput = document.getElementById('auth-token');

    const storedToken = window.localStorage.getItem('dashboard_token');
    if (storedToken) { tokenInput.value = storedToken; }
    tokenInput.addEventListener('change', () => {
      const value = tokenInput.value.trim();
      if (value) {
        window.localStorage.setItem('dashboard_token', value);
      } else {
        window.localStorage.removeItem('dashboard_token');
      }
      connectWebSocket();
      refreshAll();
    });

    function authHeaders() {
      const token = tokenInput.value.trim();
      return token ? { 'X-Auth-Token': token } : {};
    }

    async function fetchJson(url) {
      const response = await fetch(url, { headers: authHeaders() });
      if (!response.ok) {
        throw new Error(`Request failed: ${response.status}`);
      }
      return await response.json();
    }

    function escapeHtml(value) {
      return value.replace(/[&<>"']/g, (char) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
      })[char] || char);
    }

    function formatUsd(value) {
      return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 2 }).format(value || 0);
    }

    async function refreshOverview() {
      try {
        const data = await fetchJson('/api/metrics');
        const metrics = data.metrics.gauges || {};
        const pnl = data.latest_pnl || {};
        const rejectionRate = data.metrics.gauges?.risk_rejection_rate || 0;
        const latency = data.metrics.histograms?.router_latency_ms?.p90 || 0;
        const disclaimer = data.disclaimer ? `<p class="disclaimer">${escapeHtml(data.disclaimer)}</p>` : '';
        overviewEl.innerHTML = `
          <p><strong>Realized PnL:</strong> ${formatUsd(pnl.realized_usd || 0)}</p>
          <p><strong>Unrealized PnL:</strong> ${formatUsd(pnl.unrealized_usd || 0)}</p>
          <p><strong>Inventory Value:</strong> ${formatUsd(metrics.inventory_value_usd || 0)}</p>
          <p><strong>Queue Depth:</strong> ${metrics.transaction_queue_depth || 0}</p>
          <p><strong>Risk Rejection Rate:</strong> ${(rejectionRate * 100).toFixed(2)}%</p>
          <p><strong>Router Latency p90:</strong> ${latency.toFixed(1)} ms</p>
          ${disclaimer}`;
      } catch (err) {
        overviewEl.innerHTML = `<span class="status-err">${err}</span>`;
      }
    }

    async function refreshPerMarket() {
      try {
        const positions = await fetchJson('/api/positions');
        if (!positions.length) {
          perMarketEl.innerHTML = '<em>No active positions</em>';
          return;
        }
        const rows = positions.map((pos) => `
          <tr>
            <td>${pos.token.symbol}</td>
            <td>${pos.venue || 'n/a'}</td>
            <td>${formatUsd(pos.allocation)}</td>
            <td>${pos.base_quantity.toFixed(4)}</td>
            <td>${pos.unrealized_pnl_pct.toFixed(2)}%</td>
          </tr>`).join('');
        perMarketEl.innerHTML = `<table><thead><tr><th>Token</th><th>Venue</th><th>Allocation</th><th>Base Qty</th><th>Unrealized %</th></tr></thead><tbody>${rows}</tbody></table>`;
      } catch (err) {
        perMarketEl.innerHTML = `<span class="status-err">${err}</span>`;
      }
    }

    async function refreshFills() {
      try {
        const fills = await fetchJson('/api/fills');
        const rows = fills.slice(0, 10).map((fill) => `
          <tr>
            <td>${fill.timestamp}</td>
            <td>${fill.token_symbol}</td>
            <td>${fill.venue}</td>
            <td>${fill.side}</td>
            <td>${formatUsd(fill.quote_quantity)}</td>
          </tr>`).join('');
        fillsEl.innerHTML = `<table><thead><tr><th>Time</th><th>Token</th><th>Venue</th><th>Side</th><th>Notional</th></tr></thead><tbody>${rows}</tbody></table>`;
      } catch (err) {
        fillsEl.innerHTML = `<span class="status-err">${err}</span>`;
      }
    }

    async function refreshInventory() {
      try {
        const metrics = await fetchJson('/api/metrics');
        const exposures = metrics.metrics.mappings?.inventory_exposure_usd || {};
        const rows = Object.entries(exposures).map(([mint, value]) => `
          <tr><td>${mint}</td><td>${formatUsd(value)}</td></tr>`).join('');
        inventoryEl.innerHTML = rows ? `<table><thead><tr><th>Mint</th><th>Exposure</th></tr></thead><tbody>${rows}</tbody></table>` : '<em>No exposure</em>';
      } catch (err) {
        inventoryEl.innerHTML = `<span class="status-err">${err}</span>`;
      }
    }

    async function refreshControls() {
      try {
        const controls = await fetchJson('/api/token-controls');
        if (!controls.length) {
          controlsEl.innerHTML = '<em>No recorded decisions</em>';
          return;
        }
        const rows = controls.slice(0, 50).map((item) => `
          <tr>
            <td>${escapeHtml(item.mint_address)}</td>
            <td>${escapeHtml(item.status)}</td>
            <td>${escapeHtml(item.source || '')}</td>
            <td>${escapeHtml(item.reason || '')}</td>
            <td>${escapeHtml(item.updated_at || '')}</td>
          </tr>`).join('');
        controlsEl.innerHTML = `<table><thead><tr><th>Mint</th><th>Status</th><th>Source</th><th>Reason</th><th>Updated</th></tr></thead><tbody>${rows}</tbody></table>`;
      } catch (err) {
        controlsEl.innerHTML = `<span class="status-err">${err}</span>`;
      }
    }

    async function refreshRouter() {
      try {
        const decisions = await fetchJson('/api/router-decisions');
        const rows = decisions.slice(0, 10).map((item) => `
          <tr>
            <td>${item.timestamp}</td>
            <td>${item.mint_address}</td>
            <td>${item.venue}</td>
            <td>${item.score.toFixed(3)}</td>
            <td>${item.slippage_bps.toFixed(1)} bps</td>
          </tr>`).join('');
        routerEl.innerHTML = `<table><thead><tr><th>Time</th><th>Mint</th><th>Venue</th><th>Score</th><th>Slippage</th></tr></thead><tbody>${rows}</tbody></table>`;
      } catch (err) {
        routerEl.innerHTML = `<span class="status-err">${err}</span>`;
      }
    }

    async function refreshEvents() {
      try {
        const events = await fetchJson('/api/events');
        eventsEl.innerHTML = events.slice(0, 50).map((event) => `<li>[${event.timestamp}] ${event.type.toUpperCase()} :: ${JSON.stringify(event.payload)}</li>`).join('');
      } catch (err) {
        eventsEl.innerHTML = `<li class="status-err">${err}</li>`;
      }
    }

    function refreshAll() {
      refreshOverview();
      refreshPerMarket();
      refreshFills();
      refreshInventory();
      refreshControls();
      refreshRouter();
      refreshEvents();
    }

    let ws;
    function connectWebSocket() {
      if (ws) {
        ws.close();
      }
      const token = tokenInput.value.trim();
      const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
      const base = `${protocol}://${window.location.host}`;
      const url = token ? `${base}/ws/events?token=${encodeURIComponent(token)}` : `${base}/ws/events`;
      ws = new WebSocket(url);
      ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          const entry = document.createElement('li');
          entry.textContent = `[${payload.timestamp}] ${payload.type.toUpperCase()} :: ${JSON.stringify(payload.payload)}`;
          eventsEl.prepend(entry);
          while (eventsEl.children.length > 200) {
            eventsEl.removeChild(eventsEl.lastChild);
          }
        } catch (err) {
          console.error('Failed to parse event', err);
        }
      };
      ws.onclose = () => {
        setTimeout(connectWebSocket, 5000);
      };
    }

    connectWebSocket();
    refreshAll();
    setInterval(refreshAll, 7000);
  </script>
</body>
</html>
"""

try:  # pragma: no cover - exercised when FastAPI is installed
    from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

    FASTAPI_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - fallback path used in tests
    FastAPI = None  # type: ignore[assignment]
    FASTAPI_AVAILABLE = False


if FASTAPI_AVAILABLE:

    def create_dashboard_app(state: DashboardState):
        app = FastAPI(title="Solana Liquidity Bot Dashboard", version="1.0.0")
        cfg = state.config.dashboard
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cfg.allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        def get_state() -> DashboardState:
            return state

        async def require_auth(
            token_header: Optional[str] = Header(default=None, alias="X-Auth-Token"),
            token_query: Optional[str] = Query(default=None, alias="token"),
            dashboard_state: DashboardState = Depends(get_state),
        ) -> Optional[str]:
            token = token_header or token_query or None
            auth_cfg = dashboard_state.config.dashboard
            expected = auth_cfg.read_only_token
            if expected:
                if token != expected:
                    raise HTTPException(status_code=401, detail="Invalid token")
            elif auth_cfg.auth_mode == DashboardAuthMode.AUTHENTICATED and not token:
                raise HTTPException(status_code=401, detail="Authentication required")
            return token

        @app.get("/", response_class=HTMLResponse)
        async def index() -> str:
            return HTML_TEMPLATE

        @app.get("/health")
        async def healthcheck() -> Dict[str, str]:
            return {"status": "ok"}

        @app.get("/metrics", response_class=PlainTextResponse)
        async def prometheus_metrics() -> str:
            return METRICS.export_prometheus()

        @app.get("/api/metrics")
        async def api_metrics(_: Optional[str] = Depends(require_auth)) -> JSONResponse:
            return JSONResponse(state.metrics_snapshot())

        @app.get("/api/fills")
        async def api_fills(
            limit: int = Query(100, ge=1, le=500),
            _: Optional[str] = Depends(require_auth),
        ) -> JSONResponse:
            return JSONResponse(state.fills(limit=limit))

        @app.get("/api/router-decisions")
        async def api_router_decisions(
            limit: int = Query(100, ge=1, le=500),
            _: Optional[str] = Depends(require_auth),
        ) -> JSONResponse:
            return JSONResponse(state.router_decisions(limit=limit))

        @app.get("/api/pnl")
        async def api_pnl(
            limit: int = Query(200, ge=1, le=500),
            _: Optional[str] = Depends(require_auth),
        ) -> JSONResponse:
            return JSONResponse(state.pnl_history(limit=limit))

        @app.get("/api/positions")
        async def api_positions(_: Optional[str] = Depends(require_auth)) -> JSONResponse:
            return JSONResponse(state.positions())

        @app.get("/api/events")
        async def api_events(
            limit: int = Query(200, ge=1, le=1000),
            _: Optional[str] = Depends(require_auth),
        ) -> JSONResponse:
            return JSONResponse(state.event_history(limit=limit))

        @app.get("/api/token-controls")
        async def api_token_controls(
            _: Optional[str] = Depends(require_auth),
        ) -> JSONResponse:
            return JSONResponse(state.token_controls())

        async def _ws_auth(websocket: WebSocket) -> None:
            token = websocket.headers.get("X-Auth-Token") or websocket.query_params.get("token")
            auth_cfg = state.config.dashboard
            expected = auth_cfg.read_only_token
            if expected:
                if token != expected:
                    await websocket.close(code=4403)
                    raise HTTPException(status_code=4403, detail="Invalid token")
            elif auth_cfg.auth_mode == DashboardAuthMode.AUTHENTICATED and not token:
                await websocket.close(code=4403)
                raise HTTPException(status_code=4403, detail="Authentication required")

        @app.websocket("/ws/events")
        async def ws_events(websocket: WebSocket) -> None:
            try:
                await _ws_auth(websocket)
            except HTTPException:
                return
            await websocket.accept()
            listener = state.subscribe_events()
            try:
                while True:
                    event = await asyncio.to_thread(listener.get)
                    await websocket.send_json(event.to_dict())
            except WebSocketDisconnect:
                pass
            finally:
                state.remove_listener(listener)

        return app

else:

    class _ASGIResponse:
        def __init__(self, body: bytes, status: int = 200, content_type: str = "application/json") -> None:
            self.body = body
            self.status = status
            self.content_type = content_type

        async def __call__(self, send) -> None:
            headers = [
                (b"content-type", self.content_type.encode()),
                (b"content-length", str(len(self.body)).encode()),
            ]
            await send({"type": "http.response.start", "status": self.status, "headers": headers})
            await send({"type": "http.response.body", "body": self.body})

    class _JSONResponse(_ASGIResponse):
        def __init__(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload, default=str).encode()
            super().__init__(body, status=status, content_type="application/json")

    class _PlainTextResponse(_ASGIResponse):
        def __init__(self, text: str, status: int = 200) -> None:
            super().__init__(text.encode(), status=status, content_type="text/plain; charset=utf-8")

    class _HTMLResponse(_ASGIResponse):
        def __init__(self, html: str, status: int = 200) -> None:
            super().__init__(html.encode(), status=status, content_type="text/html; charset=utf-8")

    class _MinimalDashboardApp:
        def __init__(self, state: DashboardState) -> None:
            self._state = state

        def _extract_token(self, headers: Dict[str, str], query: Dict[str, list[str]]) -> Optional[str]:
            if "x-auth-token" in headers:
                return headers["x-auth-token"]
            tokens = query.get("token")
            if tokens:
                return tokens[-1]
            return None

        def _authorise(self, headers: Dict[str, str], query: Dict[str, list[str]]) -> tuple[bool, Optional[str], str]:
            token = self._extract_token(headers, query)
            auth_cfg = self._state.config.dashboard
            expected = getattr(auth_cfg, "read_only_token", None)
            if expected:
                if token != expected:
                    return False, token, "Invalid token"
            elif auth_cfg.auth_mode == DashboardAuthMode.AUTHENTICATED and not token:
                return False, token, "Authentication required"
            return True, token, ""

        async def __call__(self, scope, receive, send) -> None:
            scope_type = scope.get("type")
            if scope_type == "lifespan":
                while True:
                    message = await receive()
                    message_type = message.get("type")
                    if message_type == "lifespan.startup":
                        await send({"type": "lifespan.startup.complete"})
                    elif message_type == "lifespan.shutdown":
                        await send({"type": "lifespan.shutdown.complete"})
                        return
                return

            if scope_type != "http":
                response = _PlainTextResponse("Not Found", status=404)
                await response(send)
                return

            headers = {key.decode().lower(): value.decode() for key, value in scope.get("headers", [])}
            query = parse_qs(scope.get("query_string", b"").decode())

            while True:
                message = await receive()
                if message.get("type") == "http.request" and not message.get("more_body", False):
                    break

            method = scope.get("method", "GET").upper()
            path = scope.get("path", "/")

            if method != "GET":
                response = _PlainTextResponse("Method Not Allowed", status=405)
                await response(send)
                return

            if path == "/":
                response = _HTMLResponse(HTML_TEMPLATE)
                await response(send)
                return

            if path == "/health":
                response = _JSONResponse({"status": "ok"})
                await response(send)
                return

            if path == "/metrics":
                response = _PlainTextResponse(METRICS.export_prometheus())
                await response(send)
                return

            authorised, _token, reason = self._authorise(headers, query)
            if not authorised:
                response = _JSONResponse({"detail": reason}, status=401)
                await response(send)
                return

            if path == "/api/metrics":
                response = _JSONResponse(self._state.metrics_snapshot())
            elif path == "/api/token-controls":
                response = _JSONResponse(self._state.token_controls())
            elif path == "/api/positions":
                response = _JSONResponse(self._state.positions())
            elif path == "/api/fills":
                limit = int(query.get("limit", ["100"])[-1])
                response = _JSONResponse(self._state.fills(limit=min(max(limit, 1), 500)))
            elif path == "/api/router-decisions":
                limit = int(query.get("limit", ["100"])[-1])
                response = _JSONResponse(self._state.router_decisions(limit=min(max(limit, 1), 500)))
            elif path == "/api/pnl":
                limit = int(query.get("limit", ["200"])[-1])
                response = _JSONResponse(self._state.pnl_history(limit=min(max(limit, 1), 500)))
            elif path == "/api/events":
                limit = int(query.get("limit", ["200"])[-1])
                response = _JSONResponse(self._state.event_history(limit=min(max(limit, 1), 1000)))
            else:
                response = _PlainTextResponse("Not Found", status=404)
            await response(send)

    def create_dashboard_app(state: DashboardState):
        return _MinimalDashboardApp(state)


__all__ = ["create_dashboard_app"]
