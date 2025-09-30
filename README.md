# Solana Early-Coin Liquidity Bot

This repository contains a comprehensive reference implementation that follows the ten-phase framework for building a Solana bot capable of identifying promising early-stage tokens (e.g., via Axiom, PumpFun) and earning liquidity fees from DAMM v2 pools.

The codebase is intentionally modular and testable. Each phase in the framework maps to a set of packages, documented workflows, and reproducible scripts so that teams can customize the logic for their particular strategy.

> **Risk disclaimer:** Trading digital assets is highly speculative and can result in substantial losses. The bot's guardrails reduce risk but do not guarantee profits or prevent all exploits. Conduct independent due diligence and never deploy capital you cannot afford to lose.

## Key Capabilities

* **Live discovery sources** – Axiom and PumpFun REST clients with configurable authentication and retry logic, plus a Jupiter-backed price oracle.
* **On-chain enrichment** – Automated Solana RPC probes capture mint metadata, holder distributions, and recent activity with graceful fallbacks when data is missing.
* **DAMM v2 pool integration** – A dedicated client ingests pool TVL, fees, and 24h volume so that strategies can reason about liquidity depth and potential fee APR in real time.
* **Risk-aware scoring and allocation** – Enhanced feature engineering feeds a configurable scoring engine, risk filters, and a capital allocator that scales position sizes based on liquidity, fees, and conviction.
* **Multi-venue routing & async execution** – A pluggable venue adapter stack provides per-pool quotes for Meteora DAMM v2 and DLMM, while an asynchronous transaction queue introduces backpressure, retries with jitter, and persistence for safe restarts.
* **Real-time PnL and observability** – The analytics engine computes realized/unrealized PnL with per-venue attribution, emits structured metrics, and exposes deterministic dry-run diagnostics to triage performance regressions.
* **Production ergonomics** – Resilient RPC handling, environment-driven configuration via Pydantic settings, and persistent storage for discoveries, fills, and open positions.
* **Rich observability surface** – Structured JSON logging with correlation IDs, an internal event bus that fan-outs fills/rebalances/health signals, Prometheus-compatible metrics, and a real-time dashboard served via FastAPI.

## Project Structure

```
├── src/solana_liquidity_bot
│   ├── analysis/           # Feature extraction and scoring logic.
│   ├── analytics/          # PnL accounting, exposure tracking, and attribution utilities.
│   ├── config/             # Configuration management and defaults.
│   ├── datalake/           # Storage abstractions (historical, state).
│   ├── execution/          # Transaction builders and Solana RPC client helpers.
│   ├── ingestion/          # Axiom/PumpFun connectors and Solana log listeners.
│   ├── monitoring/         # Logging, alerting, and metrics utilities.
│   ├── strategy/           # Risk controls, allocators, and backtests.
│   ├── tests/              # Unit tests and example fixtures.
│   └── main.py             # Orchestrates the end-to-end bot workflow.
├── pyproject.toml          # Packaging, dependencies, lint/test configuration.
├── README.md               # Project overview (this file).
└── .gitignore              # Development environment hygiene.
```

## Getting Started

1. **Install dependencies**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
(cd node_bridge && npm install)
```

2. **Configure environment variables and profiles**

Copy `.env.example` to `.env` and adjust values to match your infrastructure (RPC endpoints, signing keys, observability hooks). Environment variables can override any nested setting using the `SECTION__FIELD` convention shown in the template.

Runtime defaults now live in `config/app.toml`. The file ships with `default`, `dry_run`, `live`, and `testnet` profiles—select one via `BOT_MODE` or override any key inline. Environment variables take precedence over the profile file so sensitive credentials never need to be written to disk. The new `[execution]` section controls queue capacity, concurrency, and retry behaviour so you can tune backpressure without editing code.

3. **Run the unit tests**

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
```

4. **Launch the orchestrator**

The entry point script discovers real candidates, enriches them with live metrics (RPC, DAMM pools, and price feeds), and produces trading decisions. Use `--dry-run` to avoid submitting transactions:

```bash
python -m solana_liquidity_bot.main --dry-run
```

For live execution remove the `--dry-run` flag. Ensure that your wallet holds sufficient balances of both the base token and its quote asset so the Node-backed DLMM helper can construct the deposit transactions.

### Observability & Dashboard

* **Dashboard** – Serve the control panel locally with:

  ```bash
  python -m solana_liquidity_bot.dashboard --host 0.0.0.0 --port 8080
  ```

  The UI streams events over WebSockets, renders overview metrics (PnL, queue depth, latency percentiles, rejection rates), exposes per-market drill downs, and lists recent fills/router decisions. Set `DASHBOARD__READ_ONLY_TOKEN` to require a bearer token (supplied via the `X-Auth-Token` header or query string).

* **Prometheus endpoint** – Metrics are available at `/metrics` (enabled by default) and include latency histograms, slippage distributions, exposure gauges, and router selection counts.

* **Event bus** – Structured events (`fill`, `rebalance`, `cancel`, `reject`, `health`) are persisted to SQLite and surfaced through the dashboard API (`/api/events`) for alerting or replay.

### Configuration quick reference

* `config/app.toml` – declarative defaults for RPC timeouts, venue tuning, risk limits, and dashboard settings. Profiles are merged with the `default` section so shared values only need to be declared once.
* `.env` – secrets and environment-specific overrides (RPC URLs, API keys, Prometheus, Slack). See `.env.example` for a complete list of supported keys.
* `BOT_MODE` – selects which profile to apply (`dry_run`, `live`, or `testnet`). Dry mode never signs or broadcasts transactions but still executes the full discovery/risk pipeline for deterministic replay.

### Strategy, risk, and PnL overview

* **Strategies** – The `strategy` package now exposes a `StrategyCoordinator` with three production-ready implementations: spread-based market making, volatility-aware LP rebalancing, and a momentum-informed taker. Strategies request venue diagnostics through the router before emitting decisions and attach deterministic correlation IDs for tracing.
* **Risk engine** – `RiskEngine` enforces global/per-market notionals, slippage caps, and circuit-breaker style kill switches. The engine is fed by the PnL analytics module so daily loss, drawdown, and rejection rates automatically trigger protective cool-downs.
* **PnL analytics** – `analytics/pnl.py` tracks realized/unrealized performance per venue, pair, and strategy. Snapshots are persisted to SQLite for dashboards, and dry-run executions emit the same metrics for debugging without signing transactions.

### Token controls and risk monitoring

The token registry tracks every discovered mint, its on-chain health metrics, and the latest allow/deny status. Use the CLI to manage overrides:

```bash
python -m solana_liquidity_bot.ingestion.token_controls list
python -m solana_liquidity_bot.ingestion.token_controls allow <MINT> --reason "Manual approval"
python -m solana_liquidity_bot.ingestion.token_controls pause <MINT> --reason "Liquidity too thin"
```

Automated checks pause or deny tokens that lack oracle coverage, fall below liquidity/volume thresholds, or retain dangerous authorities. Decisions, liquidity metrics, fills, and PnL snapshots are persisted in SQLite for dashboard consumption and historical analysis.

The compliance engine supplements these controls with configuration-driven deny lists (mints, creators, and authorities),
keyword filters for known scams, and AML-style heuristics that flag suspicious holder concentration, sanctioned signers, and
freshly minted assets. Review `config/app.toml` and `.env.example` before enabling live trading to tailor these guardrails to
your risk appetite.

### Documentation & operational runbooks

The `docs/` directory contains supplemental material:

* `docs/dashboard.md` – dashboard setup, authentication, and API reference.
* `docs/agents.md` – responsibilities, dependencies, and failure modes for each service plus sequence diagrams.
* `docs/migration-notes.md` – configuration changes, storage migrations, and upgrade guidance.
* `docs/references.md` – curated links covering Meteora DAMM v2/DLMM, Solana RPC best practices, and compliance resources.
* `docs/samples/` – example metrics snapshots, persisted event logs, and a schematic dashboard overview (`dashboard-overview.svg`).

## Extending the Bot

* Layer in additional discovery signals (e.g., sentiment feeds or bespoke analytics) by extending `DiscoveryService`.
* Swap the storage backend for PostgreSQL or Redis by building adapters in `datalake/storage.py`.
  The `StorageAdapter` protocol documents the persistence contract used by the execution pipeline, router, and dashboard so a
  high-throughput database can be introduced without refactoring business logic.
* Fine-tune risk parameters in `StrategyConfig` and `RiskLimitsConfig` to align with your capital allocation, fee targets, and diversification rules.
* Adjust execution tuning in `[execution]` to match your RPC throughput envelope or extend `TransactionQueue` with venue-specific batching.
* Wire metrics to Prometheus/Grafana or alerting systems by enhancing the utilities in `monitoring/metrics.py` and `monitoring/alerts.py`.

## License

MIT License
