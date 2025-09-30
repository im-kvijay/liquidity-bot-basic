# Migration Notes

This document summarises changes introduced in the observability/compliance release so operators can upgrade safely.

## Configuration changes

* Added `token_universe` fields for compliance guardrails (`deny_mints`, `deny_creators`, `deny_authorities`,
  `suspicious_keywords`, `min_decimals`, `max_decimals`, and AML thresholds). Review defaults in `config/app.toml` and override
  via environment variables before going live.
* Introduced `monitoring.risk_disclaimer` to surface a mandatory risk statement in dashboards, APIs, and logs.
* New `.env.example` entries document the additional toggles; regenerate your `.env` or append the missing keys.

## Exit lifecycle & PnL tracking (2024-XX-XX)

* Strategy configuration gained `exit_min_hold_seconds`, `exit_time_stop_seconds`, `exit_stale_profit_pct`, and
  `exit_reentry_cooldown_seconds` toggles (`STRATEGY__...` environment variables) to control exit cadence.
* Fills and positions now persist LP share metadata (`lp_token_amount`, `position_address`, etc.) so exits can withdraw
  the exact stake that was deposited. The SQLite backend alters existing tables automatically when the app starts—no
  manual migration steps are required.
* `SpreadMarketMakingStrategy` emits `exit` decisions when profit targets, stop losses, or time-based guards trigger.
  A new metric `strategy.exit.decisions` tracks the cadence of exit intent, and live queues now build DAMM/DLMM withdraw
  transactions when those decisions are approved.
* Added an `AggressiveMakerStrategy` that fills any gaps left by the score-based maker strategy. It targets the most
  liquid pools and routes deposits automatically, which ensures the bot always has positions to harvest fees from.
  Use the new `scripts/seed_position.py` helper to seed synthetic LP positions for dry-run validation, including LP token
  balances so the withdraw path can execute end-to-end.
* New `LaunchSniperStrategy` focuses on DAMM v2 launches with elevated fee schedules. It looks for pools younger than
  the configured window, applies holder-distribution heuristics, sizes entries using the three-tier SOL allocations, and
  tags exits for the 45-minute decay period. Discovery now records these launch candidates in `damm_launches`, and the
  shared helper rejects pools whose fee yield falls below the configured floor.

## Storage updates

* SQLite schema version remains `2`, but new compliance flags are persisted inside `token_risk_metrics.risk_flags` and
  `token_controls`. Existing databases will pick up the fields automatically—no manual migration is required.
* Router decisions, event logs, and metrics snapshots continue to populate the dedicated tables. Vacuum the database periodically
  (`sqlite3 state.sqlite3 'VACUUM;'`) if it grows beyond operational limits.

## Operational guidance

1. **Back up configuration and database** – copy `config/app.toml`, `.env`, and `state.sqlite3` prior to deploying the new build.
2. **Install pinned dependencies** – run `pip install -e .[dev]` and `(cd node_bridge && npm install)` to align with the new lock
   files and ensure deterministic runtime environments.
3. **Validate compliance output** – execute `python -m solana_liquidity_bot.main --dry-run` and inspect `/api/token-controls` to
   confirm deny rules behave as expected before enabling live trading.
4. **Regenerate dashboard assets** – the frontend now exposes compliance data and risk disclaimers; refresh any reverse proxies or
   CDN caches that serve the static HTML.

## Launch window enhancements (2025-XX-XX)

* Discovery now enriches each token with RocketScan cohort data (`dev_holding_pct`, `sniper_holding_pct`, `insider_holding_pct`,
  `bundler_holding_pct`) and persists it in the `token_risk_metrics` table. Existing databases are migrated automatically to schema
  version `3` during startup.
* The launch candidate filter enforces tighter ownership heuristics (top-10 ≤30%, dev supply ≤0.5%, sniper/insider/bundler cohorts
  below the configured caps) and honours Bonk launchpads when capping allocation tiers.
* `LaunchSniperStrategy` is phase aware (`hill` → aggressive entries, `cook` → measured scaling, `drift` → exits) and exits early on
  strength or if fees decay below the configured floors. Phase metadata is propagated into decisions for downstream analytics.
* `scripts/seed_position.py` defaults to the `launch_sniper` strategy label and accepts an optional `--phase-tag` flag so you can
  stage hill/cook/drift exits in dry-run mode without manual database edits.
* RocketScan enrichment now runs concurrently (`data_sources.rocketscan_max_workers`, default `8`) and skips tokens older than the
  configured freshness window (`data_sources.rocketscan_max_age_minutes`, default `60`). Tune these values if discovery latency is
  still too high in your environment.
* Price discovery now prefers the Jupiter Lite API before falling back to Pyth. Use `data_sources.price_oracle_use_pyth_fallback`
  (default `true`) and `data_sources.price_oracle_jupiter_rate_limit_per_minute` (default `55`) to adjust behaviour or honour your
  own API rate limits.
* Launch filters also require base pool fees of at least 6 % (`strategy.launch.min_base_fee_bps`) and relax the early market-cap and
  fee thresholds so high-fee meteors stay eligible during the first few minutes of trading.
* Launch filters allow a wider window of opportunity: defaults now accept liquidity up to 450k USD, relax minimum fee checks to 10%
  (`min_current_fee_bps=1000`), and raise the early-stage market-cap ceiling to 400k within the first 12 minutes. Override
  `strategy.launch.*` fields to tailor risk tolerance per playbook.

## Rollback plan

* Restore the previous Git revision and reinstall dependencies.
* Replace `state.sqlite3` with the backup to revert token control decisions if the compliance engine was too aggressive.
* Re-apply older `.env` and `config/app.toml` files.

> **Reminder:** Always test migrations on a staging environment that mirrors production RPC providers and signing infrastructure
> before releasing to mainnet.
