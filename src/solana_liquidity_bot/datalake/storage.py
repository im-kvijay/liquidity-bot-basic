"""Persistence layer that keeps track of discoveries and portfolio state."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Protocol

from .schemas import (
    DammLaunchRecord,
    EventLogRecord,
    FillEvent,
    MetricsSnapshot,
    PnLSnapshot,
    PortfolioPosition,
    RouterDecisionRecord,
    TokenControlDecision,
    TokenMetadata,
    TokenRiskMetrics,
)


class StorageAdapter(Protocol):
    """Interface describing storage backends (SQLite, Postgres, Redis, ...)."""

    def record_router_decision(self, record: RouterDecisionRecord) -> None:
        ...

    def list_router_decisions(
        self, limit: int = 200, mint_address: Optional[str] = None
    ) -> List[RouterDecisionRecord]:
        ...

    def record_event_log(self, event: EventLogRecord) -> None:
        ...

    def list_event_logs(
        self, limit: int = 200, event_type: Optional[str] = None
    ) -> List[EventLogRecord]:
        ...

    def persist_metrics_snapshot(self, snapshot: MetricsSnapshot) -> None:
        ...

    def list_metrics_snapshots(self, limit: int = 200) -> List[MetricsSnapshot]:
        ...


CREATE_TOKEN_TABLE = """
CREATE TABLE IF NOT EXISTS tokens (
    mint_address TEXT PRIMARY KEY,
    symbol TEXT,
    name TEXT,
    decimals INTEGER,
    creator TEXT,
    project_url TEXT
);
"""

CREATE_POSITION_TABLE = """
CREATE TABLE IF NOT EXISTS positions (
    mint_address TEXT PRIMARY KEY,
    pool_address TEXT,
    venue TEXT,
    allocation REAL,
    entry_price REAL,
    created_at TEXT,
    unrealized_pnl_pct REAL,
    position_address TEXT,
    position_secret TEXT,
    strategy TEXT,
    base_quantity REAL,
    quote_quantity REAL,
    realized_pnl_usd REAL,
    unrealized_pnl_usd REAL,
    fees_paid_usd REAL,
    rebates_earned_usd REAL,
    last_mark_price REAL,
    last_mark_timestamp TEXT,
    pool_fee_bps INTEGER,
    lp_token_amount INTEGER,
    lp_token_mint TEXT,
    peak_price REAL,
    peak_timestamp TEXT
);
"""

CREATE_TOKEN_CONTROL_TABLE = """
CREATE TABLE IF NOT EXISTS token_controls (
    mint_address TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    reason TEXT NOT NULL,
    source TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

CREATE_TOKEN_RISK_TABLE = """
CREATE TABLE IF NOT EXISTS token_risk_metrics (
    mint_address TEXT PRIMARY KEY,
    liquidity_usd REAL NOT NULL,
    volume_24h_usd REAL NOT NULL,
    volatility_score REAL NOT NULL,
    holder_count INTEGER NOT NULL,
    top_holder_pct REAL NOT NULL,
    top10_holder_pct REAL NOT NULL,
    dev_holding_pct REAL,
    sniper_holding_pct REAL,
    insider_holding_pct REAL,
    bundler_holding_pct REAL,
    has_oracle_price INTEGER NOT NULL,
    price_confidence_bps INTEGER NOT NULL,
    last_updated TEXT NOT NULL,
    risk_flags TEXT NOT NULL
);
"""

CREATE_FILL_TABLE = """
CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    mint_address TEXT NOT NULL,
    token_symbol TEXT NOT NULL,
    venue TEXT NOT NULL,
    action TEXT NOT NULL,
    side TEXT NOT NULL,
    base_quantity REAL NOT NULL,
    quote_quantity REAL NOT NULL,
    price_usd REAL NOT NULL,
    fee_usd REAL NOT NULL,
    rebate_usd REAL NOT NULL,
    expected_value REAL NOT NULL,
    slippage_bps REAL NOT NULL,
    strategy TEXT NOT NULL,
    correlation_id TEXT NOT NULL,
    signature TEXT,
    is_dry_run INTEGER NOT NULL,
    pool_address TEXT,
    quote_mint TEXT,
    pool_fee_bps INTEGER,
    lp_token_amount INTEGER,
    lp_token_mint TEXT,
    position_address TEXT,
    position_secret TEXT,
    expected_slippage_bps REAL,
    actual_slippage_bps REAL,
    expected_fee_usd REAL,
    expected_price_usd REAL,
    actual_price_usd REAL
);
"""

CREATE_PNL_SNAPSHOT_TABLE = """
CREATE TABLE IF NOT EXISTS pnl_snapshots (
    timestamp TEXT PRIMARY KEY,
    realized_usd REAL NOT NULL,
    unrealized_usd REAL NOT NULL,
    fees_usd REAL NOT NULL,
    rebates_usd REAL NOT NULL,
    inventory_value_usd REAL NOT NULL,
    net_exposure_usd REAL NOT NULL,
    drawdown_pct REAL NOT NULL,
    venue_breakdown TEXT NOT NULL,
    pair_breakdown TEXT NOT NULL,
    strategy_breakdown TEXT NOT NULL
);
"""


CREATE_SCHEMA_VERSION_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER NOT NULL
);
"""

CREATE_ROUTER_DECISION_TABLE = """
CREATE TABLE IF NOT EXISTS router_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    mint_address TEXT NOT NULL,
    venue TEXT NOT NULL,
    pool_address TEXT NOT NULL,
    score REAL NOT NULL,
    allocation_usd REAL NOT NULL,
    slippage_bps REAL NOT NULL,
    strategy TEXT NOT NULL,
    correlation_id TEXT,
    quote_mint TEXT,
    extras TEXT
);
"""

CREATE_EVENT_LOG_TABLE = """
CREATE TABLE IF NOT EXISTS event_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    payload TEXT NOT NULL,
    correlation_id TEXT,
    labels TEXT
);
"""

CREATE_METRICS_SNAPSHOT_TABLE = """
CREATE TABLE IF NOT EXISTS metrics_snapshots (
    timestamp TEXT NOT NULL,
    payload TEXT NOT NULL,
    labels TEXT,
    PRIMARY KEY (timestamp)
);
"""

CREATE_DAMM_LAUNCH_TABLE = """
CREATE TABLE IF NOT EXISTS damm_launches (
    mint_address TEXT NOT NULL,
    pool_address TEXT NOT NULL,
    fee_bps INTEGER NOT NULL,
    liquidity_usd REAL NOT NULL,
    volume_24h_usd REAL NOT NULL,
    fee_yield REAL NOT NULL,
    age_seconds REAL NOT NULL,
    price_usd REAL,
    market_cap_usd REAL,
    fee_scheduler_mode TEXT,
    fee_scheduler_current_bps INTEGER,
    fee_scheduler_start_bps INTEGER,
    fee_scheduler_min_bps INTEGER,
    allocation_cap_sol REAL,
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (mint_address, pool_address)
);
"""

SCHEMA_VERSION = 3


class SQLiteStorage:
    """Simple SQLite-backed storage for token metadata and portfolio positions."""

    def __init__(self, database_path: Path) -> None:
        # Validate database path to prevent path traversal attacks
        database_path = database_path.resolve()
        if not database_path.parent.exists():
            raise ValueError(f"Database directory does not exist: {database_path.parent}")
        if not database_path.parent.is_dir():
            raise ValueError(f"Database path is not a directory: {database_path.parent}")

        # Ensure the path is within an allowed directory (e.g., current working directory, temp directory, or home)
        allowed_bases = [Path.cwd(), Path.home()]
        import tempfile
        import os

        # Check for temp directory patterns in the path
        path_str = str(database_path)
        is_temp_path = False

        # Check if path contains temp directory indicators
        temp_indicators = ['/tmp/', '/temp/', 'pytest-of-']
        for indicator in temp_indicators:
            if indicator in path_str:
                is_temp_path = True
                break

        # Also check if it's within system temp directory (handling symlinks)
        if hasattr(tempfile, 'gettempdir'):
            temp_dir = Path(tempfile.gettempdir())
            try:
                # Use resolve to handle symlinks
                resolved_temp = temp_dir.resolve()
                resolved_path = database_path.parent.resolve()
                if resolved_path == resolved_temp or resolved_path.is_relative_to(resolved_temp):
                    is_temp_path = True
            except (OSError, ValueError):
                pass

        # If it's a temp path or within allowed directories, allow it
        if is_temp_path:
            pass  # Allow temp paths
        else:
            for base in allowed_bases:
                try:
                    database_path.relative_to(base)
                    break  # Found allowed base
                except ValueError:
                    continue
            else:
                raise ValueError(f"Database path must be within an allowed directory: {database_path}")

        self._database_path = database_path
        self._initialize()

    def _initialize(self) -> None:
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute(CREATE_TOKEN_TABLE)
            con.execute(CREATE_POSITION_TABLE)
            con.execute(CREATE_TOKEN_CONTROL_TABLE)
            con.execute(CREATE_TOKEN_RISK_TABLE)
            self._ensure_token_risk_columns(con)
            con.execute(CREATE_FILL_TABLE)
            con.execute(CREATE_PNL_SNAPSHOT_TABLE)
            con.execute(CREATE_SCHEMA_VERSION_TABLE)
            con.execute(CREATE_DAMM_LAUNCH_TABLE)
            self._ensure_position_columns(con)
            self._ensure_fill_columns(con)
            self._apply_migrations(con)
            con.commit()

    def _ensure_position_columns(self, con: sqlite3.Connection) -> None:
        columns = {row[1] for row in con.execute("PRAGMA table_info(positions)")}
        if "position_address" not in columns:
            con.execute("ALTER TABLE positions ADD COLUMN position_address TEXT")
        if "position_secret" not in columns:
            con.execute("ALTER TABLE positions ADD COLUMN position_secret TEXT")
        if "venue" not in columns:
            con.execute("ALTER TABLE positions ADD COLUMN venue TEXT")
        if "strategy" not in columns:
            con.execute("ALTER TABLE positions ADD COLUMN strategy TEXT")
        if "base_quantity" not in columns:
            con.execute("ALTER TABLE positions ADD COLUMN base_quantity REAL DEFAULT 0")
        if "quote_quantity" not in columns:
            con.execute("ALTER TABLE positions ADD COLUMN quote_quantity REAL DEFAULT 0")
        if "realized_pnl_usd" not in columns:
            con.execute("ALTER TABLE positions ADD COLUMN realized_pnl_usd REAL DEFAULT 0")
        if "unrealized_pnl_usd" not in columns:
            con.execute("ALTER TABLE positions ADD COLUMN unrealized_pnl_usd REAL DEFAULT 0")
        if "fees_paid_usd" not in columns:
            con.execute("ALTER TABLE positions ADD COLUMN fees_paid_usd REAL DEFAULT 0")
        if "rebates_earned_usd" not in columns:
            con.execute("ALTER TABLE positions ADD COLUMN rebates_earned_usd REAL DEFAULT 0")
        if "last_mark_price" not in columns:
            con.execute("ALTER TABLE positions ADD COLUMN last_mark_price REAL")
        if "last_mark_timestamp" not in columns:
            con.execute("ALTER TABLE positions ADD COLUMN last_mark_timestamp TEXT")
        if "pool_fee_bps" not in columns:
            con.execute("ALTER TABLE positions ADD COLUMN pool_fee_bps INTEGER")
        if "lp_token_amount" not in columns:
            con.execute("ALTER TABLE positions ADD COLUMN lp_token_amount INTEGER DEFAULT 0")
        if "lp_token_mint" not in columns:
            con.execute("ALTER TABLE positions ADD COLUMN lp_token_mint TEXT")

    def _ensure_fill_columns(self, con: sqlite3.Connection) -> None:
        columns = {row[1] for row in con.execute("PRAGMA table_info(fills)")}
        if "pool_address" not in columns:
            con.execute("ALTER TABLE fills ADD COLUMN pool_address TEXT")
        if "quote_mint" not in columns:
            con.execute("ALTER TABLE fills ADD COLUMN quote_mint TEXT")
        if "pool_fee_bps" not in columns:
            con.execute("ALTER TABLE fills ADD COLUMN pool_fee_bps INTEGER")
        if "lp_token_amount" not in columns:
            con.execute("ALTER TABLE fills ADD COLUMN lp_token_amount INTEGER DEFAULT 0")
        if "lp_token_mint" not in columns:
            con.execute("ALTER TABLE fills ADD COLUMN lp_token_mint TEXT")
        if "position_address" not in columns:
            con.execute("ALTER TABLE fills ADD COLUMN position_address TEXT")
        if "position_secret" not in columns:
            con.execute("ALTER TABLE fills ADD COLUMN position_secret TEXT")

    def _ensure_token_risk_columns(self, con: sqlite3.Connection) -> None:
        columns = {row[1] for row in con.execute("PRAGMA table_info(token_risk_metrics)")}
        if "dev_holding_pct" not in columns:
            con.execute("ALTER TABLE token_risk_metrics ADD COLUMN dev_holding_pct REAL")
        if "sniper_holding_pct" not in columns:
            con.execute("ALTER TABLE token_risk_metrics ADD COLUMN sniper_holding_pct REAL")
        if "insider_holding_pct" not in columns:
            con.execute("ALTER TABLE token_risk_metrics ADD COLUMN insider_holding_pct REAL")
        if "bundler_holding_pct" not in columns:
            con.execute("ALTER TABLE token_risk_metrics ADD COLUMN bundler_holding_pct REAL")

    def _apply_migrations(self, con: sqlite3.Connection) -> None:
        current = self._get_schema_version(con)
        if current == 0:
            self._set_schema_version(con, 1)
            current = 1
        if current < 2:
            self._migrate_to_v2(con)
            current = 2
        if current < 3:
            self._migrate_to_v3(con)
            current = 3
        if current != SCHEMA_VERSION:
            self._set_schema_version(con, SCHEMA_VERSION)

    def _get_schema_version(self, con: sqlite3.Connection) -> int:
        cur = con.execute("SELECT version FROM schema_migrations ORDER BY ROWID DESC LIMIT 1")
        row = cur.fetchone()
        if row is None:
            return 0
        try:
            return int(row[0])
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return 0

    def _set_schema_version(self, con: sqlite3.Connection, version: int) -> None:
        con.execute("DELETE FROM schema_migrations")
        con.execute("INSERT INTO schema_migrations (version) VALUES (?)", (version,))

    def _migrate_to_v2(self, con: sqlite3.Connection) -> None:
        con.execute(CREATE_ROUTER_DECISION_TABLE)
        con.execute(CREATE_EVENT_LOG_TABLE)
        con.execute(CREATE_METRICS_SNAPSHOT_TABLE)

    def _migrate_to_v3(self, con: sqlite3.Connection) -> None:
        self._ensure_token_risk_columns(con)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self._database_path)
        try:
            yield con
        finally:
            con.close()

    def upsert_token(self, metadata: TokenMetadata) -> None:
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO tokens (mint_address, symbol, name, decimals, creator, project_url)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(mint_address) DO UPDATE SET
                    symbol = excluded.symbol,
                    name = excluded.name,
                    decimals = excluded.decimals,
                    creator = excluded.creator,
                    project_url = excluded.project_url
                """,
                (
                    metadata.mint_address,
                    metadata.symbol,
                    metadata.name,
                    metadata.decimals,
                    metadata.creator,
                    metadata.project_url,
                ),
            )
            con.commit()

    def list_tokens(self) -> List[TokenMetadata]:
        with self._connect() as con:
            cur = con.execute("SELECT mint_address, symbol, name, decimals, creator, project_url FROM tokens ORDER BY rowid DESC")
            rows = cur.fetchall()
        return [
            TokenMetadata(
                mint_address=row[0],
                symbol=row[1],
                name=row[2],
                decimals=row[3],
                creator=row[4],
                project_url=row[5],
            )
            for row in rows
        ]

    def get_token(self, mint_address: str) -> Optional[TokenMetadata]:
        with self._connect() as con:
            cur = con.execute(
                "SELECT mint_address, symbol, name, decimals, creator, project_url FROM tokens WHERE mint_address = ?",
                (mint_address,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return TokenMetadata(
            mint_address=row[0],
            symbol=row[1],
            name=row[2],
            decimals=row[3],
            creator=row[4],
            project_url=row[5],
        )

    def upsert_token_control(self, decision: TokenControlDecision) -> None:
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO token_controls (mint_address, status, reason, source, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(mint_address) DO UPDATE SET
                    status = excluded.status,
                    reason = excluded.reason,
                    source = excluded.source,
                    updated_at = excluded.updated_at
                """,
                (
                    decision.mint_address,
                    decision.status,
                    decision.reason,
                    decision.source,
                    decision.updated_at.isoformat(),
                ),
            )
            con.commit()

    def get_token_control(self, mint_address: str) -> Optional[TokenControlDecision]:
        with self._connect() as con:
            cur = con.execute(
                "SELECT mint_address, status, reason, source, updated_at FROM token_controls WHERE mint_address = ?",
                (mint_address,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return TokenControlDecision(
            mint_address=row[0],
            status=row[1],
            reason=row[2],
            source=row[3],
            updated_at=datetime.fromisoformat(row[4]),
        )

    def list_token_controls(self) -> List[TokenControlDecision]:
        with self._connect() as con:
            cur = con.execute(
                "SELECT mint_address, status, reason, source, updated_at FROM token_controls ORDER BY updated_at DESC"
            )
            rows = cur.fetchall()
        return [
            TokenControlDecision(
                mint_address=row[0],
                status=row[1],
                reason=row[2],
                source=row[3],
                updated_at=datetime.fromisoformat(row[4]),
            )
            for row in rows
        ]

    def delete_token_control(self, mint_address: str) -> None:
        with self._connect() as con:
            con.execute("DELETE FROM token_controls WHERE mint_address = ?", (mint_address,))
            con.commit()

    def upsert_token_risk_metrics(self, metrics: TokenRiskMetrics) -> None:
        flags_json = json.dumps(metrics.risk_flags, separators=(",", ":"))
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO token_risk_metrics (
                    mint_address,
                    liquidity_usd,
                    volume_24h_usd,
                    volatility_score,
                    holder_count,
                    top_holder_pct,
                    top10_holder_pct,
                    dev_holding_pct,
                    sniper_holding_pct,
                    insider_holding_pct,
                    bundler_holding_pct,
                    has_oracle_price,
                    price_confidence_bps,
                    last_updated,
                    risk_flags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(mint_address) DO UPDATE SET
                    liquidity_usd = excluded.liquidity_usd,
                    volume_24h_usd = excluded.volume_24h_usd,
                    volatility_score = excluded.volatility_score,
                    holder_count = excluded.holder_count,
                    top_holder_pct = excluded.top_holder_pct,
                    top10_holder_pct = excluded.top10_holder_pct,
                    dev_holding_pct = excluded.dev_holding_pct,
                    sniper_holding_pct = excluded.sniper_holding_pct,
                    insider_holding_pct = excluded.insider_holding_pct,
                    bundler_holding_pct = excluded.bundler_holding_pct,
                    has_oracle_price = excluded.has_oracle_price,
                    price_confidence_bps = excluded.price_confidence_bps,
                    last_updated = excluded.last_updated,
                    risk_flags = excluded.risk_flags
                """,
                (
                    metrics.mint_address,
                    metrics.liquidity_usd,
                    metrics.volume_24h_usd,
                    metrics.volatility_score,
                    metrics.holder_count,
                    metrics.top_holder_pct,
                    metrics.top10_holder_pct,
                    metrics.dev_holding_pct,
                    metrics.sniper_holding_pct,
                    metrics.insider_holding_pct,
                    metrics.bundler_holding_pct,
                    1 if metrics.has_oracle_price else 0,
                    metrics.price_confidence_bps,
                    metrics.last_updated.isoformat(),
                    flags_json,
                ),
            )
            con.commit()

    def get_token_risk_metrics(self, mint_address: str) -> Optional[TokenRiskMetrics]:
        with self._connect() as con:
            cur = con.execute(
                """
                SELECT
                    mint_address,
                    liquidity_usd,
                    volume_24h_usd,
                    volatility_score,
                    holder_count,
                    top_holder_pct,
                    top10_holder_pct,
                    dev_holding_pct,
                    sniper_holding_pct,
                    insider_holding_pct,
                    bundler_holding_pct,
                    has_oracle_price,
                    price_confidence_bps,
                    last_updated,
                    risk_flags
                FROM token_risk_metrics WHERE mint_address = ?
                """,
                (mint_address,),
            )
            row = cur.fetchone()
        if row is None:
            return None
        return TokenRiskMetrics(
            mint_address=row[0],
            liquidity_usd=row[1],
            volume_24h_usd=row[2],
            volatility_score=row[3],
            holder_count=row[4],
            top_holder_pct=row[5],
            top10_holder_pct=row[6],
            has_oracle_price=bool(row[11]),
            price_confidence_bps=row[12],
            last_updated=datetime.fromisoformat(row[13]),
            dev_holding_pct=row[7],
            sniper_holding_pct=row[8],
            insider_holding_pct=row[9],
            bundler_holding_pct=row[10],
            risk_flags=json.loads(row[14]) if row[14] else [],
        )

    def list_token_risk_metrics(self) -> List[TokenRiskMetrics]:
        with self._connect() as con:
            cur = con.execute(
                """
                SELECT
                    mint_address,
                    liquidity_usd,
                    volume_24h_usd,
                    volatility_score,
                    holder_count,
                    top_holder_pct,
                    top10_holder_pct,
                    dev_holding_pct,
                    sniper_holding_pct,
                    insider_holding_pct,
                    bundler_holding_pct,
                    has_oracle_price,
                    price_confidence_bps,
                    last_updated,
                    risk_flags
                FROM token_risk_metrics
                ORDER BY last_updated DESC
                """
            )
            rows = cur.fetchall()
        metrics: List[TokenRiskMetrics] = []
        for row in rows:
            metrics.append(
                TokenRiskMetrics(
                    mint_address=row[0],
                    liquidity_usd=row[1],
                    volume_24h_usd=row[2],
                    volatility_score=row[3],
                    holder_count=row[4],
                    top_holder_pct=row[5],
                    top10_holder_pct=row[6],
                    has_oracle_price=bool(row[11]),
                    price_confidence_bps=row[12],
                    last_updated=datetime.fromisoformat(row[13]),
                    dev_holding_pct=row[7],
                    sniper_holding_pct=row[8],
                    insider_holding_pct=row[9],
                    bundler_holding_pct=row[10],
                    risk_flags=json.loads(row[14]) if row[14] else [],
                )
            )
        return metrics

    def get_token_risk_metrics_bulk(self, mints: Iterable[str]) -> Dict[str, TokenRiskMetrics]:
        tokens = list(dict.fromkeys(mints))
        if not tokens:
            return {}
        placeholders = ",".join("?" for _ in tokens)
        query = (
            "SELECT mint_address, liquidity_usd, volume_24h_usd, volatility_score, holder_count, "
            "top_holder_pct, top10_holder_pct, dev_holding_pct, sniper_holding_pct, insider_holding_pct, "
            "bundler_holding_pct, has_oracle_price, price_confidence_bps, last_updated, risk_flags "
            f"FROM token_risk_metrics WHERE mint_address IN ({placeholders})"
        )
        with self._connect() as con:
            cur = con.execute(query, tokens)
            rows = cur.fetchall()
        results: Dict[str, TokenRiskMetrics] = {}
        for row in rows:
            results[row[0]] = TokenRiskMetrics(
                mint_address=row[0],
                liquidity_usd=row[1],
                volume_24h_usd=row[2],
                volatility_score=row[3],
                holder_count=row[4],
                top_holder_pct=row[5],
                top10_holder_pct=row[6],
                dev_holding_pct=row[7],
                sniper_holding_pct=row[8],
                insider_holding_pct=row[9],
                bundler_holding_pct=row[10],
                has_oracle_price=bool(row[11]),
                price_confidence_bps=row[12],
                last_updated=datetime.fromisoformat(row[13]),
                risk_flags=json.loads(row[14]) if row[14] else [],
            )
        return results

    def upsert_position(self, position: PortfolioPosition) -> None:
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO positions (
                    mint_address,
                    pool_address,
                    venue,
                    allocation,
                    entry_price,
                    created_at,
                    unrealized_pnl_pct,
                    position_address,
                    position_secret,
                    strategy,
                    base_quantity,
                    quote_quantity,
                    realized_pnl_usd,
                    unrealized_pnl_usd,
                    fees_paid_usd,
                    rebates_earned_usd,
                    last_mark_price,
                    last_mark_timestamp,
                    pool_fee_bps,
                    lp_token_amount,
                    lp_token_mint,
                    peak_price,
                    peak_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(mint_address) DO UPDATE SET
                    pool_address = excluded.pool_address,
                    venue = excluded.venue,
                    allocation = excluded.allocation,
                    entry_price = excluded.entry_price,
                    created_at = excluded.created_at,
                    unrealized_pnl_pct = excluded.unrealized_pnl_pct,
                    position_address = excluded.position_address,
                    position_secret = excluded.position_secret,
                    strategy = excluded.strategy,
                    base_quantity = excluded.base_quantity,
                    quote_quantity = excluded.quote_quantity,
                    realized_pnl_usd = excluded.realized_pnl_usd,
                    unrealized_pnl_usd = excluded.unrealized_pnl_usd,
                    fees_paid_usd = excluded.fees_paid_usd,
                    rebates_earned_usd = excluded.rebates_earned_usd,
                    last_mark_price = excluded.last_mark_price,
                    last_mark_timestamp = excluded.last_mark_timestamp,
                    pool_fee_bps = excluded.pool_fee_bps,
                    lp_token_amount = excluded.lp_token_amount,
                    lp_token_mint = excluded.lp_token_mint,
                    peak_price = excluded.peak_price,
                    peak_timestamp = excluded.peak_timestamp
                """,
                (
                    position.token.mint_address,
                    position.pool_address,
                    position.venue,
                    position.allocation,
                    position.entry_price,
                    position.created_at.isoformat(),
                    position.unrealized_pnl_pct,
                    position.position_address,
                    position.position_secret,
                    position.strategy,
                    position.base_quantity,
                    position.quote_quantity,
                    position.realized_pnl_usd,
                    position.unrealized_pnl_usd,
                    position.fees_paid_usd,
                    position.rebates_earned_usd,
                    position.last_mark_price,
                    position.last_mark_timestamp.isoformat()
                    if position.last_mark_timestamp
                    else None,
                    position.pool_fee_bps,
                    position.lp_token_amount,
                    position.lp_token_mint,
                    position.peak_price,
                    position.peak_timestamp.isoformat()
                    if position.peak_timestamp
                    else None,
                ),
            )
            con.commit()

    def delete_position(self, mint_address: str) -> None:
        with self._connect() as con:
            con.execute("DELETE FROM positions WHERE mint_address = ?", (mint_address,))
            con.commit()

    def list_positions(self) -> List[PortfolioPosition]:
        with self._connect() as con:
            cur = con.execute(
                """
                SELECT
                    mint_address,
                    pool_address,
                    venue,
                    allocation,
                    entry_price,
                    created_at,
                    unrealized_pnl_pct,
                    position_address,
                    position_secret,
                    strategy,
                    base_quantity,
                    quote_quantity,
                    realized_pnl_usd,
                    unrealized_pnl_usd,
                    fees_paid_usd,
                    rebates_earned_usd,
                    last_mark_price,
                    last_mark_timestamp,
                    pool_fee_bps,
                    lp_token_amount,
                    lp_token_mint,
                    peak_price,
                    peak_timestamp
                FROM positions
                """
            )
            rows = cur.fetchall()
        tokens = {token.mint_address: token for token in self.list_tokens()}
        positions: List[PortfolioPosition] = []
        for row in rows:
            (
                mint_address,
                pool_address,
                venue,
                allocation,
                entry_price,
                created_at,
                pnl,
                position_address,
                position_secret,
                strategy,
                base_quantity,
                quote_quantity,
                realized_pnl_usd,
                unrealized_pnl_usd,
                fees_paid_usd,
                rebates_earned_usd,
                last_mark_price,
                last_mark_timestamp,
                pool_fee_bps,
                lp_token_amount,
                lp_token_mint,
                peak_price,
                peak_timestamp,
            ) = row
            token = tokens.get(mint_address)
            if token is None:
                # If the token is missing we skip the position; it will be repopulated on next discovery.
                continue
            positions.append(
                PortfolioPosition(
                    token=token,
                    pool_address=pool_address,
                    venue=venue,
                    allocation=allocation,
                    entry_price=entry_price,
                    created_at=datetime.fromisoformat(created_at),
                    unrealized_pnl_pct=pnl,
                    position_address=position_address,
                    position_secret=position_secret,
                    strategy=strategy,
                    base_quantity=base_quantity or 0.0,
                    quote_quantity=quote_quantity or 0.0,
                    realized_pnl_usd=realized_pnl_usd or 0.0,
                    unrealized_pnl_usd=unrealized_pnl_usd or 0.0,
                    fees_paid_usd=fees_paid_usd or 0.0,
                    rebates_earned_usd=rebates_earned_usd or 0.0,
                    last_mark_price=last_mark_price,
                    last_mark_timestamp=datetime.fromisoformat(last_mark_timestamp)
                    if last_mark_timestamp
                    else None,
                    pool_fee_bps=pool_fee_bps,
                    lp_token_amount=lp_token_amount or 0,
                    lp_token_mint=lp_token_mint,
                    peak_price=peak_price,
                    peak_timestamp=datetime.fromisoformat(peak_timestamp)
                    if peak_timestamp
                    else None,
                )
            )
        return positions

    def record_fill(self, fill: FillEvent) -> None:
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO fills (
                    timestamp,
                    mint_address,
                    token_symbol,
                    venue,
                    action,
                    side,
                    base_quantity,
                    quote_quantity,
                    price_usd,
                    fee_usd,
                    rebate_usd,
                    expected_value,
                    slippage_bps,
                    strategy,
                    correlation_id,
                    signature,
                    is_dry_run,
                    pool_address,
                    quote_mint,
                    pool_fee_bps,
                    lp_token_amount,
                    lp_token_mint,
                    position_address,
                    position_secret,
                    expected_slippage_bps,
                    actual_slippage_bps,
                    expected_fee_usd,
                    expected_price_usd,
                    actual_price_usd
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fill.timestamp.isoformat(),
                    fill.mint_address,
                    fill.token_symbol,
                    fill.venue,
                    fill.action,
                    fill.side,
                    fill.base_quantity,
                    fill.quote_quantity,
                    fill.price_usd,
                    fill.fee_usd,
                    fill.rebate_usd,
                    fill.expected_value,
                    fill.slippage_bps,
                    fill.strategy,
                    fill.correlation_id,
                    fill.signature,
                    1 if fill.is_dry_run else 0,
                    fill.pool_address,
                    fill.quote_mint,
                    fill.pool_fee_bps,
                    fill.lp_token_amount,
                    fill.lp_token_mint,
                    fill.position_address,
                    fill.position_secret,
                    fill.expected_slippage_bps,
                    fill.actual_slippage_bps,
                    fill.expected_fee_usd,
                    fill.expected_price_usd,
                    fill.actual_price_usd,
                ),
            )
            con.commit()

    def list_fills(self, limit: int = 500, since: Optional[datetime] = None) -> List[FillEvent]:
        params: List[object] = []
        where_clause = ""
        if since is not None:
            where_clause = "WHERE timestamp >= ?"
            params.append(since.isoformat())
        query = (
            "SELECT timestamp, mint_address, token_symbol, venue, action, side, base_quantity, "
            "quote_quantity, price_usd, fee_usd, rebate_usd, expected_value, slippage_bps, "
            "strategy, correlation_id, signature, is_dry_run, pool_address, quote_mint, pool_fee_bps, "
            "lp_token_amount, lp_token_mint, position_address, position_secret, "
            "expected_slippage_bps, actual_slippage_bps, expected_fee_usd, expected_price_usd, actual_price_usd FROM fills "
            f"{where_clause} ORDER BY timestamp DESC LIMIT ?"
        )
        params.append(limit)
        with self._connect() as con:
            cur = con.execute(query, params)
            rows = cur.fetchall()
        fills: List[FillEvent] = []
        for row in rows:
            (
                timestamp,
                mint_address,
                token_symbol,
                venue,
                action,
                side,
                base_quantity,
                quote_quantity,
                price_usd,
                fee_usd,
                rebate_usd,
                expected_value,
                slippage_bps,
                strategy,
                correlation_id,
                signature,
                is_dry_run,
                pool_address,
                quote_mint,
                pool_fee_bps,
                lp_token_amount,
                lp_token_mint,
                position_address,
                position_secret,
                expected_slippage_bps,
                actual_slippage_bps,
                expected_fee_usd,
                expected_price_usd,
                actual_price_usd,
            ) = row
            fills.append(
                FillEvent(
                    timestamp=datetime.fromisoformat(timestamp),
                    mint_address=mint_address,
                    token_symbol=token_symbol,
                    venue=venue,
                    action=action,
                    side=side,
                    base_quantity=base_quantity,
                    quote_quantity=quote_quantity,
                    price_usd=price_usd,
                    fee_usd=fee_usd,
                    rebate_usd=rebate_usd,
                    expected_value=expected_value,
                    slippage_bps=slippage_bps,
                    strategy=strategy,
                    correlation_id=correlation_id,
                    signature=signature,
                    is_dry_run=bool(is_dry_run),
                    pool_address=pool_address,
                    quote_mint=quote_mint,
                    pool_fee_bps=pool_fee_bps,
                    lp_token_amount=lp_token_amount or 0,
                    lp_token_mint=lp_token_mint,
                    position_address=position_address,
                    position_secret=position_secret,
                    expected_slippage_bps=expected_slippage_bps,
                    actual_slippage_bps=actual_slippage_bps,
                    expected_fee_usd=expected_fee_usd,
                    expected_price_usd=expected_price_usd,
                    actual_price_usd=actual_price_usd,
                )
            )
        return fills

    def record_damm_launches(self, records: Iterable[DammLaunchRecord]) -> None:
        payload = [
            (
                record.mint_address,
                record.pool_address,
                record.fee_bps,
                record.liquidity_usd,
                record.volume_24h_usd,
                record.fee_yield,
                record.age_seconds,
                record.price_usd,
                record.market_cap_usd,
                record.fee_scheduler_mode,
                record.fee_scheduler_current_bps,
                record.fee_scheduler_start_bps,
                record.fee_scheduler_min_bps,
                record.allocation_cap_sol,
                record.recorded_at.isoformat(),
            )
            for record in records
        ]
        if not payload:
            return
        with self._connect() as con:
            con.executemany(
                """
                INSERT INTO damm_launches (
                    mint_address,
                    pool_address,
                    fee_bps,
                    liquidity_usd,
                    volume_24h_usd,
                    fee_yield,
                    age_seconds,
                    price_usd,
                    market_cap_usd,
                    fee_scheduler_mode,
                    fee_scheduler_current_bps,
                    fee_scheduler_start_bps,
                    fee_scheduler_min_bps,
                    allocation_cap_sol,
                    recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(mint_address, pool_address) DO UPDATE SET
                    fee_bps = excluded.fee_bps,
                    liquidity_usd = excluded.liquidity_usd,
                    volume_24h_usd = excluded.volume_24h_usd,
                    fee_yield = excluded.fee_yield,
                    age_seconds = excluded.age_seconds,
                    price_usd = excluded.price_usd,
                    market_cap_usd = excluded.market_cap_usd,
                    fee_scheduler_mode = excluded.fee_scheduler_mode,
                    fee_scheduler_current_bps = excluded.fee_scheduler_current_bps,
                    fee_scheduler_start_bps = excluded.fee_scheduler_start_bps,
                    fee_scheduler_min_bps = excluded.fee_scheduler_min_bps,
                    allocation_cap_sol = excluded.allocation_cap_sol,
                    recorded_at = excluded.recorded_at
                """,
                payload,
            )
            con.commit()

    def list_recent_damm_launches(
        self, limit: int = 50, max_age_seconds: Optional[float] = None
    ) -> List[DammLaunchRecord]:
        query = (
            "SELECT mint_address, pool_address, fee_bps, liquidity_usd, volume_24h_usd, "
            "fee_yield, age_seconds, price_usd, market_cap_usd, fee_scheduler_mode, "
            "fee_scheduler_current_bps, fee_scheduler_start_bps, fee_scheduler_min_bps, allocation_cap_sol, recorded_at FROM damm_launches"
        )
        params: List[object] = []
        if max_age_seconds is not None:
            query += " WHERE age_seconds <= ?"
            params.append(max_age_seconds)
        query += " ORDER BY fee_yield DESC, recorded_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as con:
            cur = con.execute(query, params)
            rows = cur.fetchall()
        return [
            DammLaunchRecord(
                mint_address=row[0],
                pool_address=row[1],
                fee_bps=row[2],
                liquidity_usd=row[3],
                volume_24h_usd=row[4],
                fee_yield=row[5],
                age_seconds=row[6],
                price_usd=row[7],
                market_cap_usd=row[8],
                fee_scheduler_mode=row[9],
                fee_scheduler_current_bps=row[10],
                fee_scheduler_start_bps=row[11],
                fee_scheduler_min_bps=row[12],
                allocation_cap_sol=row[13],
                recorded_at=datetime.fromisoformat(row[14]),
            )
            for row in rows
        ]

    def record_router_decision(self, record: RouterDecisionRecord) -> None:
        extras = json.dumps(record.extras, separators=(",", ":")) if record.extras else None
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO router_decisions (
                    timestamp,
                    mint_address,
                    venue,
                    pool_address,
                    score,
                    allocation_usd,
                    slippage_bps,
                    strategy,
                    correlation_id,
                    quote_mint,
                    extras
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.timestamp.isoformat(),
                    record.mint_address,
                    record.venue,
                    record.pool_address,
                    record.score,
                    record.allocation_usd,
                    record.slippage_bps,
                    record.strategy,
                    record.correlation_id,
                    record.quote_mint,
                    extras,
                ),
            )
            con.commit()

    def list_router_decisions(
        self, limit: int = 200, mint_address: Optional[str] = None
    ) -> List[RouterDecisionRecord]:
        query = (
            "SELECT timestamp, mint_address, venue, pool_address, score, allocation_usd, "
            "slippage_bps, strategy, correlation_id, quote_mint, extras FROM router_decisions"
        )
        params: List[object] = []
        if mint_address:
            query += " WHERE mint_address = ?"
            params.append(mint_address)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        with self._connect() as con:
            cur = con.execute(query, params)
            rows = cur.fetchall()
        decisions: List[RouterDecisionRecord] = []
        for row in rows:
            extras = json.loads(row[10]) if row[10] else {}
            decisions.append(
                RouterDecisionRecord(
                    timestamp=datetime.fromisoformat(row[0]),
                    mint_address=row[1],
                    venue=row[2],
                    pool_address=row[3],
                    score=row[4],
                    allocation_usd=row[5],
                    slippage_bps=row[6],
                    strategy=row[7],
                    correlation_id=row[8],
                    quote_mint=row[9],
                    extras=extras,
                )
            )
        return decisions

    def record_event_log(self, event: EventLogRecord) -> None:
        payload = json.dumps(event.payload, separators=(",", ":"))
        labels = json.dumps(event.labels, separators=(",", ":")) if event.labels else None
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO event_logs (
                    timestamp,
                    event_type,
                    severity,
                    payload,
                    correlation_id,
                    labels
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event.timestamp.isoformat(),
                    event.event_type,
                    event.severity,
                    payload,
                    event.correlation_id,
                    labels,
                ),
            )
            con.commit()

    def list_event_logs(
        self, limit: int = 200, event_type: Optional[str] = None
    ) -> List[EventLogRecord]:
        query = (
            "SELECT timestamp, event_type, severity, payload, correlation_id, labels FROM event_logs"
        )
        params: List[object] = []
        if event_type:
            query += " WHERE event_type = ?"
            params.append(event_type)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        with self._connect() as con:
            cur = con.execute(query, params)
            rows = cur.fetchall()
        events: List[EventLogRecord] = []
        for row in rows:
            events.append(
                EventLogRecord(
                    timestamp=datetime.fromisoformat(row[0]),
                    event_type=row[1],
                    severity=row[2],
                    payload=json.loads(row[3]) if row[3] else {},
                    correlation_id=row[4],
                    labels=json.loads(row[5]) if row[5] else {},
                )
            )
        return events

    def persist_metrics_snapshot(self, snapshot: MetricsSnapshot) -> None:
        payload = json.dumps(snapshot.data, separators=(",", ":"))
        labels = json.dumps(snapshot.labels, separators=(",", ":")) if snapshot.labels else None
        with self._connect() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO metrics_snapshots (timestamp, payload, labels)
                VALUES (?, ?, ?)
                """,
                (snapshot.timestamp.isoformat(), payload, labels),
            )
            con.commit()

    def list_metrics_snapshots(self, limit: int = 200) -> List[MetricsSnapshot]:
        with self._connect() as con:
            cur = con.execute(
                """
                SELECT timestamp, payload, labels
                FROM metrics_snapshots
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cur.fetchall()
        snapshots: List[MetricsSnapshot] = []
        for row in rows:
            snapshots.append(
                MetricsSnapshot(
                    timestamp=datetime.fromisoformat(row[0]),
                    data=json.loads(row[1]) if row[1] else {},
                    labels=json.loads(row[2]) if row[2] else {},
                )
            )
        return snapshots

    def record_pnl_snapshot(self, snapshot: PnLSnapshot) -> None:
        with self._connect() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO pnl_snapshots (
                    timestamp,
                    realized_usd,
                    unrealized_usd,
                    fees_usd,
                    rebates_usd,
                    inventory_value_usd,
                    net_exposure_usd,
                    drawdown_pct,
                    venue_breakdown,
                    pair_breakdown,
                    strategy_breakdown
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.timestamp.isoformat(),
                    snapshot.realized_usd,
                    snapshot.unrealized_usd,
                    snapshot.fees_usd,
                    snapshot.rebates_usd,
                    snapshot.inventory_value_usd,
                    snapshot.net_exposure_usd,
                    snapshot.drawdown_pct,
                    json.dumps(snapshot.venue_breakdown, separators=(",", ":")),
                    json.dumps(snapshot.pair_breakdown, separators=(",", ":")),
                    json.dumps(snapshot.strategy_breakdown, separators=(",", ":")),
                ),
            )
            con.commit()

    def list_pnl_snapshots(self, limit: int = 200) -> List[PnLSnapshot]:
        with self._connect() as con:
            cur = con.execute(
                """
                SELECT
                    timestamp,
                    realized_usd,
                    unrealized_usd,
                    fees_usd,
                    rebates_usd,
                    inventory_value_usd,
                    net_exposure_usd,
                    drawdown_pct,
                    venue_breakdown,
                    pair_breakdown,
                    strategy_breakdown
                FROM pnl_snapshots
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cur.fetchall()
        snapshots: List[PnLSnapshot] = []
        for row in rows:
            (
                timestamp,
                realized_usd,
                unrealized_usd,
                fees_usd,
                rebates_usd,
                inventory_value_usd,
                net_exposure_usd,
                drawdown_pct,
                venue_breakdown,
                pair_breakdown,
                strategy_breakdown,
            ) = row
            snapshots.append(
                PnLSnapshot(
                    timestamp=datetime.fromisoformat(timestamp),
                    realized_usd=realized_usd,
                    unrealized_usd=unrealized_usd,
                    fees_usd=fees_usd,
                    rebates_usd=rebates_usd,
                    inventory_value_usd=inventory_value_usd,
                    net_exposure_usd=net_exposure_usd,
                    drawdown_pct=drawdown_pct,
                    venue_breakdown=json.loads(venue_breakdown) if venue_breakdown else {},
                    pair_breakdown=json.loads(pair_breakdown) if pair_breakdown else {},
                    strategy_breakdown=json.loads(strategy_breakdown)
                    if strategy_breakdown
                    else {},
                )
            )
        return snapshots


__all__ = ["SQLiteStorage", "StorageAdapter", "SCHEMA_VERSION"]
