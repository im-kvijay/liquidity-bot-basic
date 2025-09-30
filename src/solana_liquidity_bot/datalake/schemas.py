"""Data models used across ingestion, analysis, and storage layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class TokenMetadata:
    """Describes a newly discovered token on Solana."""

    mint_address: str
    symbol: str
    name: str
    decimals: int = 9
    creator: Optional[str] = None
    project_url: Optional[str] = None
    social_handles: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)


@dataclass(slots=True)
class LiquidityEvent:
    """Metadata for liquidity-related events (e.g., pool creation, LP changes)."""

    timestamp: datetime
    token: TokenMetadata
    pool_address: str
    base_liquidity: float = 0.0
    quote_liquidity: float = 0.0
    pool_fee_bps: int = 30
    tvl_usd: Optional[float] = None
    volume_24h_usd: Optional[float] = None
    price_usd: Optional[float] = None
    quote_token_mint: Optional[str] = None
    source: Optional[str] = None
    launchpad: Optional[str] = None


@dataclass(slots=True)
class TokenOnChainStats:
    """Aggregated on-chain statistics gathered via RPC."""

    token: TokenMetadata
    total_supply: float
    decimals: int
    holder_count: int
    top_holder_pct: float
    top10_holder_pct: float
    liquidity_estimate: float
    minted_at: Optional[datetime]
    last_activity_at: Optional[datetime]
    mint_authority: Optional[str]
    freeze_authority: Optional[str]


@dataclass(slots=True)
class DammPoolSnapshot:
    """Snapshot of a DAMM v2 pool for a particular token."""

    address: str
    base_token_mint: str
    quote_token_mint: str
    base_token_amount: float
    quote_token_amount: float
    fee_bps: int
    base_symbol: Optional[str] = None
    quote_symbol: Optional[str] = None
    tvl_usd: Optional[float] = None
    volume_24h_usd: Optional[float] = None
    price_usd: Optional[float] = None
    is_active: bool = True
    created_at: Optional[datetime] = None
    fee_scheduler_mode: Optional[str] = None
    fee_scheduler_current_bps: Optional[int] = None
    fee_scheduler_min_bps: Optional[int] = None
    fee_scheduler_start_bps: Optional[int] = None
    fee_collection_token: Optional[str] = None
    launchpad: Optional[str] = None


@dataclass(slots=True)
class DlmmPoolSnapshot:
    """Snapshot of a DLMM pool."""

    address: str
    base_token_mint: str
    quote_token_mint: str
    base_token_amount: float
    quote_token_amount: float
    fee_bps: int
    bin_step: Optional[int] = None
    base_virtual_liquidity: Optional[float] = None
    quote_virtual_liquidity: Optional[float] = None
    tvl_usd: Optional[float] = None
    volume_24h_usd: Optional[float] = None
    price_usd: Optional[float] = None
    is_active: bool = True


@dataclass(slots=True)
class TokenScore:
    """Represents a score assigned by the analysis engine."""

    token: TokenMetadata
    score: float
    reasoning: str
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class StrategyDecision:
    """Decision produced by the strategy engine."""

    token: TokenMetadata
    action: str = "hold"
    allocation: float = 0.0
    priority: float = 0.0
    venue: Optional[str] = None
    pool_address: Optional[str] = None
    quote_token_mint: Optional[str] = None
    price_usd: Optional[float] = None
    token_decimals: Optional[int] = None
    quote_token_decimals: Optional[int] = None
    base_liquidity: Optional[float] = None
    quote_liquidity: Optional[float] = None
    pool_fee_bps: Optional[int] = None
    strategy: str = "baseline"
    side: str = "neutral"
    expected_value: float = 0.0
    expected_slippage_bps: float = 0.0
    max_slippage_bps: int = 100
    expected_fill_probability: float = 1.0
    expected_fees_usd: float = 0.0
    expected_rebate_usd: float = 0.0
    correlation_id: Optional[str] = None
    risk_tags: List[str] = field(default_factory=list)
    cooldown_seconds: int = 0
    metadata: Dict[str, float] = field(default_factory=dict)
    diagnostics: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    position_snapshot: Optional["PortfolioPosition"] = None


@dataclass(slots=True)
class ExecutionResult:
    """Summary of a transaction or transaction batch."""

    success: bool
    signature: Optional[str] = None
    error: Optional[str] = None
    slot: Optional[int] = None


@dataclass(slots=True)
class PortfolioPosition:
    """Represents an active liquidity position."""

    token: TokenMetadata
    pool_address: str
    allocation: float
    entry_price: float
    created_at: datetime
    unrealized_pnl_pct: float = 0.0
    position_address: Optional[str] = None
    position_secret: Optional[str] = None
    venue: Optional[str] = None
    strategy: Optional[str] = None
    base_quantity: float = 0.0
    quote_quantity: float = 0.0
    realized_pnl_usd: float = 0.0
    unrealized_pnl_usd: float = 0.0
    fees_paid_usd: float = 0.0
    rebates_earned_usd: float = 0.0
    last_mark_price: Optional[float] = None
    last_mark_timestamp: Optional[datetime] = None
    pool_fee_bps: Optional[int] = None
    lp_token_amount: int = 0
    lp_token_mint: Optional[str] = None
    # Peak price tracking for trailing stops
    peak_price: Optional[float] = None
    peak_timestamp: Optional[datetime] = None


@dataclass(slots=True)
class TokenControlDecision:
    """Manual or automated allow/deny/paused state for a token."""

    mint_address: str
    status: str
    reason: str
    source: str
    updated_at: datetime


@dataclass(slots=True)
class TokenRiskMetrics:
    """Aggregated risk metrics captured during discovery."""

    mint_address: str
    liquidity_usd: float
    volume_24h_usd: float
    volatility_score: float
    holder_count: int
    top_holder_pct: float
    top10_holder_pct: float
    has_oracle_price: bool
    price_confidence_bps: int
    last_updated: datetime
    dev_holding_pct: Optional[float] = None
    sniper_holding_pct: Optional[float] = None
    insider_holding_pct: Optional[float] = None
    bundler_holding_pct: Optional[float] = None
    risk_flags: List[str] = field(default_factory=list)


@dataclass(slots=True)
class TokenUniverseEntry:
    """Comprehensive view of a token candidate with decisions and risk."""

    token: TokenMetadata
    stats: Optional[TokenOnChainStats]
    damm_pools: List[DammPoolSnapshot] = field(default_factory=list)
    dlmm_pools: List[DlmmPoolSnapshot] = field(default_factory=list)
    control: Optional[TokenControlDecision] = None
    risk: Optional[TokenRiskMetrics] = None
    liquidity_event: Optional[LiquidityEvent] = None


@dataclass(slots=True)
class RocketscanMetrics:
    """Holder cohort distribution as reported by RocketScan."""

    dev_balance_pct: Optional[float] = None
    snipers_pct: Optional[float] = None
    insiders_pct: Optional[float] = None
    bundlers_pct: Optional[float] = None
    top_holders_pct: Optional[float] = None


@dataclass(slots=True)
class DammLaunchRecord:
    """Captures a high-fee DAMM v2 opportunity candidate."""

    mint_address: str
    pool_address: str
    fee_bps: int
    liquidity_usd: float
    volume_24h_usd: float
    fee_yield: float
    age_seconds: float
    price_usd: Optional[float]
    market_cap_usd: Optional[float]
    fee_scheduler_mode: Optional[str]
    fee_scheduler_current_bps: Optional[int]
    fee_scheduler_start_bps: Optional[int]
    fee_scheduler_min_bps: Optional[int]
    allocation_cap_sol: Optional[float]
    recorded_at: datetime


@dataclass(slots=True)
class FillEvent:
    """Auditable representation of a fill or simulated fill."""

    timestamp: datetime
    mint_address: str
    token_symbol: str
    venue: str
    action: str
    side: str
    base_quantity: float
    quote_quantity: float
    price_usd: float
    fee_usd: float
    rebate_usd: float
    expected_value: float
    slippage_bps: float
    strategy: str
    correlation_id: str
    signature: Optional[str] = None
    is_dry_run: bool = False
    pool_address: Optional[str] = None
    quote_mint: Optional[str] = None
    pool_fee_bps: Optional[int] = None
    lp_token_amount: int = 0
    lp_token_mint: Optional[str] = None
    position_address: Optional[str] = None
    position_secret: Optional[str] = None
    # Drift monitoring fields
    expected_slippage_bps: Optional[float] = None
    actual_slippage_bps: Optional[float] = None
    expected_fee_usd: Optional[float] = None
    expected_price_usd: Optional[float] = None
    actual_price_usd: Optional[float] = None


@dataclass(slots=True)
class PnLSnapshot:
    """Aggregated PnL view captured at a point in time."""

    timestamp: datetime
    realized_usd: float
    unrealized_usd: float
    fees_usd: float
    rebates_usd: float
    inventory_value_usd: float
    net_exposure_usd: float
    drawdown_pct: float
    venue_breakdown: Dict[str, float] = field(default_factory=dict)
    pair_breakdown: Dict[str, float] = field(default_factory=dict)
    strategy_breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class RouterDecisionRecord:
    """Persisted representation of a router choice for audit and analytics."""

    timestamp: datetime
    mint_address: str
    venue: str
    pool_address: str
    score: float
    allocation_usd: float
    slippage_bps: float
    strategy: str
    correlation_id: Optional[str] = None
    quote_mint: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EventLogRecord:
    """Structured event emitted by the internal observability bus."""

    timestamp: datetime
    event_type: str
    severity: str
    payload: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class MetricsSnapshot:
    """Compressed view of metrics suitable for persistence."""

    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)


__all__ = [
    "TokenMetadata",
    "LiquidityEvent",
    "TokenOnChainStats",
    "DammPoolSnapshot",
    "DlmmPoolSnapshot",
    "TokenScore",
    "StrategyDecision",
    "ExecutionResult",
    "PortfolioPosition",
    "TokenControlDecision",
    "TokenRiskMetrics",
    "TokenUniverseEntry",
    "DammLaunchRecord",
    "FillEvent",
    "PnLSnapshot",
    "RouterDecisionRecord",
    "EventLogRecord",
    "MetricsSnapshot",
    "RocketscanMetrics",
]
