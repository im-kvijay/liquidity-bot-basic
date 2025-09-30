"""Shared dataclasses and interfaces for venue adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Protocol

from solders.keypair import Keypair
from solders.pubkey import Pubkey
try:  # pragma: no cover - exercised when dependency installed
    from solana.transaction import Transaction
except ModuleNotFoundError:  # pragma: no cover - fallback for test environments
    from ..solana_compat import Transaction  # type: ignore[no-redef]

from ...config.settings import AppMode
from ...datalake.schemas import (
    PortfolioPosition,
    StrategyDecision,
    TokenRiskMetrics,
    TokenUniverseEntry,
)


@dataclass(slots=True)
class PoolContext:
    """Normalized representation of a liquidity pool regardless of venue."""

    venue: str
    address: str
    base_mint: str
    quote_mint: str
    base_liquidity: float
    quote_liquidity: float
    fee_bps: int
    tvl_usd: Optional[float]
    volume_24h_usd: Optional[float]
    price_usd: Optional[float]
    is_active: bool = True
    metadata: Mapping[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class QuoteRequest:
    """Input supplied to venue adapters when requesting a route quote."""

    decision: StrategyDecision
    pool: PoolContext
    risk: Optional[TokenRiskMetrics]
    mode: AppMode
    allocation_usd: float
    base_price_hint: Optional[float]
    quote_price_hint: Optional[float]
    universe_entry: Optional[TokenUniverseEntry] = None
    position: Optional[PortfolioPosition] = None


@dataclass(slots=True)
class VenueQuote:
    """Quote describing the expected execution characteristics for a venue."""

    venue: str
    pool_address: str
    base_mint: str
    quote_mint: str
    allocation_usd: float
    pool_liquidity_usd: float
    expected_price: float
    expected_slippage_bps: float
    liquidity_score: float
    depth_score: float
    volatility_penalty: float
    fee_bps: int
    rebate_bps: float
    expected_fees_usd: float
    base_contribution_lamports: int
    quote_contribution_lamports: int
    notes: List[str] = field(default_factory=list)
    extras: Dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class TransactionPlan:
    """Container for the finalised transaction and metadata."""

    decision: StrategyDecision
    venue: str
    quote: VenueQuote
    transaction: Transaction
    signers: List
    correlation_id: Optional[str] = None
    position_keypair: Optional[Keypair] = None
    position_address: Optional[str] = None
    position_secret: Optional[str] = None
    lp_token_amount: Optional[int] = None
    lp_token_mint: Optional[str] = None


class VenueAdapter(Protocol):
    """Protocol implemented by all execution venue adapters."""

    name: str

    def pools(self, entry: TokenUniverseEntry) -> Iterable[PoolContext]:
        """Return normalized pools for a given universe entry."""

    def quote(self, request: QuoteRequest) -> Optional[VenueQuote]:
        """Produce a quote for the supplied request."""

    def build_plan(self, request: QuoteRequest, quote: VenueQuote, owner: Pubkey) -> TransactionPlan:
        """Convert a quote into a signed transaction plan for execution."""
