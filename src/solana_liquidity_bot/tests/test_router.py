from dataclasses import dataclass
from datetime import datetime, timezone

from solana_liquidity_bot.config.settings import AppMode
from solana_liquidity_bot.datalake.schemas import (
    StrategyDecision,
    TokenMetadata,
    TokenRiskMetrics,
    TokenUniverseEntry,
)
from solana_liquidity_bot.execution.router import OrderRouter
from solana_liquidity_bot.execution.venues.base import PoolContext, QuoteRequest, VenueAdapter, VenueQuote


@dataclass
class _StaticAdapter(VenueAdapter):
    name: str
    pool: PoolContext
    quote_value: VenueQuote

    def pools(self, entry):
        return [self.pool]

    def quote(self, request: QuoteRequest):
        return self.quote_value

    def build_plan(self, request: QuoteRequest, quote: VenueQuote, owner):
        raise NotImplementedError


def test_router_prefers_higher_score():
    decision = StrategyDecision(
        token=TokenMetadata(mint_address="mint", symbol="TOK", name="Token"),
        action="enter",
        allocation=1000.0,
        priority=1.0,
        price_usd=1.0,
        quote_token_mint="Quote",
    )
    risk = TokenRiskMetrics(
        mint_address="mint",
        liquidity_usd=50_000.0,
        volume_24h_usd=120_000.0,
        volatility_score=0.2,
        holder_count=500,
        top_holder_pct=0.1,
        top10_holder_pct=0.3,
        has_oracle_price=True,
        price_confidence_bps=20,
        last_updated=datetime.now(timezone.utc),
        risk_flags=[],
    )
    entry = TokenUniverseEntry(
        token=decision.token,
        stats=None,
        damm_pools=[],
        dlmm_pools=[],
        control=None,
        risk=risk,
    )

    pool_a = PoolContext(
        venue="alpha",
        address="PoolA",
        base_mint=decision.token.mint_address,
        quote_mint=decision.quote_token_mint,
        base_liquidity=5_000.0,
        quote_liquidity=5_000.0,
        fee_bps=30,
        tvl_usd=20_000.0,
        volume_24h_usd=50_000.0,
        price_usd=1.0,
        is_active=True,
        metadata={},
    )
    quote_a = VenueQuote(
        venue="alpha",
        pool_address=pool_a.address,
        base_mint=pool_a.base_mint,
        quote_mint=pool_a.quote_mint,
        allocation_usd=decision.allocation,
        pool_liquidity_usd=20_000.0,
        expected_price=1.0,
        expected_slippage_bps=40.0,
        liquidity_score=0.5,
        depth_score=0.5,
        volatility_penalty=0.1,
        fee_bps=30,
        rebate_bps=0.0,
        expected_fees_usd=3.0,
        base_contribution_lamports=0,
        quote_contribution_lamports=0,
        extras={"pool_base_value_usd": 10_000.0, "pool_quote_value_usd": 10_000.0},
    )

    pool_b = PoolContext(
        venue="bravo",
        address="PoolB",
        base_mint=decision.token.mint_address,
        quote_mint=decision.quote_token_mint,
        base_liquidity=12_000.0,
        quote_liquidity=12_000.0,
        fee_bps=20,
        tvl_usd=55_000.0,
        volume_24h_usd=95_000.0,
        price_usd=1.0,
        is_active=True,
        metadata={},
    )
    quote_b = VenueQuote(
        venue="bravo",
        pool_address=pool_b.address,
        base_mint=pool_b.base_mint,
        quote_mint=pool_b.quote_mint,
        allocation_usd=decision.allocation,
        pool_liquidity_usd=55_000.0,
        expected_price=1.0,
        expected_slippage_bps=12.0,
        liquidity_score=1.0,
        depth_score=1.0,
        volatility_penalty=0.05,
        fee_bps=20,
        rebate_bps=0.0,
        expected_fees_usd=2.0,
        base_contribution_lamports=0,
        quote_contribution_lamports=0,
        extras={"pool_base_value_usd": 27_500.0, "pool_quote_value_usd": 27_500.0},
    )

    adapter_a = _StaticAdapter(name="alpha", pool=pool_a, quote_value=quote_a)
    adapter_b = _StaticAdapter(name="bravo", pool=pool_b, quote_value=quote_b)

    router = OrderRouter([adapter_a, adapter_b])
    decision_result = router.route(decision, entry, AppMode.DRY_RUN)

    assert decision_result is not None
    assert decision_result.quote.venue == "bravo"
