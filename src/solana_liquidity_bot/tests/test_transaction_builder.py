from solders.pubkey import Pubkey

from solders.pubkey import Pubkey

from solana_liquidity_bot.config.settings import AppMode
from solana_liquidity_bot.datalake.schemas import StrategyDecision, TokenMetadata
from solana_liquidity_bot.execution.transaction_builder import DlmmTransactionBuilder
from solana_liquidity_bot.execution.venues.base import PoolContext, QuoteRequest, VenueQuote


def test_dlmm_builder_invokes_node_bridge(monkeypatch):
    builder = DlmmTransactionBuilder()

    monkeypatch.setattr(builder, "_resolve_decimals", lambda *args, **kwargs: 6)
    monkeypatch.setattr(builder, "_resolve_price", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr(builder, "_resolve_quote_price", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(
        builder,
        "_node_bridge",
        type(
            "MockBridge",
            (),
            {
                "run": lambda self, script, payload: {
                    "instructions": [
                        {
                            "programId": "11111111111111111111111111111111",
                            "data": "",
                            "accounts": [],
                        }
                    ]
                }
            },
        )(),
    )

    decision = StrategyDecision(
        token=TokenMetadata(mint_address="mint", symbol="TOK", name="Token"),
        action="enter",
        allocation=100.0,
        priority=1.0,
        quote_token_mint="QuoteMint111111111111111111111111111111111",
        price_usd=0.5,
    )

    pool = PoolContext(
        venue="dlmm",
        address="Pool1111111111111111111111111111111111111",
        base_mint=decision.token.mint_address,
        quote_mint=decision.quote_token_mint,
        base_liquidity=1_000.0,
        quote_liquidity=1_000.0,
        fee_bps=30,
        tvl_usd=5_000.0,
        volume_24h_usd=25_000.0,
        price_usd=0.5,
        is_active=True,
        metadata={},
    )
    quote = VenueQuote(
        venue="dlmm",
        pool_address=pool.address,
        base_mint=pool.base_mint,
        quote_mint=pool.quote_mint,
        allocation_usd=decision.allocation,
        pool_liquidity_usd=5_000.0,
        expected_price=0.5,
        expected_slippage_bps=25.0,
        liquidity_score=1.0,
        depth_score=1.0,
        volatility_penalty=0.0,
        fee_bps=30,
        rebate_bps=0.0,
        expected_fees_usd=0.3,
        base_contribution_lamports=0,
        quote_contribution_lamports=0,
        extras={"pool_base_value_usd": 2_500.0, "pool_quote_value_usd": 2_500.0},
    )
    request = QuoteRequest(
        decision=decision,
        pool=pool,
        risk=None,
        mode=AppMode.DRY_RUN,
        allocation_usd=decision.allocation,
        base_price_hint=0.5,
        quote_price_hint=1.0,
        universe_entry=None,
    )

    plan = builder.build_plan(request, quote, Pubkey.from_string("11111111111111111111111111111111"))

    assert plan.transaction.instructions, "builder should produce at least one instruction"
    assert plan.signers, "position keypair should be included as a signer"
