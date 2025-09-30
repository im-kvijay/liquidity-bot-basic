"""Tests for exit generation in the spread market making strategy."""

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

from solana_liquidity_bot.config.settings import AppMode, StrategyConfig
from solana_liquidity_bot.datalake.schemas import (
    LiquidityEvent,
    PortfolioPosition,
    TokenMetadata,
    TokenScore,
    TokenUniverseEntry,
)
from solana_liquidity_bot.strategy.allocator import Allocator
from solana_liquidity_bot.strategy.base import StrategyContext
from solana_liquidity_bot.strategy.market_making import SpreadMarketMakingStrategy


class _StubRouter:
    def evaluate_routes(self, decision, entry, mode, limit=1):  # noqa: D401
        return []

    def route(self, decision, entry, mode):
        return None


def _make_context(position: PortfolioPosition, score: TokenScore, entry: TokenUniverseEntry) -> StrategyContext:
    config = StrategyConfig()
    allocator = Allocator(config)
    router = _StubRouter()
    price_oracle = MagicMock()
    price_oracle.get_price.return_value = None
    risk_engine = MagicMock()
    risk_engine.update_exposures.side_effect = lambda exposures: None
    pnl_engine = MagicMock()
    pnl_engine.exposures.return_value = {}
    risk_limits = MagicMock()
    risk_limits.max_slippage_bps = 90
    return StrategyContext(
        timestamp=datetime.now(timezone.utc),
        mode=AppMode.DRY_RUN,
        scores={score.token.mint_address: score},
        features={},
        universe={entry.token.mint_address: entry},
        positions={position.token.mint_address: position},
        allocator=allocator,
        risk_engine=risk_engine,
        pnl_engine=pnl_engine,
        router=router,
        price_oracle=price_oracle,
        strategy_config=config,
        risk_limits=risk_limits,
        max_decisions=5,
    )


def test_spread_strategy_emits_exit_on_take_profit():
    token = TokenMetadata(mint_address="mint", symbol="TOK", name="Token", decimals=6)
    entry = TokenUniverseEntry(
        token=token,
        stats=None,
        damm_pools=[],
        dlmm_pools=[],
        control=None,
        risk=None,
        liquidity_event=LiquidityEvent(
            timestamp=datetime.now(timezone.utc) - timedelta(hours=2),
            token=token,
            pool_address="pool",
            base_liquidity=1000.0,
            quote_liquidity=2000.0,
            pool_fee_bps=30,
            price_usd=1.0,
            quote_token_mint="quote",
            source="damm",
        ),
    )
    position = PortfolioPosition(
        token=token,
        pool_address="pool",
        allocation=100.0,
        entry_price=1.0,
        created_at=datetime.now(timezone.utc) - timedelta(hours=3),
        venue="damm",
        strategy="spread_mm",
        base_quantity=10.0,
        quote_quantity=100.0,
        lp_token_amount=1_000,
    )
    score = TokenScore(
        token=token,
        score=1.0,
        reasoning="",
        metrics={"price_usd": 1.5, "quote_token_decimals": 6},
    )
    context = _make_context(position, score, entry)
    strategy = SpreadMarketMakingStrategy(Allocator(context.strategy_config), context.strategy_config)

    decisions = strategy.generate(context)

    assert decisions, "expected at least one decision"
    exit_decisions = [decision for decision in decisions if decision.action == "exit"]
    assert exit_decisions, "expected an exit decision"
    decision = exit_decisions[0]
    assert decision.metadata.get("exit_reason") == "take_profit"
    assert decision.allocation > 0
    assert decision.position_snapshot is position
