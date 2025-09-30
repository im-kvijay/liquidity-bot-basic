from datetime import datetime, timezone

from solana_liquidity_bot.analysis.features import TokenFeatures
from solana_liquidity_bot.analytics.pnl import PnLEngine
from solana_liquidity_bot.config.settings import AppMode, StrategyConfig, get_app_config
from solana_liquidity_bot.datalake.schemas import (
    DammPoolSnapshot,
    TokenMetadata,
    TokenRiskMetrics,
    TokenScore,
    TokenUniverseEntry,
)
from solana_liquidity_bot.datalake.storage import SQLiteStorage
from solana_liquidity_bot.execution.router import RoutingDecision
from solana_liquidity_bot.execution.venues.base import PoolContext, QuoteRequest, VenueQuote
from solana_liquidity_bot.strategy import (
    Allocator,
    StrategyContext,
    StrategyCoordinator,
)
from solana_liquidity_bot.strategy.market_making import SpreadMarketMakingStrategy
from solana_liquidity_bot.strategy.risk import RiskEngine


class DummyOracle:
    def get_mark_price(self, mint: str, *, source: str = "oracle", window_seconds: int | None = None, fallback=None):
        return 1.0

    def get_price(self, mint: str) -> float:
        return 1.0


class DummyRouter:
    def evaluate_routes(self, decision, entry, mode, limit=None):
        quote = VenueQuote(
            venue="test",
            pool_address="pool",
            base_mint=decision.token.mint_address,
            quote_mint="Quote",
            allocation_usd=decision.allocation,
            pool_liquidity_usd=10_000.0,
            expected_price=1.0,
            expected_slippage_bps=15.0,
            liquidity_score=1.0,
            depth_score=1.0,
            volatility_penalty=0.0,
            fee_bps=30,
            rebate_bps=5.0,
            expected_fees_usd=decision.allocation * 0.003,
            base_contribution_lamports=0,
            quote_contribution_lamports=0,
            extras={},
        )
        request = QuoteRequest(
            decision=decision,
            pool=PoolContext(
                venue="test",
                address="pool",
                base_mint=decision.token.mint_address,
                quote_mint="Quote",
                base_liquidity=5_000.0,
                quote_liquidity=5_000.0,
                fee_bps=30,
                tvl_usd=10_000.0,
                volume_24h_usd=25_000.0,
                price_usd=1.0,
                is_active=True,
                metadata={},
            ),
            risk=None,
            mode=mode,
            allocation_usd=decision.allocation,
            base_price_hint=1.0,
            quote_price_hint=1.0,
            universe_entry=entry,
        )
        return [RoutingDecision(adapter=None, request=request, quote=quote, score=1.0)]

    def route(self, decision, entry, mode):
        routes = self.evaluate_routes(decision, entry, mode, limit=1)
        return routes[0] if routes else None


def test_strategy_coordinator_produces_decisions(tmp_path):
    storage = SQLiteStorage(tmp_path / "state.sqlite3")
    token = TokenMetadata(mint_address="Mint11111111111111111111111111111111", symbol="TKN", name="Token")
    storage.upsert_token(token)
    pnl_engine = PnLEngine(storage=storage, price_oracle=DummyOracle())
    risk_engine = RiskEngine(pnl_engine=pnl_engine)
    strategy_config = StrategyConfig()
    allocator = Allocator(strategy_config)
    strategy = SpreadMarketMakingStrategy(allocator, strategy_config=strategy_config)
    coordinator = StrategyCoordinator([strategy])

    features = {
        token.mint_address: TokenFeatures(
            token=token,
            liquidity_depth_usd=20_000.0,
            tvl_usd=20_000.0,
            volume_24h_usd=40_000.0,
            liquidity_velocity=2.0,
            fee_apr=0.2,
            holder_concentration=0.1,
            top10_holder_concentration=0.2,
            holder_count=500,
            minted_minutes_ago=10.0,
            dev_trust_score=0.8,
            social_score=0.6,
            price_usd=1.0,
            pool_address="pool",
            quote_token_mint="Quote",
            base_liquidity=10_000.0,
            quote_liquidity=10_000.0,
            pool_fee_bps=30,
            token_decimals=6,
            quote_token_decimals=6,
        )
    }
    scores = {token.mint_address: TokenScore(token=token, score=0.9, reasoning="", metrics={})}
    entry = TokenUniverseEntry(
        token=token,
        stats=None,
        damm_pools=[
            DammPoolSnapshot(
                address="pool",
                base_token_mint=token.mint_address,
                quote_token_mint="Quote",
                base_token_amount=10_000.0,
                quote_token_amount=10_000.0,
                fee_bps=30,
                tvl_usd=20_000.0,
                volume_24h_usd=40_000.0,
                price_usd=1.0,
                is_active=True,
            )
        ],
        dlmm_pools=[],
        control=None,
        risk=TokenRiskMetrics(
            mint_address=token.mint_address,
            liquidity_usd=20_000.0,
            volume_24h_usd=40_000.0,
            volatility_score=0.4,
            holder_count=500,
            top_holder_pct=0.1,
            top10_holder_pct=0.2,
            has_oracle_price=True,
            price_confidence_bps=10,
            last_updated=datetime.now(timezone.utc),
            risk_flags=[],
        ),
        liquidity_event=None,
    )

    context = StrategyContext(
        timestamp=datetime.now(timezone.utc),
        mode=AppMode.DRY_RUN,
        scores=scores,
        features=features,
        universe={token.mint_address: entry},
        positions={},
        allocator=allocator,
        risk_engine=risk_engine,
        pnl_engine=pnl_engine,
        router=DummyRouter(),
        price_oracle=DummyOracle(),
        strategy_config=strategy_config,
        risk_limits=get_app_config().risk,
        max_decisions=3,
    )

    decisions = coordinator.plan(context)
    assert decisions, "strategy should emit at least one decision"
    assert any(decision.strategy == "spread_mm" for decision in decisions)
