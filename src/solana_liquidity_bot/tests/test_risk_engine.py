from datetime import datetime, timezone

from solana_liquidity_bot.analytics.pnl import ExposureSummary
from solana_liquidity_bot.config.settings import CircuitBreakerConfig, RiskLimitsConfig
from solana_liquidity_bot.datalake.schemas import (
    StrategyDecision,
    TokenMetadata,
    TokenRiskMetrics,
    TokenUniverseEntry,
)
from solana_liquidity_bot.strategy.risk import RiskEngine


def test_risk_engine_blocks_notional_and_slippage():
    risk_engine = RiskEngine(
        config=RiskLimitsConfig(
            max_global_notional_usd=100.0,
            max_market_notional_usd=50.0,
            max_position_notional_usd=25.0,
            max_inventory_pct=0.5,
            max_open_orders=5,
            daily_loss_limit_usd=1_000.0,
            max_slippage_bps=40,
        ),
        breaker=CircuitBreakerConfig(
            max_drawdown_pct=1.0,
            max_reject_rate=1.0,
            max_volatility_pct=1.0,
            health_check_backoff_seconds=60,
            cooldown_seconds=60,
        ),
        pnl_engine=None,
    )
    risk_engine.update_exposures(
        {"Mint": ExposureSummary(mint_address="Mint", notional_usd=60.0, base_quantity=0.0, last_price=1.0)}
    )

    token = TokenMetadata(mint_address="Mint", symbol="TKN", name="Token")
    decision = StrategyDecision(
        token=token,
        allocation=60.0,
        strategy="spread_mm",
        side="maker",
        max_slippage_bps=60,
    )
    entry = TokenUniverseEntry(
        token=token,
        stats=None,
        damm_pools=[],
        dlmm_pools=[],
        control=None,
        risk=TokenRiskMetrics(
            mint_address="Mint",
            liquidity_usd=10.0,
            volume_24h_usd=100.0,
            volatility_score=0.5,
            holder_count=100,
            top_holder_pct=0.1,
            top10_holder_pct=0.2,
            has_oracle_price=True,
            price_confidence_bps=50,
            last_updated=datetime.now(timezone.utc),
            risk_flags=[],
        ),
        liquidity_event=None,
    )

    result = risk_engine.evaluate(decision, entry)
    assert not result.approved
    assert "global_notional_limit" in result.reasons
    assert "slippage_limit" in result.reasons
