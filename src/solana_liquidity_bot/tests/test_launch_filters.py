from datetime import datetime, timezone

from solana_liquidity_bot.config.settings import LaunchSniperConfig
from solana_liquidity_bot.datalake.schemas import (
    LiquidityEvent,
    TokenMetadata,
    TokenRiskMetrics,
    TokenUniverseEntry,
)
from solana_liquidity_bot.ingestion.launch_filters import evaluate_launch_candidate


def _build_entry_with_risk(dev_pct: float, sniper_pct: float, insider_pct: float, bundler_pct: float):
    token = TokenMetadata(mint_address="MintX", symbol="XYZ", name="Token XYZ")
    risk = TokenRiskMetrics(
        mint_address=token.mint_address,
        liquidity_usd=80_000.0,
        volume_24h_usd=120_000.0,
        volatility_score=1.0,
        holder_count=500,
        top_holder_pct=0.15,
        top10_holder_pct=0.25,
        has_oracle_price=True,
        price_confidence_bps=10,
        last_updated=datetime.now(timezone.utc),
        dev_holding_pct=dev_pct,
        sniper_holding_pct=sniper_pct,
        insider_holding_pct=insider_pct,
        bundler_holding_pct=bundler_pct,
    )
    event = LiquidityEvent(
        timestamp=datetime.now(timezone.utc),
        token=token,
        pool_address="pool_xyz",
        base_liquidity=5_000,
        quote_liquidity=5_000,
        pool_fee_bps=3_000,
        price_usd=1.0,
        source="damm",
    )
    return TokenUniverseEntry(
        token=token,
        stats=None,
        damm_pools=[],
        dlmm_pools=[],
        control=None,
        risk=risk,
        liquidity_event=event,
    )


def test_launch_filter_rejects_high_sniper_segment():
    entry = _build_entry_with_risk(dev_pct=0.2, sniper_pct=25.0, insider_pct=5.0, bundler_pct=5.0)
    config = LaunchSniperConfig(allow_missing_holder_segments=False, require_fee_scheduler=False)
    record = evaluate_launch_candidate(entry, config, now=datetime.now(timezone.utc))
    assert record is None


def test_launch_filter_accepts_balanced_segments():
    entry = _build_entry_with_risk(dev_pct=0.1, sniper_pct=10.0, insider_pct=8.0, bundler_pct=12.0)
    config = LaunchSniperConfig(allow_missing_holder_segments=False, require_fee_scheduler=False)
    record = evaluate_launch_candidate(entry, config, now=datetime.now(timezone.utc))
    assert record is not None
    assert record.fee_bps == 3_000
