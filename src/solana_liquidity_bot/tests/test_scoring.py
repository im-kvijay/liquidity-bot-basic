"""Unit tests for the scoring engine."""

from datetime import datetime, timedelta, timezone

from solana_liquidity_bot.analysis.features import build_features
from solana_liquidity_bot.analysis.scoring import ScoreEngine
from solana_liquidity_bot.datalake.schemas import (
    LiquidityEvent,
    TokenMetadata,
    TokenOnChainStats,
)


def _make_token(index: int) -> TokenMetadata:
    return TokenMetadata(
        mint_address=f"mint{index}",
        symbol=f"TOK{index}",
        name=f"Token {index}",
        creator="creator" if index % 2 == 0 else None,
        project_url="https://example.com" if index % 3 == 0 else None,
    )


def _make_event(token: TokenMetadata) -> LiquidityEvent:
    return LiquidityEvent(
        timestamp=datetime.now(timezone.utc),
        token=token,
        pool_address=f"pool_{token.mint_address}",
        base_liquidity=1000.0,
        quote_liquidity=2000.0,
        pool_fee_bps=30,
        tvl_usd=6000.0,
        volume_24h_usd=12000.0,
        price_usd=0.5,
        quote_token_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8WdGCr3vZ9V4Wrh",
    )


def _make_stats(token: TokenMetadata) -> TokenOnChainStats:
    return TokenOnChainStats(
        token=token,
        total_supply=1_000_000,
        decimals=9,
        holder_count=150,
        top_holder_pct=0.2,
        top10_holder_pct=0.5,
        liquidity_estimate=5000.0,
        minted_at=datetime.now(timezone.utc) - timedelta(minutes=45),
        last_activity_at=datetime.now(timezone.utc),
        mint_authority=None,
        freeze_authority=None,
    )


def test_scoring_rewards_liquidity_and_metadata():
    token = _make_token(1)
    event = _make_event(token)
    stats = _make_stats(token)

    features = build_features([token], [event], [stats])
    engine = ScoreEngine()
    scores = engine.score(features.values())

    score = scores[token.mint_address]
    assert score.score > 0
    assert score.metrics["fee_apr"] > 0


def test_social_score_influences_total_score():
    token_a = _make_token(1)
    token_b = _make_token(2)
    event_a = _make_event(token_a)
    event_b = _make_event(token_b)
    stats_a = _make_stats(token_a)
    stats_b = _make_stats(token_b)

    features = build_features([token_a, token_b], [event_a, event_b], [stats_a, stats_b])
    engine = ScoreEngine()
    scores = engine.score(features.values())

    assert scores[token_b.mint_address].score >= scores[token_a.mint_address].score
