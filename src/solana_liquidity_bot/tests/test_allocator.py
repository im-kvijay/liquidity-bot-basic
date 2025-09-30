from datetime import datetime, timezone

import pytest

from solana_liquidity_bot.config.settings import StrategyConfig
from solana_liquidity_bot.datalake.schemas import PortfolioPosition, TokenMetadata, TokenScore
from solana_liquidity_bot.strategy.allocator import Allocator


def _base_metrics(liquidity_component: float) -> dict:
    return {
        "liquidity_component": liquidity_component,
        "fee_apr": 0.0,
        "liquidity_velocity": 0.0,
        "momentum_signal": 0.0,
        "risk_penalty": 0.0,
    }


def test_allocator_respects_inventory_with_missing_price_data() -> None:
    config = StrategyConfig(portfolio_size=1_000.0, allocation_per_position=500.0)
    allocator = Allocator(config)
    token = TokenMetadata(mint_address="MintA", symbol="TKNA", name="Token A", decimals=6)
    score = TokenScore(token=token, score=1.0, reasoning="", metrics=_base_metrics(0.4))
    score.metrics["price_usd"] = None

    baseline_allocation = allocator.determine_allocation(score, strategy="maker")

    position = PortfolioPosition(
        token=token,
        pool_address="poolA",
        allocation=baseline_allocation,
        entry_price=1.0,
        created_at=datetime.now(timezone.utc),
        base_quantity=100.0,
        quote_quantity=0.0,
        last_mark_price=3.0,
    )

    adjusted_allocation = allocator.determine_allocation(
        score, position=position, strategy="maker"
    )

    held_notional = position.base_quantity * position.last_mark_price
    inventory_ratio = min(held_notional / config.portfolio_size, 1.0)
    expected_multiplier = max(0.4, 1.0 - inventory_ratio)

    assert adjusted_allocation == pytest.approx(baseline_allocation * expected_multiplier)
    assert adjusted_allocation < baseline_allocation


def test_plan_entries_continues_after_oversized_candidate() -> None:
    config = StrategyConfig(portfolio_size=500.0, allocation_per_position=500.0)
    allocator = Allocator(config)

    large_token = TokenMetadata(mint_address="MintLarge", symbol="LRG", name="Large Token")
    small_token = TokenMetadata(mint_address="MintSmall", symbol="SML", name="Small Token")

    large_metrics = _base_metrics(0.48)
    large_metrics.update({"price_usd": 1.0, "token_decimals": float(large_token.decimals)})
    small_metrics = _base_metrics(0.2)
    small_metrics.update({"price_usd": 1.0, "token_decimals": float(small_token.decimals)})

    large_score = TokenScore(token=large_token, score=1.0, reasoning="", metrics=large_metrics)
    small_score = TokenScore(token=small_token, score=1.0, reasoning="", metrics=small_metrics)

    decisions = allocator.plan_entries([large_score, small_score])

    assert [decision.token.mint_address for decision in decisions] == [small_token.mint_address]
    assert decisions[0].allocation == pytest.approx(250.0)
