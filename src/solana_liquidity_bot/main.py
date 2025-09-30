"""Entrypoint for the Solana early-coin liquidity bot."""

from __future__ import annotations

import argparse
import asyncio
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Optional

from solders.pubkey import Pubkey


@contextmanager
def performance_monitor(operation_name: str):
    """Context manager for performance monitoring."""
    start_time = time.time()
    METRICS.observe(f"bot.{operation_name}.duration_seconds", 0.0)  # Initialize
    try:
        yield
    finally:
        duration = time.time() - start_time
        METRICS.observe(f"bot.{operation_name}.duration_seconds", duration)
        METRICS.increment(f"bot.{operation_name}.calls_total", 1.0)

from .analysis.features import build_features
from .analysis.scoring import ScoreEngine
from .analytics.pnl import PnLEngine
from .config.settings import AppMode, get_app_config
from .datalake.storage import SQLiteStorage
from .execution.router import OrderRouter
from .execution.transaction_queue import TransactionQueue
from .execution.venues import DammVenueAdapter, DlmmVenueAdapter
from .execution.wallet import load_wallet
from .ingestion.event_listener import DiscoveryService
from .utils.constants import utc_now
from .ingestion.pricing import PriceOracle
from .monitoring import bootstrap_observability
from .monitoring.logger import get_logger
from .monitoring.metrics import METRICS
from .strategy import (
    AggressiveMakerStrategy,
    Allocator,
    LaunchSniperStrategy,
    LiquidityProvisionStrategy,
    SignalTakerStrategy,
    SpreadMarketMakingStrategy,
    StrategyContext,
    StrategyCoordinator,
)
from .strategy.risk import RiskEngine

logger = get_logger(__name__)


async def run_async(dry_run: bool = True) -> None:
    config = get_app_config()
    storage = SQLiteStorage(config.storage.database_path)
    bootstrap_observability(storage, config=config)
    discovery = DiscoveryService(storage=storage, app_config=config)
    price_oracle = PriceOracle()
    pnl_engine = PnLEngine(storage=storage, price_oracle=price_oracle, config=config.pnl)
    existing_positions = storage.list_positions()
    pnl_engine.prime(existing_positions)
    risk_engine = RiskEngine(pnl_engine=pnl_engine)
    allocator = Allocator(config.strategy)
    strategies = [
        LaunchSniperStrategy(allocator, config.strategy.launch),
        SpreadMarketMakingStrategy(allocator, strategy_config=config.strategy),
        LiquidityProvisionStrategy(allocator),
        SignalTakerStrategy(allocator, strategy_config=config.strategy),
        AggressiveMakerStrategy(allocator, max_entries=config.strategy.max_positions * 2),
    ]
    coordinator = StrategyCoordinator(strategies)
    router = OrderRouter(
        [
            DammVenueAdapter(app_config=config),
            DlmmVenueAdapter(app_config=config),
        ],
        app_config=config,
    )
    queue = TransactionQueue(storage=storage, pnl_engine=pnl_engine, router=router)

    with performance_monitor("universe_discovery"):
        universe = discovery.discover_universe(limit=config.strategy.max_candidates)
    universe_map = {entry.token.mint_address: entry for entry in universe}
    tokens = [
        entry.token
        for entry in universe
        if entry.control is None or entry.control.status == "allow"
    ]
    for token in tokens:
        storage.upsert_token(token)
    with performance_monitor("onchain_stats_fetch"):
        stats = discovery.fetch_onchain_stats(tokens)
    with performance_monitor("liquidity_events_detection"):
        liquidity_events = discovery.detect_liquidity_events(tokens, stats)
    risk_metrics = {
        mint: entry.risk
        for mint, entry in universe_map.items()
        if entry.risk is not None
    }

    features = build_features(tokens, liquidity_events, stats, risk_metrics=risk_metrics)
    score_engine = ScoreEngine()
    scores = score_engine.score(features.values())
    positions = {position.token.mint_address: position for position in storage.list_positions()}

    mode = AppMode.DRY_RUN if dry_run else config.mode.active
    context = StrategyContext(
        timestamp=utc_now(),
        mode=mode,
        scores=scores,
        features=features,
        universe=universe_map,
        positions=positions,
        allocator=allocator,
        risk_engine=risk_engine,
        pnl_engine=pnl_engine,
        router=router,
        price_oracle=price_oracle,
        strategy_config=config.strategy,
        risk_limits=config.risk,
        max_decisions=config.strategy.max_positions,
    )
    with performance_monitor("strategy_planning"):
        decisions = coordinator.plan(context)

    logger.info(
        "Discovered %d tokens, %d decisions ready", len(tokens), len(decisions)
    )
    METRICS.increment("tokens_discovered", len(tokens))
    METRICS.increment("eligible_tokens", len(decisions))

    owner = Pubkey.default()
    wallet = None
    if dry_run:
        await queue.enqueue(decisions, owner, universe_map, mode, dry_run=True)
    else:
        wallet = load_wallet()
        owner = wallet.public_key
        await queue.recover_pending(universe_map, owner, mode)
        await queue.enqueue(decisions, owner, universe_map, mode, dry_run=False)
        await queue.process(wallet)

    snapshot = pnl_engine.snapshot(universe_map, persist=not dry_run)
    logger.info(
        "PnL snapshot: realized=%.2f unrealized=%.2f fees=%.2f inventory=%.2f",
        snapshot.realized_usd,
        snapshot.unrealized_usd,
        snapshot.fees_usd,
        snapshot.inventory_value_usd,
    )


def run(dry_run: bool = True) -> None:
    asyncio.run(run_async(dry_run=dry_run))


async def run_loop(
    dry_run: bool,
    interval_seconds: float,
    max_cycles: Optional[int] = None,
) -> None:
    cycle = 0
    while True:
        cycle += 1
        try:
            await run_async(dry_run=dry_run)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Loop iteration %d failed: %s", cycle, exc, extra={"cycle": cycle})
        if max_cycles is not None and cycle >= max_cycles:
            break
        await asyncio.sleep(max(interval_seconds, 0.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Solana early-coin liquidity bot")
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run the orchestrator continuously with the supplied interval.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=300.0,
        help="Seconds to wait between iterations when --loop is enabled (default: 300)",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Optional limit to the number of loop iterations to execute.",
    )
    args = parser.parse_args()
    if args.loop:
        asyncio.run(run_loop(args.dry_run, args.interval, args.max_cycles))
    else:
        asyncio.run(run_async(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
