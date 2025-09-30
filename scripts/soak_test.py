"""Dry-run soak test harness for the Solana liquidity bot."""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from datetime import datetime, timedelta
from typing import List

from solana_liquidity_bot.main import run_async as run_bot_async
from solana_liquidity_bot.monitoring.logger import get_logger
from solana_liquidity_bot.monitoring.metrics import METRICS

logger = get_logger(__name__)


async def _cycle_once(dry_run: bool) -> float:
    start = time.perf_counter()
    await run_bot_async(dry_run=dry_run)
    duration = time.perf_counter() - start
    logger.info("Soak iteration completed in %.2fs", duration)
    return duration


async def soak_test(minutes: int, *, dry_run: bool = True, pause_seconds: float = 2.0) -> None:
    """Run the trading loop repeatedly for the requested duration."""

    deadline = datetime.utcnow() + timedelta(minutes=minutes)
    durations: List[float] = []
    iterations = 0
    while datetime.utcnow() < deadline:
        iterations += 1
        try:
            duration = await _cycle_once(dry_run)
            durations.append(duration)
        except Exception:  # noqa: BLE001 - soak tests should surface errors but continue
            logger.exception("Soak iteration %d failed", iterations)
        if datetime.utcnow() >= deadline:
            break
        await asyncio.sleep(pause_seconds)

    if durations:
        p95 = durations[0] if len(durations) == 1 else statistics.quantiles(durations, n=20)[18]
        logger.info(
            "Soak summary: %d runs, mean duration %.2fs, p95 %.2fs",
            len(durations),
            statistics.mean(durations),
            p95,
        )
    else:
        logger.warning("Soak test finished without successful iterations")

    snapshot = METRICS.snapshot()
    logger.info("Final metrics snapshot: %s", snapshot)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a dry-run soak test for the liquidity bot")
    parser.add_argument("--minutes", type=int, default=30, help="Duration of the soak test")
    parser.add_argument("--live", action="store_true", help="Run in live mode (default: dry run)")
    parser.add_argument("--pause", type=float, default=2.0, help="Delay between iterations")
    args = parser.parse_args()

    asyncio.run(soak_test(args.minutes, dry_run=not args.live, pause_seconds=args.pause))


if __name__ == "__main__":
    main()
