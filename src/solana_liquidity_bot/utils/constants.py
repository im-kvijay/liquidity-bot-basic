"""Shared constants for Solana token analysis."""

from datetime import datetime, timezone

# Utility function to get timezone-aware UTC datetime
def utc_now() -> datetime:
    """Get current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)

LAMPORTS_PER_SOL = 1_000_000_000

# Common stablecoin mints used when inferring USD liquidity.
STABLECOIN_MINTS: dict[str, str] = {
    "EPjFWdd5AufqSSqeM2qN1xzybapC8WdGCr3vZ9V4Wrh": "USDC",
    "Es9vMFrzaCER1ZXS9dZxXn6vufGAaHo9dZYkTf5ZNVY": "USDT",
    "USDVQa5H421y1AGG1iL93AdtSZdPdvHb2nTeS9n8AV": "USDV",
    "BXXkv6z5CAV5gZSkhUbsxLZkj3ZJ8nFa43uRq1CXY5a": "DAI",
}

SOL_MINT = "So11111111111111111111111111111111111111112"

__all__ = ["utc_now", "LAMPORTS_PER_SOL", "STABLECOIN_MINTS", "SOL_MINT"]
