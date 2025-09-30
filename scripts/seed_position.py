"""Seed a synthetic portfolio position in the local SQLite state."""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Optional

from solana_liquidity_bot.config.settings import get_app_config
from solana_liquidity_bot.datalake.schemas import PortfolioPosition, TokenMetadata
from solana_liquidity_bot.datalake.storage import SQLiteStorage
from solana_liquidity_bot.ingestion.event_listener import DiscoveryService


def _resolve_token(
    storage: SQLiteStorage,
    universe,
    discovery: DiscoveryService,
    mint: str,
    candidate_limit: int,
) -> TokenMetadata:
    token = storage.get_token(mint)
    if token:
        return token
    for entry in universe:
        if entry.token.mint_address == mint:
            storage.upsert_token(entry.token)
            return entry.token
    raise SystemExit(f"Token {mint} not found in registry. Run discovery first.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed a synthetic position for dry-run testing.")
    parser.add_argument("mint", help="Token mint address")
    parser.add_argument("base_quantity", type=float, help="Base quantity to seed")
    parser.add_argument("price_usd", type=float, help="Entry price in USD")
    parser.add_argument("venue", choices=["damm", "dlmm"], help="Execution venue")
    parser.add_argument("--pool-address", default=None, help="Pool address for the position")
    parser.add_argument("--fee-bps", type=int, default=30, help="Pool fee in basis points")
    parser.add_argument("--lp-mint", default=None, help="LP token mint address")
    parser.add_argument(
        "--strategy",
        default="launch_sniper",
        help="Strategy label to tag the position (defaults to launch_sniper)",
    )
    parser.add_argument(
        "--lp-tokens",
        type=int,
        default=0,
        help="LP token amount to associate with the position (required for live exits)",
    )
    parser.add_argument(
        "--phase-tag",
        choices=["hill", "cook", "drift"],
        help="Optional launch phase tag to append to the strategy label",
    )
    args = parser.parse_args()

    config = get_app_config()
    storage = SQLiteStorage(config.storage.database_path)
    discovery = DiscoveryService(storage=storage, app_config=config)

    candidate_limit = config.strategy.max_candidates * 4
    universe = discovery.discover_universe(limit=candidate_limit)
    token = _resolve_token(storage, universe, discovery, args.mint, candidate_limit)
    allocation = args.base_quantity * args.price_usd
    pool_address = args.pool_address
    if pool_address is None:
        for item in universe:
            if item.token.mint_address == token.mint_address and item.liquidity_event:
                pool_address = item.liquidity_event.pool_address
                position_fee = item.liquidity_event.pool_fee_bps
                break
        if pool_address is None:
            raise SystemExit("Unable to determine pool address automatically; specify --pool-address.")
    else:
        position_fee = args.fee_bps

    position = PortfolioPosition(
        token=token,
        pool_address=pool_address,
        venue=args.venue,
        allocation=allocation,
        entry_price=args.price_usd,
        created_at=datetime.utcnow(),
        base_quantity=args.base_quantity,
        quote_quantity=args.base_quantity * args.price_usd,
        strategy=args.strategy,
        pool_fee_bps=position_fee,
    )
    position.lp_token_amount = max(int(args.lp_tokens), 0)
    if args.lp_mint:
        position.lp_token_mint = args.lp_mint
    if args.phase_tag:
        position.strategy = f"{position.strategy}:{args.phase_tag}"
    storage.upsert_position(position)
    tag_msg = f" (phase={args.phase_tag})" if args.phase_tag else ""
    print(
        f"Seeded position for {token.symbol or token.mint_address} at {args.price_usd} USD"
        f" using strategy {position.strategy}{tag_msg}"
    )


if __name__ == "__main__":
    main()
