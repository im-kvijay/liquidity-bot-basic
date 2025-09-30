"""Shared heuristics for identifying profitable DAMM v2 launch pools."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from ..config.settings import LaunchSniperConfig
from ..datalake.schemas import DammLaunchRecord, DammPoolSnapshot, TokenUniverseEntry
from ..utils.constants import STABLECOIN_MINTS

SOL_MINT = "So11111111111111111111111111111111111111112"


def evaluate_launch_candidate(
    entry: TokenUniverseEntry,
    config: LaunchSniperConfig,
    *,
    now: Optional[datetime] = None,
    sol_price: Optional[float] = None,
) -> Optional[DammLaunchRecord]:
    if not config.enabled:
        return None
    if entry.liquidity_event is None or entry.risk is None:
        return None

    pool = _select_launch_pool(entry)
    event = entry.liquidity_event
    risk = entry.risk

    created_at = pool.created_at if pool and pool.created_at else event.timestamp
    reference_time = now or datetime.now(timezone.utc)
    if getattr(reference_time, 'tzinfo', None) is not None:
        reference_time = reference_time.astimezone(timezone.utc).replace(tzinfo=None)
    if created_at is not None and getattr(created_at, 'tzinfo', None) is not None:
        created_at = created_at.astimezone(timezone.utc).replace(tzinfo=None)
    if created_at is None:
        created_at = reference_time
    age_seconds = max((reference_time - created_at).total_seconds(), 0.0)
    age_minutes = age_seconds / 60.0
    if age_minutes > config.max_age_minutes:
        return None

    liquidity = risk.liquidity_usd or 0.0
    volume = risk.volume_24h_usd or 0.0
    min_liquidity = config.min_liquidity_usd
    min_volume = getattr(config, "min_volume_24h_usd", config.min_liquidity_usd)
    if age_minutes <= getattr(config, "early_age_minutes", config.max_age_minutes):
        min_liquidity = min(min_liquidity, getattr(config, "early_min_liquidity_usd", min_liquidity))
        min_volume = min(min_volume, getattr(config, "early_min_volume_24h_usd", min_volume))
    if liquidity <= 0 or volume < 0:
        return None
    if liquidity < min_liquidity or liquidity > config.max_liquidity_usd:
        return None
    if volume < min_volume:
        return None

    if risk.top_holder_pct < config.min_top_holder_pct or risk.top_holder_pct > config.max_top_holder_pct:
        return None
    if (
        risk.top10_holder_pct < config.min_top10_holder_pct
        or risk.top10_holder_pct > config.max_top10_holder_pct
    ):
        return None

    def _segment_within(value: Optional[float], limit: float) -> bool:
        if value is None:
            return config.allow_missing_holder_segments
        return value <= limit

    if not _segment_within(risk.dev_holding_pct, config.max_dev_holding_pct):
        return None
    if not _segment_within(risk.sniper_holding_pct, config.max_sniper_holding_pct):
        return None
    if not _segment_within(risk.insider_holding_pct, config.max_insider_holding_pct):
        return None
    if not _segment_within(risk.bundler_holding_pct, config.max_bundler_holding_pct):
        return None

    scheduler_mode = pool.fee_scheduler_mode if pool else None
    if config.require_fee_scheduler and scheduler_mode is None:
        return None
    if (
        config.allowed_scheduler_modes
        and scheduler_mode is not None
        and scheduler_mode.lower() not in {mode.lower() for mode in config.allowed_scheduler_modes}
    ):
        return None

    current_fee_bps = _current_fee_bps(event, pool)
    if current_fee_bps < config.min_current_fee_bps:
        return None

    initial_fee_bps = event.pool_fee_bps if event.pool_fee_bps is not None else current_fee_bps
    if config.max_initial_fee_bps is not None and initial_fee_bps > config.max_initial_fee_bps:
        return None

    start_fee_bps = pool.fee_scheduler_start_bps if pool else current_fee_bps
    min_fee_bps = pool.fee_scheduler_min_bps if pool else event.pool_fee_bps
    base_fee_bps = min_fee_bps or start_fee_bps or current_fee_bps
    if base_fee_bps and base_fee_bps < config.min_base_fee_bps:
        return None

    fee_rate = current_fee_bps / 10_000
    fee_yield = (volume * fee_rate) / max(liquidity, 1.0)
    if fee_yield < config.min_fee_yield:
        return None

    price_usd = _resolve_price_usd(entry, sol_price)
    market_cap_usd = None
    stats = entry.stats
    max_market_cap = config.max_market_cap_usd
    if config.early_max_market_cap_usd and age_minutes <= getattr(config, "early_age_minutes", config.max_age_minutes):
        max_market_cap = max(max_market_cap, config.early_max_market_cap_usd)

    if stats and price_usd is not None and stats.total_supply > 0:
        market_cap_usd = price_usd * stats.total_supply
        if market_cap_usd > max_market_cap:
            return None

    allocation_cap = None
    symbol = (entry.token.symbol or "").upper()
    name = (entry.token.name or "").upper()
    if any(keyword.upper() in symbol or keyword.upper() in name for keyword in config.bonk_keywords):
        allocation_cap = config.bonk_allocation_sol
    launchpad = ""
    if entry.liquidity_event and entry.liquidity_event.launchpad:
        launchpad = entry.liquidity_event.launchpad.lower()
    if launchpad and any(keyword.lower() in launchpad for keyword in config.bonk_launchpads):
        allocation_cap = config.bonk_allocation_sol if allocation_cap is None else min(allocation_cap, config.bonk_allocation_sol)

    return DammLaunchRecord(
        mint_address=entry.token.mint_address,
        pool_address=event.pool_address,
        fee_bps=current_fee_bps,
        liquidity_usd=liquidity,
        volume_24h_usd=volume,
        fee_yield=fee_yield,
        age_seconds=age_seconds,
        price_usd=price_usd,
        market_cap_usd=market_cap_usd,
        fee_scheduler_mode=scheduler_mode,
        fee_scheduler_current_bps=current_fee_bps,
        fee_scheduler_start_bps=start_fee_bps,
        fee_scheduler_min_bps=min_fee_bps,
        allocation_cap_sol=allocation_cap,
        recorded_at=reference_time,
    )


def _select_launch_pool(entry: TokenUniverseEntry) -> Optional[DammPoolSnapshot]:
    if entry.liquidity_event:
        for pool in entry.damm_pools:
            if pool.address == entry.liquidity_event.pool_address:
                return pool
    if entry.damm_pools:
        return max(
            entry.damm_pools,
            key=lambda pool: (pool.tvl_usd or (pool.base_token_amount + pool.quote_token_amount)),
        )
    return None


def _current_fee_bps(event, pool: Optional[DammPoolSnapshot]) -> int:
    if pool and pool.fee_scheduler_current_bps is not None:
        return int(pool.fee_scheduler_current_bps)
    if event.pool_fee_bps is not None:
        return int(event.pool_fee_bps)
    return 0


def _resolve_price_usd(entry: TokenUniverseEntry, sol_price: Optional[float]) -> Optional[float]:
    event = entry.liquidity_event
    if event is None:
        return None
    if event.price_usd:
        return event.price_usd
    quote_mint = event.quote_token_mint
    if quote_mint in STABLECOIN_MINTS:
        if event.base_liquidity > 0:
            return event.quote_liquidity / max(event.base_liquidity, 1e-9)
        return None
    if quote_mint == SOL_MINT:
        sol_reference = sol_price if sol_price is not None else 150.0
        if event.base_liquidity > 0:
            return (event.quote_liquidity / max(event.base_liquidity, 1e-9)) * sol_reference
    return None


__all__ = ["evaluate_launch_candidate"]
