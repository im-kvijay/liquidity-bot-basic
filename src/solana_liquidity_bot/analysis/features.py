"""Feature engineering utilities that compute metrics for scoring."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, Optional

from ..config.settings import DataSourceConfig, get_app_config
from ..datalake.schemas import (
    LiquidityEvent,
    TokenMetadata,
    TokenOnChainStats,
    TokenRiskMetrics,
)
from ..utils.constants import STABLECOIN_MINTS


@dataclass(slots=True)
class TokenFeatures:
    """Aggregated set of features for a token."""

    token: TokenMetadata
    liquidity_depth_usd: float
    tvl_usd: float
    volume_24h_usd: float
    liquidity_velocity: float
    fee_apr: float
    holder_concentration: float
    top10_holder_concentration: float
    holder_count: int
    minted_minutes_ago: float
    dev_trust_score: float
    social_score: float
    price_usd: Optional[float]
    pool_address: Optional[str]
    quote_token_mint: Optional[str]
    base_liquidity: float
    quote_liquidity: float
    pool_fee_bps: int
    token_decimals: Optional[int]
    quote_token_decimals: Optional[int]
    volatility_score: float = 0.0
    dynamic_fee_bps: float = 0.0
    expected_daily_fee_usd: float = 0.0
    price_impact_bps: float = 0.0
    alpha_score: float = 0.0
    profit_factor: float = 0.0


def estimate_social_score(token: TokenMetadata) -> float:
    """Estimate a naive social score based on metadata richness."""

    score = 0.0
    if token.project_url:
        score += 0.25
    if token.creator:
        score += 0.2
    if token.social_handles:
        score += min(len(token.social_handles) * 0.15, 0.45)
    if len(token.sources) > 1:
        score += 0.1
    return min(score, 1.0)


def compute_dev_trust(stats: Optional[TokenOnChainStats]) -> float:
    if stats is None:
        return 0.3
    score = 0.3
    if stats.mint_authority is None:
        score += 0.3
    if stats.freeze_authority is None:
        score += 0.2
    if stats.top_holder_pct < 0.25:
        score += 0.1
    if stats.top10_holder_pct < 0.6:
        score += 0.1
    return min(max(score, 0.0), 1.0)


def _minutes_since(timestamp: Optional[datetime]) -> float:
    if timestamp is None:
        return 9999.0
    delta = datetime.now(timezone.utc) - timestamp
    return delta.total_seconds() / 60.0


def _compute_tvl(event: LiquidityEvent, stats: Optional[TokenOnChainStats]) -> float:
    if event.tvl_usd is not None:
        return float(event.tvl_usd)
    if event.price_usd:
        base_value = event.base_liquidity * event.price_usd
        quote_value = 0.0
        if event.quote_token_mint in STABLECOIN_MINTS:
            quote_value = event.quote_liquidity
        tvl = base_value + quote_value
    elif stats and stats.liquidity_estimate:
        tvl = stats.liquidity_estimate * 2
    else:
        tvl = event.base_liquidity + event.quote_liquidity
    return float(tvl)


def _compute_fee_apr(volume_24h_usd: float, tvl_usd: float, pool_fee_bps: int) -> float:
    if tvl_usd <= 0 or volume_24h_usd <= 0:
        return 0.0
    fee_rate = pool_fee_bps / 10_000
    daily_fees = volume_24h_usd * fee_rate
    return float((daily_fees / tvl_usd) * 365)


def _estimate_dynamic_fee_bps(pool_fee_bps: int, risk: Optional[TokenRiskMetrics]) -> float:
    if risk is None:
        return 0.0
    # Scale dynamic fee potential with observed volatility (cap at 2.5x base fee)
    multiplier = min(risk.volatility_score * 0.25, 1.5)
    return float(pool_fee_bps * multiplier)


def _estimate_price_impact_bps(tvl_usd: float, notional_usd: float = 1_000.0) -> float:
    if tvl_usd <= 0:
        return 5_000.0
    impact = (notional_usd / tvl_usd) * 10_000.0
    return float(min(max(impact, 0.0), 5_000.0))


def _compute_alpha_score(
    fee_apr: float,
    liquidity_velocity: float,
    volatility_score: float,
) -> float:
    if fee_apr <= 0:
        return 0.0
    # Combine fee efficiency with turnover and volatility bonus.
    volatility_bonus = 1.0 + min(volatility_score, 4.0) * 0.1
    return float(fee_apr * liquidity_velocity * volatility_bonus)


def build_features(
    tokens: Iterable[TokenMetadata],
    liquidity_events: Iterable[LiquidityEvent],
    onchain_stats: Iterable[TokenOnChainStats],
    risk_metrics: Optional[Dict[str, TokenRiskMetrics]] = None,
    config: DataSourceConfig | None = None,
) -> Dict[str, TokenFeatures]:
    """Compute features for all known tokens."""

    cfg = config or get_app_config().data_sources
    liquidity_by_token: Dict[str, LiquidityEvent] = {
        event.token.mint_address: event for event in liquidity_events
    }
    stats_by_token: Dict[str, TokenOnChainStats] = {
        stat.token.mint_address: stat for stat in onchain_stats
    }
    risk_by_token = risk_metrics or {}

    features: Dict[str, TokenFeatures] = {}
    for token in tokens:
        event = liquidity_by_token.get(token.mint_address)
        stats = stats_by_token.get(token.mint_address)
        if not event:
            continue
        risk = risk_by_token.get(token.mint_address)

        tvl_usd = _compute_tvl(event, stats)
        liquidity_depth = tvl_usd
        if liquidity_depth < cfg.min_liquidity_usd:
            liquidity_depth *= 0.5

        volume_24h_usd = float(event.volume_24h_usd or 0.0)
        liquidity_velocity = volume_24h_usd / max(tvl_usd, 1.0)
        fee_apr = _compute_fee_apr(volume_24h_usd, tvl_usd, event.pool_fee_bps)
        minted_minutes_ago = _minutes_since(stats.minted_at if stats else None)
        volatility_score = risk.volatility_score if risk else liquidity_velocity
        dynamic_fee_bps = _estimate_dynamic_fee_bps(event.pool_fee_bps, risk)
        expected_daily_fee_usd = volume_24h_usd * (
            (event.pool_fee_bps + dynamic_fee_bps) / 10_000.0
        )
        price_impact_bps = _estimate_price_impact_bps(tvl_usd)
        alpha_score = _compute_alpha_score(fee_apr, liquidity_velocity, volatility_score)
        profit_factor = (
            expected_daily_fee_usd / max(tvl_usd, 1.0) * 365.0 if tvl_usd > 0 else 0.0
        )

        features[token.mint_address] = TokenFeatures(
            token=token,
            liquidity_depth_usd=liquidity_depth,
            tvl_usd=tvl_usd,
            volume_24h_usd=volume_24h_usd,
            liquidity_velocity=liquidity_velocity,
            fee_apr=fee_apr,
            holder_concentration=stats.top_holder_pct if stats else 1.0,
            top10_holder_concentration=stats.top10_holder_pct if stats else 1.0,
            holder_count=stats.holder_count if stats else 0,
            minted_minutes_ago=minted_minutes_ago,
            dev_trust_score=compute_dev_trust(stats),
            social_score=estimate_social_score(token),
            price_usd=event.price_usd,
            pool_address=event.pool_address,
            quote_token_mint=event.quote_token_mint,
            base_liquidity=float(event.base_liquidity),
            quote_liquidity=float(event.quote_liquidity),
            pool_fee_bps=event.pool_fee_bps,
            token_decimals=stats.decimals if stats else token.decimals,
            quote_token_decimals=None,
            volatility_score=float(volatility_score),
            dynamic_fee_bps=float(dynamic_fee_bps),
            expected_daily_fee_usd=float(expected_daily_fee_usd),
            price_impact_bps=float(price_impact_bps),
            alpha_score=float(alpha_score),
            profit_factor=float(profit_factor),
        )
    return features


__all__ = [
    "TokenFeatures",
    "build_features",
    "estimate_social_score",
    "compute_dev_trust",
]
