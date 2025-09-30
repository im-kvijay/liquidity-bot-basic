"""Scoring logic for determining promising tokens."""

from __future__ import annotations

from typing import Dict, Iterable

from ..config.settings import StrategyConfig, get_app_config
from ..datalake.schemas import TokenScore
from .features import TokenFeatures


class ScoreEngine:
    """Deterministic scoring engine with transparent metrics."""

    def __init__(self, strategy_config: StrategyConfig | None = None) -> None:
        self._config = strategy_config or get_app_config().strategy

    def _scale(self, value: float, target: float) -> float:
        if target <= 0:
            return 0.0
        return max(0.0, min(value / target, 1.5))

    def score(self, features: Iterable[TokenFeatures]) -> Dict[str, TokenScore]:
        results: Dict[str, TokenScore] = {}
        for feature in features:
            liquidity_component = min(
                self._scale(feature.tvl_usd, self._config.liquidity_target_usd), 1.0
            )
            volume_component = min(
                self._scale(feature.volume_24h_usd, self._config.volume_target_usd), 1.0
            )
            velocity_component = min(
                self._scale(
                    feature.liquidity_velocity, self._config.liquidity_velocity_target
                ),
                1.2,
            )
            fee_component = min(
                self._scale(feature.fee_apr, self._config.fee_apr_target), 1.2
            )
            volatility_component = min(
                self._scale(feature.volatility_score, self._config.volatility_target), 1.5
            )
            dynamic_fee_component = min(
                self._scale(feature.dynamic_fee_bps, self._config.dynamic_fee_target_bps), 1.5
            )
            alpha_component = min(
                self._scale(feature.alpha_score, self._config.alpha_score_target), 1.5
            )
            profit_component = min(
                self._scale(feature.profit_factor, self._config.fee_apr_target * 2.0), 1.5
            )
            holder_component = max(
                0.0,
                1.0
                - (feature.holder_concentration / max(self._config.max_single_holder_pct, 1e-6)),
            )
            top10_component = max(
                0.0,
                1.0
                - (feature.top10_holder_concentration / max(self._config.max_top10_holder_pct, 1e-6)),
            )
            recency_component = max(
                0.0,
                1.0 - (feature.minted_minutes_ago / max(self._config.max_token_age_minutes, 1)),
            )
            dev_component = max(0.0, min(feature.dev_trust_score, 1.0))
            social_component = max(0.0, min(feature.social_score, 1.0))

            momentum_weight = (
                self._config.momentum_liquidity_weight
                + self._config.momentum_fee_weight
                + self._config.momentum_velocity_weight
            )
            momentum_weight = momentum_weight or 1.0
            momentum_signal = (
                (volume_component * self._config.momentum_liquidity_weight)
                + (fee_component * self._config.momentum_fee_weight)
                + (velocity_component * self._config.momentum_velocity_weight)
            ) / momentum_weight
            momentum_component = min(
                max(
                    momentum_signal
                    / max(self._config.momentum_score_cap, 1e-6),
                    0.0,
                ),
                1.0,
            )
            momentum_bonus = momentum_component * self._config.momentum_score_weight

            concentration_penalty = 0.0
            if feature.holder_concentration > self._config.max_single_holder_pct > 0:
                concentration_penalty += (
                    feature.holder_concentration - self._config.max_single_holder_pct
                ) / max(1.0 - self._config.max_single_holder_pct, 1e-6)
            if feature.top10_holder_concentration > self._config.max_top10_holder_pct > 0:
                concentration_penalty += (
                    feature.top10_holder_concentration - self._config.max_top10_holder_pct
                ) / max(1.0 - self._config.max_top10_holder_pct, 1e-6)
            concentration_penalty = max(0.0, min(concentration_penalty, 2.0))
            risk_penalty = min(
                concentration_penalty * self._config.concentration_penalty_weight,
                0.6,
            )
            price_impact_penalty = 0.0
            if feature.price_impact_bps > self._config.price_impact_tolerance_bps > 0:
                price_impact_penalty = min(
                    (feature.price_impact_bps - self._config.price_impact_tolerance_bps)
                    / max(self._config.price_impact_tolerance_bps, 1e-6),
                    1.0,
                )
            risk_penalty = min(risk_penalty + price_impact_penalty * 0.3, 0.8)

            total_score = (
                (0.18 * liquidity_component)
                + (0.10 * volume_component)
                + (0.18 * fee_component)
                + (0.08 * velocity_component)
                + (0.09 * volatility_component)
                + (0.08 * dynamic_fee_component)
                + (0.08 * alpha_component)
                + (0.07 * profit_component)
                + (0.05 * holder_component)
                + (0.03 * top10_component)
                + (0.04 * recency_component)
                + (0.02 * dev_component)
                + (0.02 * social_component)
            )
            total_score = max(0.0, min(total_score + momentum_bonus - risk_penalty, 1.2))

            reasoning = (
                f"liq={liquidity_component:.2f}, volume={volume_component:.2f}, fee={fee_component:.2f}, "
                f"vel={velocity_component:.2f}, vol_score={volatility_component:.2f}, dyn_fee={dynamic_fee_component:.2f}, "
                f"alpha={alpha_component:.2f}, profit={profit_component:.2f}, holder={holder_component:.2f}, "
                f"top10={top10_component:.2f}, recency={recency_component:.2f}, dev={dev_component:.2f}, social={social_component:.2f}, "
                f"mom={momentum_component:.2f}, risk_penalty={risk_penalty:.2f}"
            )
            metrics = {
                "liquidity_component": liquidity_component,
                "volume_component": volume_component,
                "liquidity_velocity_component": velocity_component,
                "volatility_component": volatility_component,
                "fee_component": fee_component,
                "dynamic_fee_component": dynamic_fee_component,
                "alpha_component": alpha_component,
                "profit_component": profit_component,
                "holder_component": holder_component,
                "top10_component": top10_component,
                "recency_component": recency_component,
                "dev_component": dev_component,
                "social_component": social_component,
                "tvl_usd": feature.tvl_usd,
                "volume_24h_usd": feature.volume_24h_usd,
                "liquidity_velocity": feature.liquidity_velocity,
                "fee_apr": feature.fee_apr,
                "expected_daily_fee_usd": feature.expected_daily_fee_usd,
                "dynamic_fee_bps": feature.dynamic_fee_bps,
                "volatility_score": feature.volatility_score,
                "alpha_score": feature.alpha_score,
                "profit_factor": feature.profit_factor,
                "holder_concentration": feature.holder_concentration,
                "top10_holder_concentration": feature.top10_holder_concentration,
                "holder_count": feature.holder_count,
                "minted_minutes_ago": feature.minted_minutes_ago,
                "dev_trust_score": feature.dev_trust_score,
                "social_score": feature.social_score,
                "momentum_signal": momentum_signal,
                "momentum_component": momentum_component,
                "momentum_bonus": momentum_bonus,
                "concentration_penalty": concentration_penalty,
                "risk_penalty": risk_penalty,
                "price_impact_bps": feature.price_impact_bps,
                "price_impact_penalty": price_impact_penalty,
                "pool_address": feature.pool_address or "",
                "quote_token_mint": feature.quote_token_mint or "",
                "base_liquidity": feature.base_liquidity,
                "quote_liquidity": feature.quote_liquidity,
                "pool_fee_bps": feature.pool_fee_bps,
                "token_decimals": float(feature.token_decimals or feature.token.decimals),
                "quote_token_decimals": float(feature.quote_token_decimals or 0),
            }
            if feature.price_usd is not None:
                metrics["price_usd"] = feature.price_usd

            results[feature.token.mint_address] = TokenScore(
                token=feature.token,
                score=total_score,
                reasoning=reasoning,
                metrics=metrics,
            )
        return results


__all__ = ["ScoreEngine"]
