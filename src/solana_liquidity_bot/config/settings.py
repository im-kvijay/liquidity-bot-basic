"""Comprehensive configuration management for the trading bot."""

from __future__ import annotations

import os
import tomllib
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, TypeVar, Union, cast, get_origin

try:  # pragma: no cover - optional dependency branch retained for compatibility
    from pydantic import AnyHttpUrl, BaseModel, Field, field_validator, model_validator
    from pydantic_settings import BaseSettings, SettingsConfigDict

    HAVE_PYDANTIC = True
except ModuleNotFoundError:  # pragma: no cover - exercised when pydantic is unavailable
    HAVE_PYDANTIC = False


DEFAULT_CONFIG_FILE = Path("config/app.toml")
CONFIG_FILE_ENV_VAR = "APP_CONFIG_FILE"
MODE_ENV_VAR = "BOT_MODE"


class AppMode(str, Enum):
    """Supported runtime modes."""

    DRY_RUN = "dry_run"
    LIVE = "live"
    TESTNET = "testnet"


class MarkPriceSource(str, Enum):
    """Options for valuing inventory marks."""

    ORACLE = "oracle"
    MIDPRICE = "midprice"
    TWAP = "twap"
    VWAP = "vwap"


class DashboardAuthMode(str, Enum):
    """Authentication styles supported by the dashboard."""

    READ_ONLY = "read_only"
    AUTHENTICATED = "authenticated"


def _resolve_config_path() -> Path:
    env_value = os.getenv(CONFIG_FILE_ENV_VAR)
    if env_value:
        candidate = Path(env_value)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        return candidate
    default = Path.cwd() / DEFAULT_CONFIG_FILE
    return default


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {**base}
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(cast(Dict[str, Any], result[key]), value)
        else:
            result[key] = value
    return result


def _select_profile(data: Dict[str, Any]) -> Dict[str, Any]:
    if not data:
        return {}
    base_section = cast(Dict[str, Any], data.get("default", {}))
    requested_mode = os.getenv(MODE_ENV_VAR)
    if not requested_mode:
        mode_section = base_section.get("mode")
        if isinstance(mode_section, dict):
            requested_mode = cast(str, mode_section.get("active", AppMode.DRY_RUN.value))
        elif isinstance(mode_section, str):
            requested_mode = mode_section
    requested_mode = requested_mode or AppMode.DRY_RUN.value
    requested_mode = requested_mode.lower()

    if requested_mode == "default" and "default" in data:
        requested_mode = AppMode.DRY_RUN.value

    merged = base_section
    if requested_mode in data:
        merged = _deep_merge(base_section, cast(Dict[str, Any], data[requested_mode]))
    elif base_section:
        merged = base_section
    else:
        merged = data
    return merged


def _load_toml_config() -> Tuple[Dict[str, Any], Optional[Path]]:
    path = _resolve_config_path()
    if not path.exists():
        return {}, None
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    if not isinstance(payload, dict):
        return {}, path
    merged = _select_profile(payload)
    if not isinstance(merged, dict):
        return {}, path
    merged = {k: v for k, v in merged.items()}
    mode_section = merged.get("mode")
    if isinstance(mode_section, dict):
        mode_section = dict(mode_section)
        mode_section.setdefault("config_file", str(path))
        merged["mode"] = mode_section
    else:
        merged["mode"] = {"config_file": str(path)}
    return merged, path


def _coerce_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _coerce_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


T = TypeVar("T")


def _convert_value(expected_type: Type[T], raw_value: str) -> T:
    origin = get_origin(expected_type)
    if origin is list or origin is List:
        item_type: Type[Any] = str
        args = getattr(expected_type, "__args__", None)
        if args:
            item_type = cast(Type[Any], args[0])
        return cast(T, [_convert_value(item_type, item.strip()) for item in _coerce_list(raw_value)])
    if origin is Union:
        args = [arg for arg in getattr(expected_type, "__args__", []) if arg is not type(None)]
        if len(args) == 1:
            return _convert_value(cast(Type[T], args[0]), raw_value)
    if isinstance(expected_type, type) and issubclass(expected_type, Enum):
        return cast(T, expected_type(raw_value))
    if expected_type is str or expected_type is Any:
        return cast(T, raw_value)
    if expected_type is int:
        return cast(T, int(raw_value))
    if expected_type is float:
        return cast(T, float(raw_value))
    if expected_type is bool:
        return cast(T, _coerce_bool(raw_value))
    if expected_type is Path:
        return cast(T, Path(raw_value))
    if expected_type is AppMode:
        return cast(T, AppMode(raw_value))
    if expected_type is MarkPriceSource:
        return cast(T, MarkPriceSource(raw_value))
    if expected_type is DashboardAuthMode:
        return cast(T, DashboardAuthMode(raw_value))
    return cast(T, raw_value)


def _apply_dict_to_dataclass(instance: Any, payload: Dict[str, Any]) -> None:
    if not is_dataclass(instance):
        raise TypeError("Configuration object must be a dataclass")
    if type(instance).__name__ == "StrategyConfig" and "launch" not in payload and "launch_sniper" in payload:
        payload = dict(payload)
        payload["launch"] = payload["launch_sniper"]
    for field_info in fields(instance):
        key = field_info.name
        if key not in payload:
            continue
        value = payload[key]
        current = getattr(instance, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply_dict_to_dataclass(current, value)
        else:
            if isinstance(current, Enum) and isinstance(value, str):
                setattr(instance, key, current.__class__(value))
            else:
                setattr(instance, key, value)


def _apply_env_to_dataclass(instance: Any, prefix: Tuple[str, ...] = ()) -> None:
    if not is_dataclass(instance):
        return
    for field_info in fields(instance):
        key = field_info.name
        env_key_candidates = ["__".join((*prefix, key)).upper(), "_".join((*prefix, key)).upper()]
        raw_value: Optional[str] = None
        for candidate in env_key_candidates:
            raw_value = os.getenv(candidate)
            if raw_value is not None:
                break
        current_value = getattr(instance, key)
        if raw_value is None and is_dataclass(current_value):
            _apply_env_to_dataclass(current_value, (*prefix, key))
            continue
        if raw_value is None:
            continue
        expected_type = field_info.type
        try:
            coerced = _convert_value(expected_type, raw_value)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Unable to parse environment override for {'__'.join((*prefix, key))}: {raw_value}") from exc
        setattr(instance, key, coerced)


if HAVE_PYDANTIC:

    class RPCConfig(BaseModel):
        """RPC configuration for Solana endpoints."""

        primary_url: AnyHttpUrl = Field(default="https://api.mainnet-beta.solana.com")
        fallback_urls: List[AnyHttpUrl] = Field(default_factory=lambda: ["https://solana-api.projectserum.com"])
        request_timeout: float = Field(default=12.0, ge=1.0, le=60.0)
        commitment: str = Field(default="confirmed")
        request_concurrency: int = Field(default=4, ge=1, le=64)

        @field_validator("fallback_urls", mode="before")
        @classmethod
        def _unique_urls(cls, value: Iterable[AnyHttpUrl]) -> List[AnyHttpUrl]:
            seen: set[AnyHttpUrl] = set()
            unique: List[AnyHttpUrl] = []
            for url in value:
                if url not in seen:
                    unique.append(url)
                    seen.add(url)
            return unique

        @field_validator("request_timeout", mode="before")
        @classmethod
        def _parse_request_timeout(cls, value) -> float:
            if isinstance(value, str):
                return float(value)
            return value


    class DataSourceConfig(BaseModel):
        """Configurations for external discovery sources."""

        enable_axiom: bool = True
        enable_pumpfun: bool = True
        enable_solana_token_list: bool = False
        enable_meteora_registry: bool = True
        enable_rocketscan: bool = True
        axiom_base_url: AnyHttpUrl = Field(default="https://api.axiom.xyz")
        axiom_api_key: Optional[str] = None
        pumpfun_base_url: AnyHttpUrl = Field(default="https://pump.fun/api")
        pumpfun_api_key: Optional[str] = None
        damm_base_url: AnyHttpUrl = Field(default="https://dammv2-api.meteora.ag")
        damm_devnet_base_url: AnyHttpUrl = Field(default="https://dammv2-api.devnet.meteora.ag")
        damm_pool_endpoint: str = Field(default="/pools")
        damm_page_limit: int = Field(default=50, ge=1)
        dlmm_base_url: AnyHttpUrl = Field(default="https://dlmm-api.meteora.ag")
        dlmm_devnet_base_url: AnyHttpUrl = Field(default="https://devnet-dlmm-api.meteora.ag")
        dlmm_pool_endpoint: str = Field(default="/pair/all")
        dlmm_page_limit: int = Field(default=50, ge=1)
        meteora_registry_url: AnyHttpUrl = Field(default="https://dlmm-api.meteora.ag/api/v1/pairs")
        price_oracle_url: AnyHttpUrl = Field(default="https://lite-api.jup.ag/")
        price_oracle_urls: Optional[List[AnyHttpUrl]] = None
        fallback_price_usd: Optional[float] = Field(default=None, ge=0.0)
        price_feed_overrides: Dict[str, str] = Field(default_factory=dict)
        solana_token_list_url: AnyHttpUrl = Field(
            default="https://token-list-api.solana.com/metadata.json"
        )
        price_oracle_use_pyth_fallback: bool = Field(default=True)
        price_oracle_jupiter_rate_limit_per_minute: int = Field(default=55, ge=1, le=600)
        rocketscan_base_url: AnyHttpUrl = Field(default="https://rocketscan.fun/api")
        rocketscan_timeout: float = Field(default=8.0, ge=1.0, le=30.0)
        rocketscan_max_results: int = Field(default=75, ge=1, le=200)
        rocketscan_cache_ttl_seconds: int = Field(default=180, ge=0)
        rocketscan_max_workers: int = Field(default=8, ge=1, le=32)
        rocketscan_max_age_minutes: int = Field(default=60, ge=1)
        http_timeout: float = Field(default=10.0, ge=1.0, le=45.0)
        min_liquidity_usd: float = Field(default=5_000.0, ge=0.0)
        cache_ttl_seconds: int = Field(default=120, ge=0)


    class ModeConfig(BaseModel):
        """Runtime mode and operational toggles."""

        active: AppMode = Field(default=AppMode.DRY_RUN)
        cluster: str = Field(default="mainnet-beta")
        dry_run_seed: int = Field(default=0, ge=0)
        persist_dry_run_state: bool = True
        config_file: Optional[Path] = None


    class DammVenueConfig(BaseModel):
        """Configuration for Meteora DAMM v2."""

        enabled: bool = True
        program_id: str = Field(default="Eo7WjKq67rjJQSZxS6z3YkapzY3eMj6Xy8X5EQVn5UaB")
        min_liquidity_usd: float = Field(default=20_000.0, ge=0.0)
        min_depth_levels: int = Field(default=3, ge=1)
        max_slippage_bps: int = Field(default=80, ge=1, le=1_000)
        health_check_interval_seconds: int = Field(default=30, ge=5)
        cache_ttl_seconds: int = Field(default=45, ge=0)
        default_quote_mint: str = Field(default="EPjFWdd5AufqSSqeM2qN1xzybapC8WdGCr3vZ9V4Wrh")
        quote_mint_allowlist: List[str] = Field(
            default_factory=lambda: [
                "EPjFWdd5AufqSSqeM2qN1xzybapC8WdGCr3vZ9V4Wrh",
                "Es9vMFrzaCER1ZXS9dZxXn6vufGAaHo9dZYkTf5ZNVY",
            ]
        )


    class DlmmVenueConfig(BaseModel):
        """Configuration for Meteora DLMM venues."""

        enabled: bool = True
        program_id: str = Field(default="LBUZKhRxPF3XUpBCjp4YzTKgLccjZhTSDM9YuVaPwxo")
        min_liquidity_usd: float = Field(default=25_000.0, ge=0.0)
        min_active_bins: int = Field(default=6, ge=1)
        max_slippage_bps: int = Field(default=90, ge=1, le=1_000)
        health_check_interval_seconds: int = Field(default=30, ge=5)
        cache_ttl_seconds: int = Field(default=30, ge=0)
        required_oracle_confidence_bps: int = Field(default=50, ge=1)
        default_quote_mint: str = Field(default="EPjFWdd5AufqSSqeM2qN1xzybapC8WdGCr3vZ9V4Wrh")


    class VenueConfig(BaseModel):
        """Configuration container for supported venues."""

        damm: DammVenueConfig = Field(default_factory=DammVenueConfig)
        dlmm: DlmmVenueConfig = Field(default_factory=DlmmVenueConfig)


    class RouterConfig(BaseModel):
        """Weights and toggles used by the execution router."""

        liquidity_weight: float = Field(default=0.45, ge=0.0, le=1.0)
        fee_weight: float = Field(default=0.2, ge=0.0, le=1.0)
        volatility_weight: float = Field(default=0.15, ge=0.0, le=1.0)
        depth_weight: float = Field(default=0.1, ge=0.0, le=1.0)
        rebate_weight: float = Field(default=0.1, ge=0.0, le=1.0)
        fill_weight: float = Field(default=0.08, ge=0.0, le=1.0)
        require_health_checks: bool = True
        prefer_maker_routes: bool = True
        slippage_safety_bps: int = Field(default=40, ge=0)


    class ExecutionConfig(BaseModel):
        """Transaction queue behaviour."""

        queue_capacity: int = Field(default=128, ge=1)
        max_concurrency: int = Field(default=4, ge=1)
        retry_backoff_seconds: float = Field(default=0.75, ge=0.0)
        max_retry_attempts: int = Field(default=3, ge=1)
        jitter_seconds: float = Field(default=0.25, ge=0.0)
        batch_size: int = Field(default=4, ge=1)
        # Production-ready execution controls
        dedupe_requests: bool = Field(default=True)
        cache_ttl_seconds: int = Field(default=300, ge=0)
        priority_fee_policy: str = Field(default="adaptive_percentile")
        priority_fee_percentile: float = Field(default=0.75, ge=0.0, le=1.0)
        max_priority_fee_budget_usd: float = Field(default=0.05, ge=0.0)


    class TradingConfig(BaseModel):
        """Trading system controls and toggles."""

        enable_live: bool = Field(default=False)
        dry_run: bool = Field(default=True)
        min_edge_bps: int = Field(default=60, ge=1)

    class PnLConfig(BaseModel):
        """Parameters for PnL calculation and persistence."""

        mark_price_source: MarkPriceSource = Field(default=MarkPriceSource.ORACLE)
        oracle_staleness_seconds: int = Field(default=45, ge=1)
        twap_window_seconds: int = Field(default=300, ge=60)
        vwap_window_seconds: int = Field(default=900, ge=60)
        inventory_valuation_interval_seconds: int = Field(default=15, ge=1)
        persist_frequency_seconds: int = Field(default=30, ge=5)
        # Production-ready analytics
        enable_dry_run_pnl: bool = Field(default=True)
        pnl_update_interval_seconds: int = Field(default=30, ge=1)
        position_level_pnl: bool = Field(default=True)
        profitability_metrics: bool = Field(default=True)
        performance_summary_every_n_trades: int = Field(default=10, ge=1)


    class RiskLimitsConfig(BaseModel):
        """Global and per-market risk limits."""

        max_global_notional_usd: float = Field(default=150_000.0, ge=0.0)
        max_market_notional_usd: float = Field(default=25_000.0, ge=0.0)
        max_position_notional_usd: float = Field(default=10_000.0, ge=0.0)
        max_inventory_pct: float = Field(default=0.3, ge=0.0, le=1.0)
        max_open_orders: int = Field(default=25, ge=0)
        daily_loss_limit_usd: float = Field(default=7_500.0, ge=0.0)
        max_slippage_bps: int = Field(default=90, ge=1)
        # Production-ready risk controls
        max_position_usd: float = Field(default=250.0, ge=0.0)
        max_notional_usd: float = Field(default=1000.0, ge=0.0)
        max_daily_loss_pct: float = Field(default=2.0, ge=0.0, le=100.0)
        max_token_exposure_pct: float = Field(default=25.0, ge=0.0, le=100.0)
        emergency_drawdown_pct: float = Field(default=25.0, ge=0.0, le=100.0)
        cool_off_seconds: int = Field(default=600, ge=0)


    class CircuitBreakerConfig(BaseModel):
        """Circuit breaker thresholds and recovery rules."""

        max_drawdown_pct: float = Field(default=0.2, ge=0.0, le=1.0)
        max_reject_rate: float = Field(default=0.15, ge=0.0, le=1.0)
        max_volatility_pct: float = Field(default=0.5, ge=0.0, le=1.0)
        health_check_backoff_seconds: int = Field(default=180, ge=30)
        cooldown_seconds: int = Field(default=300, ge=60)


    class EventBusConfig(BaseModel):
        """Event bus tuning."""

        enabled: bool = True
        max_queue_size: int = Field(default=2_048, ge=128)
        publish_metrics: bool = True
        replay_on_startup: bool = True


    class DashboardConfig(BaseModel):
        """Dashboard runtime configuration."""

        host: str = Field(default="0.0.0.0")
        port: int = Field(default=8080, ge=1, le=65535)
        auth_mode: DashboardAuthMode = Field(default=DashboardAuthMode.READ_ONLY)
        allowed_origins: List[str] = Field(default_factory=lambda: ["*"])
        metrics_refresh_interval_seconds: int = Field(default=5, ge=1)
        enable_incident_view: bool = True
        read_only_token: Optional[str] = None


    class TokenUniverseConfig(BaseModel):
        """Discovery, validation, and compliance controls."""

        min_liquidity_usd: float = Field(default=25_000.0, ge=0.0)
        min_volume_24h_usd: float = Field(default=75_000.0, ge=0.0)
        min_holder_count: int = Field(default=250, ge=0)
        max_holder_concentration_pct: float = Field(default=0.35, ge=0.0, le=1.0)
        max_top10_holder_pct: float = Field(default=0.7, ge=0.0, le=1.0)
        min_social_links: int = Field(default=1, ge=0)
        max_token_age_minutes: int = Field(default=720, ge=1)
        require_oracle_price: bool = True
        allow_unverified_token_list: bool = False
        deny_freeze_authority: bool = True
        deny_mints: List[str] = Field(default_factory=list)
        deny_creators: List[str] = Field(default_factory=list)
        deny_authorities: List[str] = Field(default_factory=list)
        suspicious_keywords: List[str] = Field(
            default_factory=lambda: ["rug", "honeypot", "scam", "exploit"]
        )
        min_decimals: int = Field(default=0, ge=0)
        max_decimals: int = Field(default=12, ge=0)
        aml_min_token_age_minutes: int = Field(default=15, ge=0)
        aml_min_holder_count: int = Field(default=25, ge=0)
        aml_max_single_holder_pct: float = Field(default=0.75, ge=0.0, le=1.0)
        aml_sanctioned_accounts: List[str] = Field(default_factory=list)
        autopause_liquidity_buffer: float = Field(default=0.2, ge=0.0, le=1.0)
        autopause_volume_buffer: float = Field(default=0.25, ge=0.0, le=1.0)
        autopause_recovery_minutes: int = Field(default=60, ge=1)
        tracked_quote_mints: List[str] = Field(
            default_factory=lambda: [
                "EPjFWdd5AufqSSqeM2qN1xzybapC8WdGCr3vZ9V4Wrh",
                "Es9vMFrzaCER1ZXS9dZxXn6vufGAaHo9dZYkTf5ZNVY",
            ]
        )
        allow_manual_override: bool = True
        discovery_batch_size: int = Field(default=200, ge=10)
        # Production-ready discovery thresholds
        early_min_liquidity_usd: float = Field(default=5000.0, ge=0.0)
        early_min_volume_24h_usd: float = Field(default=10000.0, ge=0.0)
        min_base_fee_bps: int = Field(default=200, ge=1, le=10000)
        max_candidates: int = Field(default=300, ge=1)


    class LaunchAllocationTier(BaseModel):
        max_liquidity_usd: float = Field(ge=0.0)
        allocation_sol: float = Field(ge=0.0)


    class LaunchSniperConfig(BaseModel):
        """Parameters guiding the high-fee DAMM launch strategy."""

        enabled: bool = True
        max_age_minutes: int = Field(default=30, ge=1)
        min_fee_bps: int = Field(default=800, ge=1, le=10_000)
        min_fee_yield: float = Field(default=0.0005, ge=0.0)
        min_liquidity_usd: float = Field(default=6_000.0, ge=0.0)
        early_min_liquidity_usd: float = Field(default=2_000.0, ge=0.0)
        max_liquidity_usd: float = Field(default=450_000.0, ge=0.0)
        min_top_holder_pct: float = Field(default=0.1, ge=0.0, le=1.0)
        max_top_holder_pct: float = Field(default=0.35, ge=0.0, le=1.0)
        min_top10_holder_pct: float = Field(default=0.1, ge=0.0, le=1.0)
        max_top10_holder_pct: float = Field(default=0.3, ge=0.0, le=1.0)
        max_market_cap_usd: float = Field(default=200_000.0, ge=0.0)
        early_age_minutes: int = Field(default=12, ge=1)
        early_max_market_cap_usd: float = Field(default=400_000.0, ge=0.0)
        min_volume_24h_usd: float = Field(default=10_000.0, ge=0.0)
        early_min_volume_24h_usd: float = Field(default=0.0, ge=0.0)
        min_base_fee_bps: int = Field(default=500, ge=1, le=10_000)
        min_current_fee_bps: int = Field(default=1_000, ge=1, le=10_000)
        high_fee_boost_bps: int = Field(default=800, ge=1, le=10_000)
        high_fee_allocation_multiplier: float = Field(default=1.2, ge=1.0, le=3.0)
        require_fee_scheduler: bool = Field(default=True)
        allowed_scheduler_modes: List[str] = Field(
            default_factory=lambda: ["linear", "exponential"]
        )
        bonk_keywords: List[str] = Field(default_factory=lambda: ["BONK"])
        bonk_allocation_sol: float = Field(default=0.05, ge=0.0)
        bonk_launchpads: List[str] = Field(default_factory=lambda: ["bonk", "bonkpad"])
        allocation_tiers: List[LaunchAllocationTier] = Field(
            default_factory=lambda: [
                LaunchAllocationTier(max_liquidity_usd=75_000.0, allocation_sol=0.05),
                LaunchAllocationTier(max_liquidity_usd=150_000.0, allocation_sol=0.1),
                LaunchAllocationTier(max_liquidity_usd=300_000.0, allocation_sol=0.2),
            ]
        )
        max_dev_holding_pct: float = Field(default=4.0, ge=0.0, le=100.0)
        max_sniper_holding_pct: float = Field(default=15.0, ge=0.0, le=100.0)
        max_insider_holding_pct: float = Field(default=15.0, ge=0.0, le=100.0)
        max_bundler_holding_pct: float = Field(default=20.0, ge=0.0, le=100.0)
        allow_missing_holder_segments: bool = Field(default=True)
        holder_segment_dampen_ratio: float = Field(default=0.75, ge=0.0, le=1.0)
        max_initial_fee_bps: Optional[int] = Field(default=None, ge=1, le=10_000)
        hill_phase_minutes: int = Field(default=25, ge=1)
        hill_min_fee_bps: int = Field(default=2_000, ge=0, le=10_000)
        cook_min_fee_bps: int = Field(default=1_200, ge=0, le=10_000)
        cook_extension_minutes: int = Field(default=10, ge=0)
        max_hold_minutes: int = Field(default=45, ge=1)
        exit_cooldown_seconds: int = Field(default=900, ge=0)
        max_decisions: int = Field(default=3, ge=1)
        momentum_velocity_threshold: float = Field(default=0.4, ge=0.0)
        momentum_min_fee_yield: float = Field(default=0.0007, ge=0.0)
        scale_up_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
        exit_fee_yield_floor: float = Field(default=0.0003, ge=0.0)
        exit_market_cap_floor: float = Field(default=50_000.0, ge=0.0)
        exit_strength_gain_pct: float = Field(default=0.45, ge=0.0)
        exit_trailing_drawdown_pct: float = Field(default=0.18, ge=0.0)
        dlmm_exit_tvl_threshold: float = Field(default=20_000.0, ge=0.0)
        fallback_sol_price: float = Field(default=150.0, ge=0.0)
        # Production-ready launch sniper controls
        require_existing_pool: bool = Field(default=True)
        prefer_damm: bool = Field(default=True)
        min_volume_1h_usd: float = Field(default=20000.0, ge=0.0)
        max_slippage_bps: int = Field(default=40, ge=1, le=1000)
        max_price_impact_bps: int = Field(default=30, ge=1, le=1000)
        take_profit_bps: int = Field(default=150, ge=1, le=1000)
        trailing_stop_bps: int = Field(default=75, ge=1, le=1000)
        stop_loss_bps: int = Field(default=80, ge=1, le=1000)
        dynamic_position_sizing: bool = Field(default=True)


    class StrategyConfig(BaseModel):
        """Strategy level allocations and heuristics."""

        @model_validator(mode="before")
        @classmethod
        def _alias_launch_config(cls, data: Any) -> Any:
            if isinstance(data, dict) and "launch" not in data and "launch_sniper" in data:
                data = dict(data)
                data["launch"] = data["launch_sniper"]
            return data

        max_positions: int = Field(default=10, ge=1)
        max_candidates: int = Field(default=300, ge=1)
        allocation_per_position: float = Field(default=500.0, ge=0.0)
        max_allocation_per_position: float = Field(default=2_000.0, ge=0.0)
        max_allocation_multiplier: float = Field(default=2.5, ge=0.5)
        portfolio_size: float = Field(default=5_000.0, ge=0.0)
        rebalance_interval_seconds: int = Field(default=1_800, ge=60)
        stop_loss_pct: float = Field(default=0.15, ge=0.0)
        take_profit_pct: float = Field(default=0.4, ge=0.0)
        min_liquidity_usd: float = Field(default=6_000.0, ge=0.0)
        min_volume_24h_usd: float = Field(default=20_000.0, ge=0.0)
        min_fee_apr: float = Field(default=0.08, ge=0.0)
        min_holder_count: int = Field(default=100, ge=0)
        max_single_holder_pct: float = Field(default=0.35, ge=0.0, le=1.0)
        max_top10_holder_pct: float = Field(default=0.75, ge=0.0, le=1.0)
        min_dev_trust_score: float = Field(default=0.3, ge=0.0)
        min_social_score: float = Field(default=0.1, ge=0.0)
        max_token_age_minutes: int = Field(default=240, ge=1)
        liquidity_target_usd: float = Field(default=25_000.0, ge=0.0)
        volume_target_usd: float = Field(default=100_000.0, ge=0.0)
        fee_apr_target: float = Field(default=0.5, ge=0.0)
        priority_fee_lamports: int = Field(default=5_000, ge=0)
        liquidity_velocity_target: float = Field(default=1.35, ge=0.0)
        momentum_threshold: float = Field(default=0.25, ge=0.0)
        momentum_liquidity_weight: float = Field(default=0.5, ge=0.0)
        momentum_fee_weight: float = Field(default=0.3, ge=0.0)
        momentum_velocity_weight: float = Field(default=0.2, ge=0.0)
        momentum_score_weight: float = Field(default=0.12, ge=0.0)
        momentum_score_cap: float = Field(default=1.5, ge=0.0)
        concentration_penalty_weight: float = Field(default=0.3, ge=0.0)
        momentum_priority_weight: float = Field(default=0.1, ge=0.0)
        volatility_target: float = Field(default=1.0, ge=0.0)
        dynamic_fee_target_bps: float = Field(default=20.0, ge=0.0)
        alpha_score_target: float = Field(default=1.0, ge=0.0)
        price_impact_tolerance_bps: float = Field(default=60.0, ge=0.0)
        maker_min_spread_bps: float = Field(default=24.0, ge=0.0)
        maker_max_spread_bps: float = Field(default=135.0, ge=0.0)
        maker_volatility_spread_factor: float = Field(default=40.0, ge=0.0)
        maker_inventory_spread_factor: float = Field(default=26.0, ge=0.0)
        maker_liquidity_edge_bps: float = Field(default=12.0, ge=0.0)
        maker_slippage_bps: int = Field(default=42, ge=0)
        lp_slippage_bps: int = Field(default=65, ge=0)
        taker_slippage_bps: int = Field(default=28, ge=0)
        default_slippage_bps: int = Field(default=32, ge=0)
        exit_min_hold_seconds: int = Field(default=300, ge=0)
        exit_time_stop_seconds: int = Field(default=2_700, ge=0)
        exit_stale_profit_pct: float = Field(default=0.01, ge=0.0)
        exit_reentry_cooldown_seconds: int = Field(default=600, ge=0)
        aggressive_min_fee_yield: float = Field(default=0.0002, ge=0.0)
        launch: LaunchSniperConfig = Field(default_factory=LaunchSniperConfig)
        # Optional fixed allocation override (e.g., for deterministic dry-runs)
        fixed_allocation_usd: Optional[float] = Field(default=None, ge=0.0)


    class WalletConfig(BaseModel):
        """Wallet and signer configuration."""

        private_key: Optional[str] = None
        keypair_path: Optional[Path] = None
        public_key: Optional[str] = None
        allow_test_keypair: bool = False


    class StorageConfig(BaseModel):
        """State persistence configuration."""

        database_path: Path = Field(default=Path("./state.sqlite3"))
        cache_ttl_seconds: int = Field(default=600, ge=0)
        metrics_snapshot_path: Path = Field(default=Path("./metrics_snapshots"))


    class MonitoringConfig(BaseModel):
        """Logging and alerting configuration."""

        log_level: str = Field(default="INFO")
        slack_webhook_url: Optional[AnyHttpUrl] = None
        sentry_dsn: Optional[AnyHttpUrl] = None
        sentry_environment: Optional[str] = None
        enable_prometheus: bool = True
        metrics_port: int = Field(default=9100, ge=1, le=65535)
        webhook_urls: List[AnyHttpUrl] = Field(default_factory=list)
        alert_throttle_seconds: int = Field(default=60, ge=0)
        risk_disclaimer: str = Field(
            default=(
                "Trading digital assets involves significant risk. Historical performance "
                "is not indicative of future results and no profits are guaranteed."
            )
        )


    class AppConfig(BaseSettings):
        """Aggregated application configuration."""

        mode: ModeConfig = Field(default_factory=ModeConfig)
        trading: TradingConfig = Field(default_factory=TradingConfig)
        rpc: RPCConfig = Field(default_factory=RPCConfig)
        data_sources: DataSourceConfig = Field(default_factory=DataSourceConfig)
        venues: VenueConfig = Field(default_factory=VenueConfig)
        router: RouterConfig = Field(default_factory=RouterConfig)
        pnl: PnLConfig = Field(default_factory=PnLConfig)
        risk: RiskLimitsConfig = Field(default_factory=RiskLimitsConfig)
        circuit_breakers: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
        event_bus: EventBusConfig = Field(default_factory=EventBusConfig)
        dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
        token_universe: TokenUniverseConfig = Field(default_factory=TokenUniverseConfig)
        strategy: StrategyConfig = Field(default_factory=StrategyConfig)
        wallet: WalletConfig = Field(default_factory=WalletConfig)
        storage: StorageConfig = Field(default_factory=StorageConfig)
        monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
        execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

        model_config = SettingsConfigDict(
            env_file=".env",
            env_nested_delimiter="__",
            case_sensitive=False,
            extra="ignore",
        )

        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls,
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        ):
            def file_settings(_: Optional[BaseSettings] = None) -> Dict[str, Any]:
                payload, _ = _load_toml_config()
                return payload

            def filtered_env_settings(_: Optional[BaseSettings] = None) -> Dict[str, Any]:
                data = env_settings()
                if isinstance(data, dict):
                    value = data.get("rpc") or data.get("RPC")
                    if isinstance(value, (str, bytes)):
                        data.pop("rpc", None)
                        data.pop("RPC", None)
                return data

            # Ensure runtime environment variables win over static config file defaults.
            return (
                init_settings,
                filtered_env_settings,
                dotenv_settings,
                file_settings,
                file_secret_settings,
            )

        @model_validator(mode="after")
        def _sync_mode_defaults(self) -> "AppConfig":
            if self.mode.active == AppMode.TESTNET:
                self.mode.cluster = "testnet"
            helius_url = os.getenv("HELIUS_RPC_URL")
            if not helius_url:
                helius_key = os.getenv("HELIUS_API_KEY")
                if helius_key:
                    helius_url = f"https://mainnet.helius-rpc.com/?api-key={helius_key.strip()}"
            if helius_url:
                previous_primary = str(self.rpc.primary_url)
                self.rpc.primary_url = helius_url
                fallback_candidates = [str(url) for url in self.rpc.fallback_urls]
                if previous_primary not in fallback_candidates:
                    self.rpc.fallback_urls = [
                        previous_primary,
                        *[str(url) for url in self.rpc.fallback_urls],
                    ]
                seen: set[str] = set()
                deduped: list[str] = []
                for url in self.rpc.fallback_urls:
                    url_str = str(url)
                    if url_str in seen:
                        continue
                    seen.add(url_str)
                    deduped.append(url_str)
                self.rpc.fallback_urls = deduped
            return self


else:

    @dataclass(slots=True)
    class RPCConfig:
        primary_url: str = "https://api.mainnet-beta.solana.com"
        fallback_urls: List[str] = field(
            default_factory=lambda: ["https://solana-api.projectserum.com"]
        )
        request_timeout: float = 12.0
        commitment: str = "confirmed"
        request_concurrency: int = 4


    @dataclass(slots=True)
    class DataSourceConfig:
        enable_axiom: bool = True
        enable_pumpfun: bool = True
        enable_solana_token_list: bool = False
        enable_meteora_registry: bool = True
        enable_rocketscan: bool = True
        axiom_base_url: str = "https://api.axiom.xyz"
        axiom_api_key: Optional[str] = None
        pumpfun_base_url: str = "https://pump.fun/api"
        pumpfun_api_key: Optional[str] = None
        damm_base_url: str = "https://dammv2-api.meteora.ag"
        damm_devnet_base_url: str = "https://dammv2-api.devnet.meteora.ag"
        damm_pool_endpoint: str = "/pools"
        damm_page_limit: int = 50
        dlmm_base_url: str = "https://dlmm-api.meteora.ag"
        dlmm_devnet_base_url: str = "https://devnet-dlmm-api.meteora.ag"
        dlmm_pool_endpoint: str = "/pair/all"
        dlmm_page_limit: int = 50
        meteora_registry_url: str = "https://dlmm-api.meteora.ag/api/v1/pairs"
        price_oracle_url: str = "https://lite-api.jup.ag/"
        price_oracle_urls: Optional[List[str]] = None
        fallback_price_usd: Optional[float] = None
        price_feed_overrides: Dict[str, str] = field(default_factory=dict)
        solana_token_list_url: str = "https://token-list-api.solana.com/metadata.json"
        price_oracle_use_pyth_fallback: bool = True
        price_oracle_jupiter_rate_limit_per_minute: int = 55
        rocketscan_base_url: str = "https://rocketscan.fun/api"
        rocketscan_timeout: float = 8.0
        rocketscan_max_results: int = 75
        rocketscan_cache_ttl_seconds: int = 180
        rocketscan_max_workers: int = 8
        rocketscan_max_age_minutes: int = 60
        http_timeout: float = 10.0
        min_liquidity_usd: float = 5_000.0
        cache_ttl_seconds: int = 120


    @dataclass(slots=True)
    class ModeConfig:
        active: AppMode = AppMode.DRY_RUN
        cluster: str = "mainnet-beta"
        dry_run_seed: int = 0
        persist_dry_run_state: bool = True
        config_file: Optional[Path] = None


    @dataclass(slots=True)
    class DammVenueConfig:
        enabled: bool = True
        program_id: str = "Eo7WjKq67rjJQSZxS6z3YkapzY3eMj6Xy8X5EQVn5UaB"
        min_liquidity_usd: float = 20_000.0
        min_depth_levels: int = 3
        max_slippage_bps: int = 80
        health_check_interval_seconds: int = 30
        cache_ttl_seconds: int = 45
        default_quote_mint: str = "EPjFWdd5AufqSSqeM2qN1xzybapC8WdGCr3vZ9V4Wrh"
        quote_mint_allowlist: List[str] = field(
            default_factory=lambda: [
                "EPjFWdd5AufqSSqeM2qN1xzybapC8WdGCr3vZ9V4Wrh",
                "Es9vMFrzaCER1ZXS9dZxXn6vufGAaHo9dZYkTf5ZNVY",
            ]
        )


    @dataclass(slots=True)
    class DlmmVenueConfig:
        enabled: bool = True
        program_id: str = "LBUZKhRxPF3XUpBCjp4YzTKgLccjZhTSDM9YuVaPwxo"
        min_liquidity_usd: float = 25_000.0
        min_active_bins: int = 6
        max_slippage_bps: int = 90
        health_check_interval_seconds: int = 30
        cache_ttl_seconds: int = 30
        required_oracle_confidence_bps: int = 50
        default_quote_mint: str = "EPjFWdd5AufqSSqeM2qN1xzybapC8WdGCr3vZ9V4Wrh"


    @dataclass(slots=True)
    class VenueConfig:
        damm: DammVenueConfig = field(default_factory=DammVenueConfig)
        dlmm: DlmmVenueConfig = field(default_factory=DlmmVenueConfig)


    @dataclass(slots=True)
    class RouterConfig:
        liquidity_weight: float = 0.45
        fee_weight: float = 0.2
        volatility_weight: float = 0.15
        depth_weight: float = 0.1
        rebate_weight: float = 0.1
        fill_weight: float = 0.08
        require_health_checks: bool = True
        prefer_maker_routes: bool = True
        slippage_safety_bps: int = 40


    @dataclass(slots=True)
    class ExecutionConfig:
        queue_capacity: int = 128
        max_concurrency: int = 4
        retry_backoff_seconds: float = 0.75
        max_retry_attempts: int = 3
        jitter_seconds: float = 0.25
        batch_size: int = 4
        dedupe_requests: bool = True
        cache_ttl_seconds: int = 300
        priority_fee_policy: str = "adaptive_percentile"
        priority_fee_percentile: float = 0.75
        max_priority_fee_budget_usd: float = 0.05


    @dataclass(slots=True)
    class TradingConfig:
        enable_live: bool = False
        dry_run: bool = True
        min_edge_bps: int = 60

    @dataclass(slots=True)
    class PnLConfig:
        mark_price_source: MarkPriceSource = MarkPriceSource.ORACLE
        oracle_staleness_seconds: int = 45
        twap_window_seconds: int = 300
        vwap_window_seconds: int = 900
        inventory_valuation_interval_seconds: int = 15
        persist_frequency_seconds: int = 30
        enable_dry_run_pnl: bool = True
        pnl_update_interval_seconds: int = 30
        position_level_pnl: bool = True
        profitability_metrics: bool = True
        performance_summary_every_n_trades: int = 10


    @dataclass(slots=True)
    class RiskLimitsConfig:
        max_global_notional_usd: float = 150_000.0
        max_market_notional_usd: float = 25_000.0
        max_position_notional_usd: float = 10_000.0
        max_inventory_pct: float = 0.3
        max_open_orders: int = 25
        daily_loss_limit_usd: float = 7_500.0
        max_slippage_bps: int = 90
        max_position_usd: float = 250.0
        max_notional_usd: float = 1000.0
        max_daily_loss_pct: float = 2.0
        max_token_exposure_pct: float = 25.0
        emergency_drawdown_pct: float = 25.0
        cool_off_seconds: int = 600


    @dataclass(slots=True)
    class CircuitBreakerConfig:
        max_drawdown_pct: float = 0.2
        max_reject_rate: float = 0.15
        max_volatility_pct: float = 0.5
        health_check_backoff_seconds: int = 180
        cooldown_seconds: int = 300


    @dataclass(slots=True)
    class EventBusConfig:
        enabled: bool = True
        max_queue_size: int = 2_048
        publish_metrics: bool = True
        replay_on_startup: bool = True


    @dataclass(slots=True)
    class DashboardConfig:
        host: str = "0.0.0.0"
        port: int = 8080
        auth_mode: DashboardAuthMode = DashboardAuthMode.READ_ONLY
        allowed_origins: List[str] = field(default_factory=lambda: ["*"])
        metrics_refresh_interval_seconds: int = 5
        enable_incident_view: bool = True
        read_only_token: Optional[str] = None


    @dataclass(slots=True)
    class TokenUniverseConfig:
        min_liquidity_usd: float = 25_000.0
        min_volume_24h_usd: float = 75_000.0
        min_holder_count: int = 250
        max_holder_concentration_pct: float = 0.35
        max_top10_holder_pct: float = 0.7
        min_social_links: int = 1
        max_token_age_minutes: int = 720
        require_oracle_price: bool = True
        allow_unverified_token_list: bool = False
        deny_freeze_authority: bool = True
        deny_mints: List[str] = field(default_factory=list)
        deny_creators: List[str] = field(default_factory=list)
        deny_authorities: List[str] = field(default_factory=list)
        suspicious_keywords: List[str] = field(
            default_factory=lambda: ["rug", "honeypot", "scam", "exploit"]
        )
        min_decimals: int = 0
        max_decimals: int = 12
        aml_min_token_age_minutes: int = 15
        aml_min_holder_count: int = 25
        aml_max_single_holder_pct: float = 0.75
        aml_sanctioned_accounts: List[str] = field(default_factory=list)
        autopause_liquidity_buffer: float = 0.2
        autopause_volume_buffer: float = 0.25
        autopause_recovery_minutes: int = 60
        tracked_quote_mints: List[str] = field(
            default_factory=lambda: [
                "EPjFWdd5AufqSSqeM2qN1xzybapC8WdGCr3vZ9V4Wrh",
                "Es9vMFrzaCER1ZXS9dZxXn6vufGAaHo9dZYkTf5ZNVY",
            ]
        )
        allow_manual_override: bool = True
        discovery_batch_size: int = 200
        early_min_liquidity_usd: float = 5000.0
        early_min_volume_24h_usd: float = 10000.0
        min_base_fee_bps: int = 200
        max_candidates: int = 300


    @dataclass(slots=True)
    class LaunchAllocationTier:
        max_liquidity_usd: float = 0.0
        allocation_sol: float = 0.0


    @dataclass(slots=True)
    class LaunchSniperConfig:
        enabled: bool = True
        max_age_minutes: int = 30
        min_fee_bps: int = 800
        min_fee_yield: float = 0.0005
        min_liquidity_usd: float = 6_000.0
        early_min_liquidity_usd: float = 2_000.0
        max_liquidity_usd: float = 450_000.0
        min_top_holder_pct: float = 0.1
        max_top_holder_pct: float = 0.35
        min_top10_holder_pct: float = 0.1
        max_top10_holder_pct: float = 0.3
        max_market_cap_usd: float = 200_000.0
        early_age_minutes: int = 12
        early_max_market_cap_usd: float = 400_000.0
        min_volume_24h_usd: float = 10_000.0
        early_min_volume_24h_usd: float = 0.0
        min_base_fee_bps: int = 500
        min_current_fee_bps: int = 1_000
        high_fee_boost_bps: int = 800
        high_fee_allocation_multiplier: float = 1.2
        require_fee_scheduler: bool = True
        allowed_scheduler_modes: List[str] = field(
            default_factory=lambda: ["linear", "exponential"]
        )
        bonk_keywords: List[str] = field(default_factory=lambda: ["BONK"])
        bonk_launchpads: List[str] = field(default_factory=lambda: ["bonk", "bonkpad"])
        bonk_allocation_sol: float = 0.05
        allocation_tiers: List["LaunchAllocationTier"] = field(
            default_factory=lambda: [
                LaunchAllocationTier(max_liquidity_usd=75_000.0, allocation_sol=0.05),
                LaunchAllocationTier(max_liquidity_usd=150_000.0, allocation_sol=0.1),
                LaunchAllocationTier(max_liquidity_usd=300_000.0, allocation_sol=0.2),
            ]
        )
        max_dev_holding_pct: float = 4.0
        max_sniper_holding_pct: float = 15.0
        max_insider_holding_pct: float = 15.0
        max_bundler_holding_pct: float = 20.0
        allow_missing_holder_segments: bool = True
        holder_segment_dampen_ratio: float = 0.75
        max_initial_fee_bps: Optional[int] = None
        hill_phase_minutes: int = 25
        hill_min_fee_bps: int = 2_000
        cook_min_fee_bps: int = 1_200
        cook_extension_minutes: int = 10
        max_hold_minutes: int = 45
        exit_cooldown_seconds: int = 900
        max_decisions: int = 3
        momentum_velocity_threshold: float = 0.4
        momentum_min_fee_yield: float = 0.0007
        scale_up_threshold: float = 0.7
        exit_fee_yield_floor: float = 0.0003
        exit_market_cap_floor: float = 50_000.0
        exit_strength_gain_pct: float = 0.45
        exit_trailing_drawdown_pct: float = 0.18
        dlmm_exit_tvl_threshold: float = 20_000.0
        fallback_sol_price: float = 150.0
        require_existing_pool: bool = True
        prefer_damm: bool = True
        min_volume_1h_usd: float = 20000.0
        max_slippage_bps: int = 40
        max_price_impact_bps: int = 30
        take_profit_bps: int = 150
        trailing_stop_bps: int = 75
        stop_loss_bps: int = 80
        dynamic_position_sizing: bool = True


    @dataclass(slots=True)
    class StrategyConfig:
        max_positions: int = 10
        max_candidates: int = 300
        allocation_per_position: float = 500.0
        max_allocation_per_position: float = 2_000.0
        max_allocation_multiplier: float = 2.5
        portfolio_size: float = 5_000.0
        rebalance_interval_seconds: int = 1_800
        stop_loss_pct: float = 0.15
        take_profit_pct: float = 0.4
        min_liquidity_usd: float = 6_000.0
        min_volume_24h_usd: float = 20_000.0
        min_fee_apr: float = 0.08
        min_holder_count: int = 100
        max_single_holder_pct: float = 0.35
        max_top10_holder_pct: float = 0.75
        min_dev_trust_score: float = 0.3
        min_social_score: float = 0.1
        max_token_age_minutes: int = 240
        liquidity_target_usd: float = 25_000.0
        volume_target_usd: float = 100_000.0
        fee_apr_target: float = 0.5
        priority_fee_lamports: int = 5_000
        liquidity_velocity_target: float = 1.35
        momentum_threshold: float = 0.25
        momentum_liquidity_weight: float = 0.5
        momentum_fee_weight: float = 0.3
        momentum_velocity_weight: float = 0.2
        momentum_score_weight: float = 0.12
        momentum_score_cap: float = 1.5
        concentration_penalty_weight: float = 0.3
        momentum_priority_weight: float = 0.1
        maker_min_spread_bps: float = 24.0
        maker_max_spread_bps: float = 135.0
        maker_volatility_spread_factor: float = 40.0
        maker_inventory_spread_factor: float = 26.0
        maker_liquidity_edge_bps: float = 12.0
        maker_slippage_bps: int = 42
        lp_slippage_bps: int = 65
        taker_slippage_bps: int = 28
        default_slippage_bps: int = 32
        volatility_target: float = 1.0
        dynamic_fee_target_bps: float = 20.0
        alpha_score_target: float = 1.0
        price_impact_tolerance_bps: float = 60.0
        exit_min_hold_seconds: int = 300
        exit_time_stop_seconds: int = 2_700
        exit_stale_profit_pct: float = 0.01
        exit_reentry_cooldown_seconds: int = 600
        aggressive_min_fee_yield: float = 0.0002
        launch: LaunchSniperConfig = field(default_factory=LaunchSniperConfig)
        fixed_allocation_usd: Optional[float] = None


    @dataclass(slots=True)
    class WalletConfig:
        private_key: Optional[str] = None
        keypair_path: Optional[Path] = None
        public_key: Optional[str] = None
        allow_test_keypair: bool = False


    @dataclass(slots=True)
    class StorageConfig:
        database_path: Path = Path("./state.sqlite3")
        cache_ttl_seconds: int = 600
        metrics_snapshot_path: Path = Path("./metrics_snapshots")


    @dataclass(slots=True)
    class MonitoringConfig:
        log_level: str = "INFO"
        slack_webhook_url: Optional[str] = None
        sentry_dsn: Optional[str] = None
        sentry_environment: Optional[str] = None
        enable_prometheus: bool = True
        metrics_port: int = 9100
        webhook_urls: List[str] = field(default_factory=list)
        alert_throttle_seconds: int = 60
        risk_disclaimer: str = (
            "Trading digital assets involves significant risk. Historical performance "
            "is not indicative of future results and no profits are guaranteed."
        )


    @dataclass(slots=True)
    class AppConfig:
        mode: ModeConfig = field(default_factory=ModeConfig)
        trading: TradingConfig = field(default_factory=TradingConfig)
        rpc: RPCConfig = field(default_factory=RPCConfig)
        data_sources: DataSourceConfig = field(default_factory=DataSourceConfig)
        venues: VenueConfig = field(default_factory=VenueConfig)
        router: RouterConfig = field(default_factory=RouterConfig)
        pnl: PnLConfig = field(default_factory=PnLConfig)
        risk: RiskLimitsConfig = field(default_factory=RiskLimitsConfig)
        circuit_breakers: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
        event_bus: EventBusConfig = field(default_factory=EventBusConfig)
        dashboard: DashboardConfig = field(default_factory=DashboardConfig)
        token_universe: TokenUniverseConfig = field(default_factory=TokenUniverseConfig)
        strategy: StrategyConfig = field(default_factory=StrategyConfig)
        wallet: WalletConfig = field(default_factory=WalletConfig)
        storage: StorageConfig = field(default_factory=StorageConfig)
        monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
        execution: ExecutionConfig = field(default_factory=ExecutionConfig)

        def __post_init__(self) -> None:
            payload, path = _load_toml_config()
            if payload:
                _apply_dict_to_dataclass(self, payload)
            if path is not None:
                self.mode.config_file = path
            _apply_env_to_dataclass(self)
            if self.mode.active == AppMode.TESTNET:
                self.mode.cluster = "testnet"


def env_path() -> Path:
    """Return the default path for the `.env` file."""

    return Path.cwd() / ".env"


@lru_cache(maxsize=1)
def get_app_config() -> AppConfig:
    """Create a cached application configuration object."""

    return AppConfig()


__all__ = [
    "AppConfig",
    "AppMode",
    "CircuitBreakerConfig",
    "DashboardAuthMode",
    "DashboardConfig",
    "DataSourceConfig",
    "DammVenueConfig",
    "DlmmVenueConfig",
    "EventBusConfig",
    "MarkPriceSource",
    "ModeConfig",
    "MonitoringConfig",
    "PnLConfig",
    "ExecutionConfig",
    "RPCConfig",
    "RiskLimitsConfig",
    "RouterConfig",
    "StorageConfig",
    "StrategyConfig",
    "TokenUniverseConfig",
    "VenueConfig",
    "WalletConfig",
    "env_path",
    "get_app_config",
]
