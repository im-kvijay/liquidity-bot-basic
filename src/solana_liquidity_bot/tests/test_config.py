from __future__ import annotations

from pathlib import Path

import pytest

from solana_liquidity_bot.config import settings


def test_app_config_loads_profiles_and_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Temporarily move .env file to prevent interference from environment variables
    env_file = Path(".env")
    env_backup = None
    if env_file.exists():
        env_backup = tmp_path / ".env.backup"
        env_file.rename(env_backup)
    
    try:
        config_path = tmp_path / "app.toml"
        config_path.write_text(
            """
[default.mode]
active = "dry_run"
cluster = "mainnet-beta"

[default.rpc]
primary_url = "https://api.default"
request_timeout = 9.5

[default.token_universe]
min_liquidity_usd = 15000.0

[live.mode]
active = "live"
cluster = "mainnet-beta"

[live.rpc]
primary_url = "https://api.mainnet"

[live.token_universe]
min_liquidity_usd = 30000.0

"""
        )

        monkeypatch.setenv("APP_CONFIG_FILE", str(config_path))
        monkeypatch.setenv("BOT_MODE", "live")
        monkeypatch.setenv("RPC__REQUEST_TIMEOUT", "18")
        monkeypatch.delenv("RPC__PRIMARY_URL", raising=False)
        monkeypatch.delenv("RPC__FALLBACK_URLS", raising=False)
        monkeypatch.delenv("HELIUS_API_KEY", raising=False)
        monkeypatch.delenv("HELIUS_RPC_URL", raising=False)
        monkeypatch.delenv("TOKEN_UNIVERSE__MIN_LIQUIDITY_USD", raising=False)
        monkeypatch.delenv("TOKEN_UNIVERSE__MIN_VOLUME_24H_USD", raising=False)
        monkeypatch.delenv("TOKEN_UNIVERSE__MIN_HOLDER_COUNT", raising=False)
        monkeypatch.delenv("TOKEN_UNIVERSE__MAX_HOLDER_CONCENTRATION_PCT", raising=False)
        monkeypatch.delenv("TOKEN_UNIVERSE__MAX_TOP10_HOLDER_PCT", raising=False)
        monkeypatch.delenv("TOKEN_UNIVERSE__MIN_SOCIAL_LINKS", raising=False)
        monkeypatch.delenv("TOKEN_UNIVERSE__REQUIRE_ORACLE_PRICE", raising=False)
        monkeypatch.delenv("TOKEN_UNIVERSE__ALLOW_UNVERIFIED_TOKEN_LIST", raising=False)
        monkeypatch.delenv("TOKEN_UNIVERSE__DENY_FREEZE_AUTHORITY", raising=False)
        monkeypatch.delenv("TOKEN_UNIVERSE__DENY_MINTS", raising=False)
        monkeypatch.delenv("TOKEN_UNIVERSE__DENY_CREATORS", raising=False)
        monkeypatch.delenv("TOKEN_UNIVERSE__DENY_AUTHORITIES", raising=False)
        monkeypatch.delenv("TOKEN_UNIVERSE__SUSPICIOUS_KEYWORDS", raising=False)
        monkeypatch.delenv("TOKEN_UNIVERSE__AML_SANCTIONED_ACCOUNTS", raising=False)
        monkeypatch.delenv("TOKEN_UNIVERSE__AML_MIN_HOLDER_COUNT", raising=False)
        monkeypatch.delenv("TOKEN_UNIVERSE__AML_MAX_SINGLE_HOLDER_PCT", raising=False)
        monkeypatch.delenv("TOKEN_UNIVERSE__MAX_TOKEN_AGE_MINUTES", raising=False)
        monkeypatch.delenv("TOKEN_UNIVERSE__AUTOPAUSE_LIQUIDITY_BUFFER", raising=False)
        monkeypatch.delenv("TOKEN_UNIVERSE__AUTOPAUSE_VOLUME_BUFFER", raising=False)
        monkeypatch.delenv("TOKEN_UNIVERSE__DISCOVERY_BATCH_SIZE", raising=False)

        settings.get_app_config.cache_clear()
        cfg = settings.get_app_config()

        assert cfg.mode.active == settings.AppMode.LIVE
        assert cfg.mode.config_file == config_path
        primary_url_str = str(cfg.rpc.primary_url)
        assert any(fragment in primary_url_str for fragment in ("api.mainnet", "helius-rpc.com"))
        assert cfg.rpc.request_timeout == 18.0
        assert cfg.token_universe.min_liquidity_usd == 30000.0
        assert cfg.token_universe.allow_unverified_token_list is False

        settings.get_app_config.cache_clear()
    
    finally:
        # Restore .env file if it was moved
        if env_backup and env_backup.exists():
            env_backup.rename(env_file)


def test_strategy_launch_sniper_alias(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "app.toml"
    monkeypatch.setenv("APP_CONFIG_FILE", str(config_path))
    monkeypatch.delenv("BOT_MODE", raising=False)

    config_path.write_text(
        """
[default.strategy.launch_sniper]
min_liquidity_usd = 12345.0
min_base_fee_bps = 650
        """
    )

    settings.get_app_config.cache_clear()
    cfg = settings.get_app_config()

    assert cfg.strategy.launch.min_liquidity_usd == 12345.0
    assert cfg.strategy.launch.min_base_fee_bps == 650

    config_path.write_text(
        """
[default.strategy.launch]
min_liquidity_usd = 4321.0
max_decisions = 7
        """
    )

    settings.get_app_config.cache_clear()
    canonical_cfg = settings.get_app_config()
    assert canonical_cfg.strategy.launch.min_liquidity_usd == 4321.0
    assert canonical_cfg.strategy.launch.max_decisions == 7

    settings.get_app_config.cache_clear()
