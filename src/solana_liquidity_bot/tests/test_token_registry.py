from __future__ import annotations

from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from solana_liquidity_bot.config import settings
from solana_liquidity_bot.datalake.schemas import (
    DammPoolSnapshot,
    TokenMetadata,
    TokenOnChainStats,
    RocketscanMetrics,
)
from solana_liquidity_bot.datalake.storage import SQLiteStorage
from solana_liquidity_bot.ingestion.token_registry import TokenRegistryAggregator
from solana_liquidity_bot.ingestion.solana_token_list import TokenListEntry


class FakeAxiomClient:
    def __init__(self, tokens: list[TokenMetadata]) -> None:
        self._tokens = tokens

    def list_recent_tokens(self, limit: int = 20) -> list[TokenMetadata]:
        return self._tokens[:limit]


class FakePumpFunClient:
    def __init__(self, tokens: list[TokenMetadata] | None = None) -> None:
        self._tokens = tokens or []

    def list_new_tokens(self, limit: int = 20) -> list[TokenMetadata]:
        return self._tokens[:limit]


class FakeTokenListClient:
    def __init__(self, entries: dict[str, TokenListEntry]) -> None:
        self._entries = entries

    def lookup(self, mints) -> dict[str, TokenListEntry]:
        result: dict[str, TokenListEntry] = {}
        for mint in mints:
            entry = self._entries.get(mint)
            if entry:
                result[mint] = entry
        return result


class FakeDammClient:
    def __init__(self, pools: dict[str, list[DammPoolSnapshot]]) -> None:
        self._pools = pools

    def fetch_pools_for_tokens(self, tokens) -> dict[str, list[DammPoolSnapshot]]:
        return {token.mint_address: self._pools.get(token.mint_address, []) for token in tokens if self._pools.get(token.mint_address)}

    def fetch_pools_for_mint(self, mint: str) -> list[DammPoolSnapshot]:
        return self._pools.get(mint, [])

    def list_recent_pools(self, limit: int) -> list[DammPoolSnapshot]:
        """Return recent pools for testing."""
        all_pools = []
        for pools_list in self._pools.values():
            all_pools.extend(pools_list)
        return all_pools[:limit]


class FakeDlmmClient:
    def fetch_pools_for_tokens(self, tokens) -> dict[str, list]:  # pragma: no cover - unused in test
        return {}


class FakeRocketscanClient:
    def __init__(self, metrics: dict[str, RocketscanMetrics]) -> None:
        self._metrics = metrics
        self.calls: list[str] = []

    def fetch_metrics_bulk(self, mints, max_workers: int | None = None):
        mints_list = list(mints)
        self.calls.extend(mints_list)
        return {mint: self._metrics.get(mint) for mint in mints_list}


class FakePriceOracle:
    def __init__(self, prices: dict[str, float]) -> None:
        self._prices = prices

    def get_prices(self, mints) -> dict[str, float]:
        mint_list = list(mints)
        print(f"DEBUG: get_prices called with {mint_list}")
        result = {mint: self._prices.get(mint) for mint in mint_list}
        print(f"DEBUG: get_prices returning {result}")
        return result

    def get_price(self, mint: str) -> float | None:
        return self._prices.get(mint)


class FakeOnchainAnalyzer:
    def __init__(self, stats: dict[str, TokenOnChainStats]) -> None:
        self._stats = stats

    def collect_stats(self, tokens) -> list[TokenOnChainStats]:
        return [self._stats[token.mint_address] for token in tokens if token.mint_address in self._stats]


def _token_list_entry(
    mint: str, symbol: str, decimals: int, verified: bool = True
) -> TokenListEntry:
    return TokenListEntry(
        mint=mint,
        name=symbol,
        symbol=symbol,
        decimals=decimals,
        tags=[],
        verified=verified,
        extensions={},
    )


def test_token_registry_assigns_controls(tmp_path: Path) -> None:
    storage = SQLiteStorage(tmp_path / "state.sqlite3")

    token_a = TokenMetadata(mint_address="MintA", symbol="AAA", name="Token A")
    token_b = TokenMetadata(mint_address="MintB", symbol="BBB", name="Token B")

    app_config = settings.AppConfig(
        storage=settings.StorageConfig(database_path=tmp_path / "state.sqlite3"),
        data_sources=settings.DataSourceConfig(
            enable_axiom=True,
            enable_pumpfun=False,
            enable_solana_token_list=True,
            enable_meteora_registry=True,
            enable_rocketscan=False,
            axiom_base_url="https://example.com",
            pumpfun_base_url="https://example.com",
            damm_base_url="https://example.com",
            damm_pool_endpoint="/pools",
            dlmm_base_url="https://example.com",
            dlmm_pool_endpoint="/pools",
            meteora_registry_url="https://example.com",
            price_oracle_url="https://example.com",
            solana_token_list_url="https://example.com",
            http_timeout=5,
        ),
        token_universe=settings.TokenUniverseConfig(
            min_liquidity_usd=5_000.0,
            min_volume_24h_usd=5_000.0,
            min_holder_count=10,
            max_holder_concentration_pct=0.5,
            max_top10_holder_pct=0.8,
            require_oracle_price=True,
            allow_unverified_token_list=True,
            deny_freeze_authority=True,
            autopause_liquidity_buffer=0.2,
            autopause_volume_buffer=0.2,
            min_social_links=0,
        ),
    )

    stats_map = {
        "MintA": TokenOnChainStats(
            token=token_a,
            total_supply=1_000_000,
            decimals=6,
            holder_count=500,
            top_holder_pct=0.1,
            top10_holder_pct=0.3,
            liquidity_estimate=20_000,
            minted_at=datetime.now(timezone.utc) - timedelta(minutes=90),
            last_activity_at=datetime.now(timezone.utc),
            mint_authority=None,
            freeze_authority=None,
        ),
        "MintB": TokenOnChainStats(
            token=token_b,
            total_supply=500_000,
            decimals=6,
            holder_count=5,
            top_holder_pct=0.85,
            top10_holder_pct=0.95,
            liquidity_estimate=1_000,
            minted_at=datetime.now(timezone.utc) - timedelta(minutes=120),
            last_activity_at=datetime.now(timezone.utc),
            mint_authority=None,
            freeze_authority="FreezeKey",
        ),
    }

    pools = {
        "MintA": [
            DammPoolSnapshot(
                address="poolA",
                base_token_mint="MintA",
                quote_token_mint="EPjFWdd5AufqSSqeM2qN1xzybapC8WdGCr3vZ9V4Wrh",
                base_token_amount=5_000,
                quote_token_amount=5_000,
                fee_bps=30,
                tvl_usd=100_000.0,
                volume_24h_usd=60_000.0,
                price_usd=1.0,
                is_active=True,
            )
        ]
    }

    aggregator = TokenRegistryAggregator(
        storage=storage,
        app_config=app_config,
        axiom_client=FakeAxiomClient([token_a, token_b]),
        pumpfun_client=FakePumpFunClient(),
        token_list_client=FakeTokenListClient(
            {
                "MintA": _token_list_entry("MintA", "AAA", 6),
                "MintB": _token_list_entry("MintB", "BBB", 6),
            }
        ),
        damm_client=FakeDammClient(pools),
        dlmm_client=FakeDlmmClient(),
        price_oracle=FakePriceOracle({"MintA": 1.0}),
        onchain_analyzer=FakeOnchainAnalyzer(stats_map),
        rocketscan_client=FakeRocketscanClient({}),
    )

    entries = aggregator.build_universe(limit=5)

    statuses = {entry.token.mint_address: entry.control.status for entry in entries if entry.control}
    assert statuses["MintA"] == "allow"  # Meets liquidity/volume thresholds and should remain active
    assert statuses["MintB"] == "deny"

    stored_tokens = {token.mint_address for token in storage.list_tokens()}
    assert {"MintA", "MintB"}.issubset(stored_tokens)

    risk_a = storage.get_token_risk_metrics("MintA")
    assert risk_a is not None
    assert pytest.approx(risk_a.liquidity_usd, rel=1e-6) == 10_000.0

    control_b = storage.get_token_control("MintB")
    assert control_b is not None and control_b.status == "deny"
    assert "freeze" in control_b.reason.lower()


def test_collect_candidates_includes_new_tokens_when_storage_full(tmp_path: Path) -> None:
    storage = SQLiteStorage(tmp_path / "state.sqlite3")

    existing_tokens = [
        TokenMetadata(mint_address=f"Existing{i}", symbol=f"E{i}", name=f"Existing {i}")
        for i in range(5)
    ]
    for token in existing_tokens:
        storage.upsert_token(token)

    new_tokens = [
        TokenMetadata(mint_address="NewMintA", symbol="NEWA", name="New Token A"),
        TokenMetadata(mint_address="NewMintB", symbol="NEWB", name="New Token B"),
    ]

    app_config = settings.AppConfig(
        storage=settings.StorageConfig(database_path=tmp_path / "state.sqlite3"),
        data_sources=settings.DataSourceConfig(
            enable_axiom=True,
            enable_pumpfun=True,
            enable_solana_token_list=False,
            enable_meteora_registry=False,
            enable_rocketscan=False,
            axiom_base_url="https://example.com",
            pumpfun_base_url="https://example.com",
            damm_base_url="https://example.com",
            damm_pool_endpoint="/pools",
            dlmm_base_url="https://example.com",
            dlmm_pool_endpoint="/pools",
            meteora_registry_url="https://example.com",
            price_oracle_url="https://example.com",
            solana_token_list_url="https://example.com",
            http_timeout=5,
        ),
        token_universe=settings.TokenUniverseConfig(
            min_liquidity_usd=1_000.0,
            min_volume_24h_usd=1_000.0,
            min_holder_count=1,
        ),
    )

    aggregator = TokenRegistryAggregator(
        storage=storage,
        app_config=app_config,
        axiom_client=FakeAxiomClient(new_tokens),
        pumpfun_client=FakePumpFunClient(),
        token_list_client=FakeTokenListClient({}),
        damm_client=FakeDammClient({}),
        dlmm_client=FakeDlmmClient(),
        price_oracle=FakePriceOracle({}),
        onchain_analyzer=FakeOnchainAnalyzer({
            "NewMintA": TokenOnChainStats(
                token=TokenMetadata(mint_address="NewMintA", symbol="NEWA", name="New Token A"),
                total_supply=1_000_000,
                decimals=6,
                holder_count=100,
                top_holder_pct=0.1,
                top10_holder_pct=0.3,
                liquidity_estimate=20_000,
                minted_at=datetime.now(timezone.utc) - timedelta(minutes=10),
                last_activity_at=datetime.now(timezone.utc),
                mint_authority=None,
                freeze_authority=None,
            ),
            "NewMintB": TokenOnChainStats(
                token=TokenMetadata(mint_address="NewMintB", symbol="NEWB", name="New Token B"),
                total_supply=1_000_000,
                decimals=6,
                holder_count=100,
                top_holder_pct=0.1,
                top10_holder_pct=0.3,
                liquidity_estimate=20_000,
                minted_at=datetime.now(timezone.utc) - timedelta(minutes=10),
                last_activity_at=datetime.now(timezone.utc),
                mint_authority=None,
                freeze_authority=None,
            ),
        }),
        rocketscan_client=FakeRocketscanClient({}),
    )

    candidates = aggregator._collect_candidates(limit=3)
    assert len(candidates) == 2  # Only 2 tokens provided, not 3
    candidate_mints = {token.mint_address for token in candidates}
    assert {"NewMintA", "NewMintB"}.issubset(candidate_mints)


def test_compliance_rules_trigger_token_controls(tmp_path: Path) -> None:
    storage = SQLiteStorage(tmp_path / "state.sqlite3")

    risky_token = TokenMetadata(
        mint_address="RugMint",
        symbol="RUG",
        name="Rug Pull",
        creator="Scammer",
    )

    app_config = settings.AppConfig(
        storage=settings.StorageConfig(database_path=tmp_path / "state.sqlite3"),
        token_universe=settings.TokenUniverseConfig(
            deny_mints=["RugMint"],
            deny_creators=["scammer"],
            suspicious_keywords=["rug"],
            min_social_links=1,
        ),
        data_sources=settings.DataSourceConfig(
            enable_axiom=True,
            enable_pumpfun=False,
            enable_solana_token_list=False,
            enable_meteora_registry=False,
            enable_rocketscan=False,
            axiom_base_url="https://example.com",
            pumpfun_base_url="https://example.com",
            damm_base_url="https://example.com",
            damm_pool_endpoint="/pools",
            dlmm_base_url="https://example.com",
            dlmm_pool_endpoint="/pools",
            meteora_registry_url="https://example.com",
            price_oracle_url="https://example.com",
            solana_token_list_url="https://example.com",
            http_timeout=5,
        ),
    )

    stats_map = {
        "RugMint": TokenOnChainStats(
            token=risky_token,
            total_supply=1_000_000,
            decimals=6,
            holder_count=5,
            top_holder_pct=0.9,
            top10_holder_pct=0.95,
            liquidity_estimate=15_000,
            minted_at=datetime.now(timezone.utc) - timedelta(minutes=5),
            last_activity_at=datetime.now(timezone.utc),
            mint_authority="Scammer",
            freeze_authority=None,
        )
    }

    aggregator = TokenRegistryAggregator(
        storage=storage,
        app_config=app_config,
        axiom_client=FakeAxiomClient([risky_token]),
        pumpfun_client=FakePumpFunClient(),
        token_list_client=FakeTokenListClient({}),
        damm_client=FakeDammClient({}),
        dlmm_client=FakeDlmmClient(),
        price_oracle=FakePriceOracle({}),
        onchain_analyzer=FakeOnchainAnalyzer(stats_map),
        rocketscan_client=FakeRocketscanClient({}),
    )

    entries = aggregator.build_universe(limit=1)
    assert entries, "Expected one entry"
    decision = entries[0].control
    assert decision is not None
    assert decision.status == "deny"
    assert "keyword 'rug' indicates potential scam" in decision.reason.lower()

    stored_metrics = storage.get_token_risk_metrics("RugMint")
    assert stored_metrics is not None
    assert any(flag.startswith("compliance_") for flag in stored_metrics.risk_flags)
    assert "compliance_keyword" in stored_metrics.risk_flags  # The actual flag name


def test_rocketscan_bulk_enrichment(tmp_path: Path) -> None:
    storage = SQLiteStorage(tmp_path / "state.sqlite3")

    token = TokenMetadata(mint_address="MintBulk", symbol="MBK", name="Mint Bulk")
    stats = TokenOnChainStats(
        token=token,
        total_supply=1_000_000,
        decimals=6,
        holder_count=500,
        top_holder_pct=0.2,
        top10_holder_pct=0.3,
        liquidity_estimate=50_000,
        minted_at=datetime.now(timezone.utc) - timedelta(minutes=10),
        last_activity_at=datetime.now(timezone.utc),
        mint_authority=None,
        freeze_authority=None,
    )
    damm_pools = {
        token.mint_address: [
            DammPoolSnapshot(
                address="pool_bulk",
                base_token_mint=token.mint_address,
                quote_token_mint="So11111111111111111111111111111111111111112",
                base_token_amount=1_000,
                quote_token_amount=5.0,
                fee_bps=300,
                tvl_usd=1_500.0,
                volume_24h_usd=5_000.0,
                price_usd=1.5,
                created_at=datetime.now(timezone.utc),
                is_active=True,
            )
        ]
    }

    rocketscan_payload = {
        token.mint_address: RocketscanMetrics(
            dev_balance_pct=1.5,
            snipers_pct=4.0,
            insiders_pct=3.0,
            bundlers_pct=2.0,
        )
    }

    app_config = settings.AppConfig(
        storage=settings.StorageConfig(database_path=tmp_path / "state.sqlite3"),
        data_sources=settings.DataSourceConfig(
            enable_axiom=True,
            enable_pumpfun=False,
            enable_solana_token_list=False,
            enable_meteora_registry=False,
            enable_rocketscan=True,
            rocketscan_max_workers=4,
            rocketscan_max_age_minutes=30,
            price_oracle_url="https://example.com",
            damm_base_url="https://example.com",
            damm_pool_endpoint="/pools",
            dlmm_base_url="https://example.com",
            dlmm_pool_endpoint="/pools",
        ),
        token_universe=settings.TokenUniverseConfig(),
    )

    rocketscan_client = FakeRocketscanClient(rocketscan_payload)

    aggregator = TokenRegistryAggregator(
        storage=storage,
        app_config=app_config,
        axiom_client=FakeAxiomClient([token]),
        pumpfun_client=FakePumpFunClient(),
        token_list_client=FakeTokenListClient({}),
        damm_client=FakeDammClient(damm_pools),
        dlmm_client=FakeDlmmClient(),
        price_oracle=FakePriceOracle({token.mint_address: 1.5}),
        onchain_analyzer=FakeOnchainAnalyzer({token.mint_address: stats}),
        rocketscan_client=rocketscan_client,
    )

    entries = aggregator.build_universe(limit=1)
    assert entries
    risk = storage.get_token_risk_metrics(token.mint_address)
    assert risk is not None
    assert pytest.approx(risk.dev_holding_pct, rel=1e-6) == 1.5
    assert pytest.approx(risk.sniper_holding_pct, rel=1e-6) == 4.0
    assert pytest.approx(risk.insider_holding_pct, rel=1e-6) == 3.0
    assert pytest.approx(risk.bundler_holding_pct, rel=1e-6) == 2.0
    assert rocketscan_client.calls == [token.mint_address]
