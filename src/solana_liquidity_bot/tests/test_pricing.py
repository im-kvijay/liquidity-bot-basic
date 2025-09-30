"""Tests for the price oracle helpers."""

from __future__ import annotations

from solana_liquidity_bot.config import settings
from solana_liquidity_bot.ingestion.pricing import PriceOracle


def test_get_prices_consumes_generators_once(monkeypatch) -> None:
    config = settings.DataSourceConfig(price_oracle_url="https://example.com", http_timeout=1)
    oracle = PriceOracle(config=config)

    requested: list[tuple[str, ...]] = []

    def fake_request(mints):
        batch = tuple(mints)
        requested.append(batch)
        return {mint: float(index) for index, mint in enumerate(batch, start=1)}

    monkeypatch.setattr(oracle, "_request", fake_request)

    generator = (mint for mint in ["MintA", "MintA", "MintB"])
    prices = oracle.get_prices(generator)

    assert prices == {"MintA": 1.0, "MintB": 2.0}
    assert requested == [("MintA", "MintB")]


def test_get_prices_uses_cache(monkeypatch) -> None:
    config = settings.DataSourceConfig(price_oracle_url="https://example.com", http_timeout=1)
    oracle = PriceOracle(config=config)

    calls: list[tuple[str, ...]] = []

    def fake_request(mints):
        batch = tuple(mints)
        calls.append(batch)
        return {mint: 42.0 for mint in batch}

    monkeypatch.setattr(oracle, "_request", fake_request)

    oracle.get_prices((mint for mint in ["MintA", "MintB"]))
    oracle.get_prices(["MintA"])

    assert calls == [("MintA", "MintB")]
