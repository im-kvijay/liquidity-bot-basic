from datetime import datetime, timezone

from solana_liquidity_bot.analytics.pnl import PnLEngine
from solana_liquidity_bot.datalake.schemas import FillEvent, TokenMetadata
from solana_liquidity_bot.datalake.storage import SQLiteStorage


class DummyOracle:
    def get_mark_price(self, mint: str, *, source: str = "oracle", window_seconds: int | None = None, fallback=None):
        return 1.0

    def get_price(self, mint: str) -> float:
        return 1.0


def test_pnl_engine_register_fill(tmp_path):
    storage = SQLiteStorage(tmp_path / "state.sqlite3")
    token = TokenMetadata(mint_address="Mint11111111111111111111111111111111", symbol="TKN", name="Token")
    storage.upsert_token(token)
    pnl_engine = PnLEngine(storage=storage, price_oracle=DummyOracle())

    fill = FillEvent(
        timestamp=datetime.now(timezone.utc),
        mint_address=token.mint_address,
        token_symbol=token.symbol,
        venue="test",
        action="enter",
        side="maker",
        base_quantity=5.0,
        quote_quantity=5.0,
        price_usd=1.0,
        fee_usd=0.05,
        rebate_usd=0.0,
        expected_value=1.0,
        slippage_bps=10.0,
        strategy="spread_mm",
        correlation_id="corr-1",
        signature=None,
        is_dry_run=False,
    )

    pnl_engine.register_fill(fill)
    snapshot = pnl_engine.snapshot({}, persist=False)

    assert snapshot.inventory_value_usd > 0
    exposures = pnl_engine.exposures()
    assert token.mint_address in exposures


def test_pnl_engine_exit_clears_lp_state(tmp_path):
    storage = SQLiteStorage(tmp_path / "state.sqlite3")
    token = TokenMetadata(mint_address="MintExit111111111111111111111111111111", symbol="EXT", name="ExitToken")
    storage.upsert_token(token)
    engine = PnLEngine(storage=storage, price_oracle=DummyOracle())

    entry_fill = FillEvent(
        timestamp=datetime.now(timezone.utc),
        mint_address=token.mint_address,
        token_symbol=token.symbol,
        venue="damm",
        action="enter",
        side="maker",
        base_quantity=2.0,
        quote_quantity=2.0,
        price_usd=1.0,
        fee_usd=0.01,
        rebate_usd=0.0,
        expected_value=1.0,
        slippage_bps=5.0,
        strategy="spread_mm",
        correlation_id="enter-test",
        signature=None,
        is_dry_run=False,
        pool_address="Pool111",
        lp_token_amount=500,
        position_address="Pos111",
        position_secret="secret",
    )
    engine.register_fill(entry_fill)
    positions = storage.list_positions()
    assert positions and positions[0].lp_token_amount == 500

    exit_fill = FillEvent(
        timestamp=datetime.now(timezone.utc),
        mint_address=token.mint_address,
        token_symbol=token.symbol,
        venue="damm",
        action="exit",
        side="maker",
        base_quantity=2.0,
        quote_quantity=2.0,
        price_usd=1.0,
        fee_usd=0.01,
        rebate_usd=0.0,
        expected_value=1.0,
        slippage_bps=5.0,
        strategy="spread_mm",
        correlation_id="exit-test",
        signature=None,
        is_dry_run=False,
        pool_address="Pool111",
        lp_token_amount=500,
        position_address="Pos111",
        position_secret="secret",
    )
    engine.register_fill(exit_fill)
    assert not storage.list_positions()
