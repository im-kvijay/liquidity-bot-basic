import asyncio
import types
from datetime import datetime, timezone

from solders.pubkey import Pubkey
from unittest.mock import patch

from solana_liquidity_bot.analytics.pnl import PnLEngine
from solana_liquidity_bot.config.settings import AppMode
from solana_liquidity_bot.datalake.schemas import (
    StrategyDecision,
    TokenMetadata,
    TokenRiskMetrics,
    TokenUniverseEntry,
)
from solana_liquidity_bot.datalake.storage import SQLiteStorage
from solana_liquidity_bot.execution.transaction_queue import TransactionQueue
from solana_liquidity_bot.execution.venues.base import VenueQuote


class DummyOracle:
    def get_mark_price(self, mint: str, *, source: str = "oracle", window_seconds: int | None = None, fallback=None):
        return 1.0

    def get_price(self, mint: str) -> float:
        return 1.0


class StaticRouter:
    def route(self, decision, entry, mode):
        quote = VenueQuote(
            venue="test",
            pool_address="pool",
            base_mint=decision.token.mint_address,
            quote_mint="Quote",
            allocation_usd=decision.allocation,
            pool_liquidity_usd=10_000.0,
            expected_price=1.0,
            expected_slippage_bps=5.0,
            liquidity_score=1.0,
            depth_score=1.0,
            volatility_penalty=0.0,
            fee_bps=30,
            rebate_bps=0.0,
            expected_fees_usd=decision.allocation * 0.003,
            base_contribution_lamports=0,
            quote_contribution_lamports=0,
            extras={},
        )
        return type("Decision", (), {"adapter": None, "request": None, "quote": quote, "score": 1.0})()


class TWAPTestRouter:
    """Router stub that produces a buildable plan for TWAP slice tests."""

    def __init__(self):
        self.last_owner = None

    class _Adapter:
        def __init__(self, router: "TWAPTestRouter", decision: StrategyDecision):
            self._router = router
            self._decision = decision

        def build_plan(self, request, quote, owner):
            from solana_liquidity_bot.execution.solana_compat import Transaction
            from solana_liquidity_bot.execution.venues.base import TransactionPlan

            self._router.last_owner = owner
            return TransactionPlan(
                decision=self._decision,
                venue=quote.venue,
                quote=quote,
                transaction=Transaction(),
                signers=[],
                correlation_id=self._decision.correlation_id,
                position_address=str(owner),
            )

    def route(self, decision, entry, mode):
        quote = VenueQuote(
            venue="test",
            pool_address="pool",
            base_mint=decision.token.mint_address,
            quote_mint="Quote",
            allocation_usd=decision.allocation,
            pool_liquidity_usd=10_000.0,
            expected_price=1.0,
            expected_slippage_bps=5.0,
            liquidity_score=1.0,
            depth_score=1.0,
            volatility_penalty=0.0,
            fee_bps=30,
            rebate_bps=0.0,
            expected_fees_usd=decision.allocation * 0.003,
            base_contribution_lamports=0,
            quote_contribution_lamports=0,
            extras={},
        )
        adapter = self._Adapter(self, decision)
        request = type("DummyRequest", (), {"decision": decision})()
        return type(
            "Decision",
            (),
            {"adapter": adapter, "request": request, "quote": quote, "score": 1.0},
        )()


class RiskCheckPnLEngine:
    """Minimal PnL engine stub for exercising global risk checks."""

    def __init__(self, daily_delta: float):
        self._daily_start_equity = 1_000.0
        self._daily_delta = daily_delta

    def get_performance_summary(self):
        return {
            "total_equity": self._daily_start_equity + self._daily_delta,
            "max_drawdown_pct": 0.0,
            "position_count": 0,
            "win_rate_pct": 1.0,
            "profit_factor": 1.0,
        }

    def daily_loss(self) -> float:
        return self._daily_delta

    def snapshot(self, persist: bool = False):
        return types.SimpleNamespace(net_exposure_usd=0.0)


def test_transaction_queue_dry_run_records_fill(tmp_path):
    storage = SQLiteStorage(tmp_path / "state.sqlite3")
    token = TokenMetadata(mint_address="Mint11111111111111111111111111111111", symbol="TKN", name="Token")
    storage.upsert_token(token)
    pnl_engine = PnLEngine(storage=storage, price_oracle=DummyOracle())
    queue = TransactionQueue(storage=storage, pnl_engine=pnl_engine, router=StaticRouter())

    decision = StrategyDecision(
        token=token,
        action="enter",
        allocation=0.1,  # Very small allocation to pass risk checks
        strategy="spread_mm",
        side="maker",
        priority=1.0,
    )
    entry = TokenUniverseEntry(
        token=token,
        stats=None,
        damm_pools=[],
        dlmm_pools=[],
        control=None,
        risk=TokenRiskMetrics(
            mint_address=token.mint_address,
            liquidity_usd=20_000.0,
            volume_24h_usd=40_000.0,
            volatility_score=0.4,
            holder_count=500,
            top_holder_pct=0.1,
            top10_holder_pct=0.2,
            has_oracle_price=True,
            price_confidence_bps=10,
            last_updated=datetime.now(timezone.utc),
            risk_flags=[],
        ),
        liquidity_event=None,
    )

    asyncio.run(
        queue.enqueue(
            [decision],
            Pubkey.default(),
            {token.mint_address: entry},
            AppMode.DRY_RUN,
            dry_run=True,
        )
    )
    fills = storage.list_fills()
    assert fills, "dry run should record synthetic fills"
    assert fills[0].is_dry_run


def test_twap_slice_enqueue_uses_executor_context(tmp_path):
    storage = SQLiteStorage(tmp_path / "state.sqlite3")
    token = TokenMetadata(
        mint_address="Mint11111111111111111111111111111111",
        symbol="TKN",
        name="Token",
    )
    storage.upsert_token(token)
    pnl_engine = PnLEngine(storage=storage, price_oracle=DummyOracle())
    router = TWAPTestRouter()
    queue = TransactionQueue(storage=storage, pnl_engine=pnl_engine, router=router)

    decision = StrategyDecision(
        token=token,
        action="enter",
        allocation=0.1,
        strategy="spread_mm",
        side="maker",
        priority=1.0,
        metadata={
            "twap_slices": 2,
            "slice_interval_seconds": 1,
            "slice_allocation": 0.05,
        },
    )
    entry = TokenUniverseEntry(
        token=token,
        stats=None,
        damm_pools=[],
        dlmm_pools=[],
        control=None,
        risk=TokenRiskMetrics(
            mint_address=token.mint_address,
            liquidity_usd=20_000.0,
            volume_24h_usd=40_000.0,
            volatility_score=0.4,
            holder_count=500,
            top_holder_pct=0.1,
            top10_holder_pct=0.2,
            has_oracle_price=True,
            price_confidence_bps=10,
            last_updated=datetime.now(timezone.utc),
            risk_flags=[],
        ),
        liquidity_event=None,
    )

    owner = Pubkey.default()
    universe = {token.mint_address: entry}
    mode = AppMode.LIVE
    queue._twap_executor.set_execution_context(owner, universe, mode)

    slices = queue._twap_executor.add_twap_decision(decision, universe)
    assert slices, "expected TWAP slices to be created"

    first_slice = slices[0]

    async def run_enqueue():
        await queue._enqueue_single_decision(first_slice.decision, first_slice.parent_correlation_id)

    asyncio.run(run_enqueue())

    assert queue._queue.qsize() == 1, "TWAP slice should be enqueued"
    assert first_slice.decision.correlation_id in queue._pending, "pending map should track slice"
    queued = queue._pending[first_slice.decision.correlation_id]
    assert queued.plan.venue == "test"
    assert router.last_owner == owner


def test_twap_executor_background_task_runs(tmp_path):
    async def run_test():
        storage = SQLiteStorage(tmp_path / "state.sqlite3")
        token = TokenMetadata(
            mint_address="Mint11111111111111111111111111111111",
            symbol="TKN",
            name="Token",
        )
        storage.upsert_token(token)
        pnl_engine = PnLEngine(storage=storage, price_oracle=DummyOracle())
        router = TWAPTestRouter()
        queue = TransactionQueue(storage=storage, pnl_engine=pnl_engine, router=router)

        decision = StrategyDecision(
            token=token,
            action="enter",
            allocation=0.1,
            strategy="spread_mm",
            side="maker",
            priority=1.0,
            metadata={
                "twap_slices": 2,
                "slice_interval_seconds": 1,
                "slice_allocation": 0.05,
            },
        )
        entry = TokenUniverseEntry(
            token=token,
            stats=None,
            damm_pools=[],
            dlmm_pools=[],
            control=None,
            risk=TokenRiskMetrics(
                mint_address=token.mint_address,
                liquidity_usd=20_000.0,
                volume_24h_usd=40_000.0,
                volatility_score=0.4,
                holder_count=500,
                top_holder_pct=0.1,
                top10_holder_pct=0.2,
                has_oracle_price=True,
                price_confidence_bps=10,
                last_updated=datetime.now(timezone.utc),
                risk_flags=[],
            ),
            liquidity_event=None,
        )

        owner = Pubkey.default()
        universe = {token.mint_address: entry}
        mode = AppMode.LIVE

        try:
            await queue.enqueue([decision], owner, universe, mode)
            assert queue._twap_executor.is_running()

            await asyncio.sleep(1.1)

            assert queue._queue.qsize() >= 1, "expected TWAP executor to enqueue slices"
            assert queue._pending, "expected pending map to contain TWAP slices"
        finally:
            await queue.close()

    asyncio.run(run_test())


def test_twap_executor_handles_midflight_enqueue(tmp_path):
    async def run_test():
        storage = SQLiteStorage(tmp_path / "state.sqlite3")
        token = TokenMetadata(
            mint_address="Mint11111111111111111111111111111111",
            symbol="TKN",
            name="Token",
        )
        storage.upsert_token(token)
        pnl_engine = PnLEngine(storage=storage, price_oracle=DummyOracle())
        router = TWAPTestRouter()
        queue = TransactionQueue(storage=storage, pnl_engine=pnl_engine, router=router)

        owner = Pubkey.default()
        entry = TokenUniverseEntry(
            token=token,
            stats=None,
            damm_pools=[],
            dlmm_pools=[],
            control=None,
            risk=TokenRiskMetrics(
                mint_address=token.mint_address,
                liquidity_usd=20_000.0,
                volume_24h_usd=40_000.0,
                volatility_score=0.4,
                holder_count=500,
                top_holder_pct=0.1,
                top10_holder_pct=0.2,
                has_oracle_price=True,
                price_confidence_bps=10,
                last_updated=datetime.now(timezone.utc),
                risk_flags=[],
            ),
            liquidity_event=None,
        )

        first_decision = StrategyDecision(
            token=token,
            action="enter",
            allocation=0.1,
            strategy="spread_mm",
            side="maker",
            priority=1.0,
            metadata={
                "twap_slices": 2,
                "slice_interval_seconds": 0,
                "slice_allocation": 0.05,
            },
        )

        second_decision = StrategyDecision(
            token=token,
            action="enter",
            allocation=0.2,
            strategy="spread_mm",
            side="maker",
            priority=1.0,
            metadata={
                "twap_slices": 2,
                "slice_interval_seconds": 0,
                "slice_allocation": 0.1,
            },
        )

        universe = {token.mint_address: entry}
        mode = AppMode.LIVE

        original_enqueue = queue._enqueue_single_decision
        original_should_abort = queue._twap_executor._should_abort_twap

        slice_started = asyncio.Event()
        allow_continue = asyncio.Event()

        async def instrumented_enqueue(self, decision, parent_correlation_id):
            slice_started.set()
            await allow_continue.wait()
            return await original_enqueue(decision, parent_correlation_id)

        queue._enqueue_single_decision = types.MethodType(instrumented_enqueue, queue)

        async def never_abort(self, slice_obj):
            return False

        queue._twap_executor._should_abort_twap = types.MethodType(
            never_abort, queue._twap_executor
        )

        try:
            with patch.object(queue._twap_executor._logger, "error") as mock_error:
                await queue.enqueue([first_decision], owner, universe, mode)

                await asyncio.wait_for(slice_started.wait(), timeout=5)

                await queue.enqueue([second_decision], owner, universe, mode)

                allow_continue.set()

                expected_slices = (
                    first_decision.metadata["twap_slices"]
                    + second_decision.metadata["twap_slices"]
                )

                async def wait_for_completion():
                    for _ in range(50):
                        if not queue._twap_executor._active_twaps:
                            return
                        await asyncio.sleep(0.1)
                    raise AssertionError("TWAP executions did not complete")

                await wait_for_completion()

                assert queue._queue.qsize() == expected_slices

                assert not mock_error.called
        finally:
            allow_continue.set()
            queue._twap_executor._should_abort_twap = original_should_abort
            queue._enqueue_single_decision = original_enqueue
            await queue.close()

    asyncio.run(run_test())


def test_global_risk_allows_profitable_day(tmp_path):
    storage = SQLiteStorage(tmp_path / "risk_state.sqlite3")
    pnl_engine = RiskCheckPnLEngine(daily_delta=250.0)
    queue = TransactionQueue(storage=storage, pnl_engine=pnl_engine, router=StaticRouter())

    try:
        assert queue._check_global_risk_limits()
    finally:
        asyncio.run(queue.close())


def test_global_risk_blocks_excessive_loss(tmp_path):
    storage = SQLiteStorage(tmp_path / "risk_loss_state.sqlite3")
    pnl_engine = RiskCheckPnLEngine(daily_delta=-100.0)
    queue = TransactionQueue(storage=storage, pnl_engine=pnl_engine, router=StaticRouter())

    try:
        assert not queue._check_global_risk_limits()
    finally:
        asyncio.run(queue.close())
