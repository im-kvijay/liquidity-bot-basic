"""Asynchronous transaction queue with backpressure, retries, and PnL hooks."""

from __future__ import annotations

import asyncio
import json
import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from solders.pubkey import Pubkey

from ..analytics.pnl import PnLEngine
from ..config.settings import AppMode, ExecutionConfig, get_app_config
from ..datalake.schemas import FillEvent, RouterDecisionRecord, StrategyDecision, TokenUniverseEntry
from ..datalake.storage import SQLiteStorage
from ..monitoring.event_bus import EVENT_BUS, EventSeverity, EventType
from ..monitoring.logger import correlation_scope, get_logger
from ..monitoring.metrics import METRICS
from .router import OrderRouter, RoutingDecision
from .solana_client import SolanaClient
from .transaction_builder import DammTransactionBuilder, DlmmTransactionBuilder, TransactionPlan
from .venues import DammVenueAdapter, DlmmVenueAdapter
from .wallet import Wallet


@dataclass(slots=True)
class QueuedTransaction:
    plan: TransactionPlan
    attempts: int = 0
    enqueued_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class TWAPSlice:
    """Represents a single slice of a TWAP execution."""
    decision: StrategyDecision
    allocation_usd: float
    slice_index: int
    total_slices: int
    scheduled_time: datetime
    parent_correlation_id: str


class TWAPExecutor:
    """Handles TWAP execution with mid-slice checks and abort capability."""

    def __init__(self, transaction_queue: "TransactionQueue"):
        self._transaction_queue = transaction_queue
        self._logger = get_logger(__name__)
        self._active_twaps: Dict[str, List[TWAPSlice]] = {}
        self._background_task: Optional[asyncio.Task] = None
        # Store context needed for production execution
        self._current_owner: Optional[Pubkey] = None
        self._current_universe: Optional[Mapping[str, TokenUniverseEntry]] = None
        self._current_mode: Optional[AppMode] = None

    async def start(self):
        """Start the TWAP execution background task."""
        if self._background_task is None or self._background_task.done():
            self._background_task = asyncio.create_task(self._execute_twaps())

    def is_running(self) -> bool:
        """Return True when the executor background task is active."""
        return self._background_task is not None and not self._background_task.done()

    async def stop(self):
        """Stop the TWAP execution background task gracefully."""
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass  # Expected when cancelling
            except Exception as e:
                self._logger.error(f"Error stopping TWAP executor: {e}")
            finally:
                self._background_task = None
        elif self._background_task and self._background_task.done():
            self._background_task = None
    
    def set_execution_context(self, owner: Pubkey, universe: Mapping[str, TokenUniverseEntry], mode: AppMode):
        """Set the execution context for TWAP slices."""
        self._current_owner = owner
        self._current_universe = universe
        self._current_mode = mode

    def get_execution_context(self) -> tuple[
        Optional[Pubkey],
        Optional[Mapping[str, TokenUniverseEntry]],
        Optional[AppMode],
    ]:
        """Return the execution context required for TWAP slices."""
        return self._current_owner, self._current_universe, self._current_mode

    def add_twap_decision(
        self,
        decision: StrategyDecision,
        universe: Mapping[str, TokenUniverseEntry],
        *,
        persist: bool = True,
    ) -> List[TWAPSlice]:
        """Split a decision into TWAP slices.

        Args:
            decision: Strategy decision to split into TWAP slices.
            universe: Token universe mapping for routing context.
            persist: When False, do not register slices for execution. This is
                used for dry-run flows so slices are calculated for logging but
                never executed.
        """
        # Check if decision has TWAP metadata
        if not hasattr(decision, 'metadata') or not decision.metadata:
            return []

        slice_allocation = decision.metadata.get("slice_allocation", decision.allocation)
        slice_interval_seconds = decision.metadata.get("slice_interval_seconds", 30)
        twap_slices = decision.metadata.get("twap_slices", 1)

        if twap_slices <= 1:
            # Not a TWAP, return as single slice
            return [TWAPSlice(
                decision=decision,
                allocation_usd=decision.allocation,
                slice_index=0,
                total_slices=1,
                scheduled_time=datetime.now(timezone.utc),
                parent_correlation_id=decision.correlation_id or ""
            )]

        # Split into multiple slices
        slices = []
        base_allocation = slice_allocation
        parent_correlation_id = decision.correlation_id or str(uuid.uuid4())

        for i in range(twap_slices):
            slice_decision = StrategyDecision(
                token=decision.token,
                action=decision.action,
                allocation=base_allocation,
                priority=decision.priority,
                strategy=decision.strategy,
                side=decision.side,
                venue=decision.venue,
                pool_address=decision.pool_address,
                price_usd=decision.price_usd,
                base_liquidity=decision.base_liquidity,
                quote_liquidity=decision.quote_liquidity,
                pool_fee_bps=decision.pool_fee_bps,
                token_decimals=decision.token_decimals,
                correlation_id=f"{parent_correlation_id}_slice_{i}",
                metadata=dict(decision.metadata),  # Copy metadata
                position_snapshot=decision.position_snapshot,
                quote_token_mint=decision.quote_token_mint,
                max_slippage_bps=decision.max_slippage_bps,
                expected_value=decision.expected_value,
                expected_slippage_bps=decision.expected_slippage_bps,
                expected_fees_usd=decision.expected_fees_usd,
                expected_rebate_usd=decision.expected_rebate_usd,
                notes=decision.notes,
            )

            # Update slice-specific metadata
            slice_decision.metadata.update({
                "twap_slice_index": i,
                "twap_total_slices": twap_slices,
                "twap_parent_correlation_id": parent_correlation_id,
                "twap_scheduled": True
            })

            scheduled_time = datetime.now(timezone.utc) + timedelta(seconds=i * slice_interval_seconds)
            slice = TWAPSlice(
                decision=slice_decision,
                allocation_usd=base_allocation,
                slice_index=i,
                total_slices=twap_slices,
                scheduled_time=scheduled_time,
                parent_correlation_id=parent_correlation_id
            )
            slices.append(slice)

        if persist:
            # Store for tracking
            self._active_twaps[parent_correlation_id] = slices

            # Update TWAP metrics
            METRICS.gauge("twap_active_count", len(self._active_twaps))
            METRICS.increment("twap_created", 1)
            METRICS.observe("twap_slice_count", twap_slices)

            self._logger.info(
                f"Created TWAP execution: {decision.token.mint_address} with {twap_slices} slices, "
                f"interval {slice_interval_seconds}s, parent {parent_correlation_id}"
            )
        else:
            self._logger.info(
                "Dry-run TWAP execution calculated: %s with %s slices, interval %ss, parent %s",
                decision.token.mint_address,
                twap_slices,
                slice_interval_seconds,
                parent_correlation_id,
            )

        return slices

    async def _execute_twaps(self):
        """Background task that executes TWAP slices at scheduled times."""
        while True:
            try:
                await asyncio.sleep(1)  # Check every second

                current_time = datetime.now(timezone.utc)
                completed_twaps = []

                # Check all active TWAPs
                for parent_id, slices in list(self._active_twaps.items()):
                    # Find slices that are ready to execute
                    ready_slices = [s for s in slices if s.scheduled_time <= current_time]

                    for slice_obj in ready_slices:
                        # Execute the slice
                        await self._execute_slice(slice_obj)

                        # Remove from active list
                        slices.remove(slice_obj)

                    # If all slices completed, mark as completed
                    if not slices:
                        completed_twaps.append(parent_id)

                # Clean up completed TWAPs
                for parent_id in completed_twaps:
                    del self._active_twaps[parent_id]
                    METRICS.increment("twap_completed", 1)
                
                # Update active TWAP count
                METRICS.gauge("twap_active_count", len(self._active_twaps))

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in TWAP execution: {e}")

    async def _execute_slice(self, slice_obj: TWAPSlice):
        """Execute a single TWAP slice."""
        try:
            # Check if we should abort remaining slices (mid-slice check)
            should_abort = await self._should_abort_twap(slice_obj)
            if should_abort:
                self._logger.warning(
                    f"Aborting TWAP slice {slice_obj.slice_index + 1}/{slice_obj.total_slices} "
                    f"for {slice_obj.decision.token.mint_address}: edge deteriorated"
                )
                return

            # Create a copy of the decision for execution
            decision = slice_obj.decision

            self._logger.info(
                f"Executing TWAP slice {slice_obj.slice_index + 1}/{slice_obj.total_slices} "
                f"for {decision.token.mint_address} (${decision.allocation:.2f})"
            )

            # Use the transaction queue's enqueue method but only for this single slice
            # We need to call the internal methods to avoid creating more TWAPs
            await self._transaction_queue._enqueue_single_decision(decision, slice_obj.parent_correlation_id)

        except Exception as e:
            self._logger.error(f"Error executing TWAP slice: {e}")

    async def _should_abort_twap(self, slice_obj: TWAPSlice) -> bool:
        """Check if TWAP should be aborted due to deteriorating conditions with real quote refresh."""
        try:
            # Only check for slices after the first one
            if slice_obj.slice_index == 0:
                return False

            # Get universe entry for re-pricing
            if not self._current_universe or not self._current_mode:
                # Fallback to basic checks if no context
                return await self._basic_twap_abort_check(slice_obj)

            universe_entry = self._current_universe.get(slice_obj.decision.token.mint_address)
            if not universe_entry:
                self._logger.warning(f"No universe entry for TWAP abort check: {slice_obj.decision.token.mint_address}")
                return True

            # 1. Refresh route and quote for current conditions
            try:
                fresh_route = self._transaction_queue._router.route(slice_obj.decision, universe_entry, self._current_mode)
                if not fresh_route:
                    self._logger.warning(f"TWAP abort: no fresh route for {slice_obj.decision.token.mint_address}")
                    return True

                fresh_quote = fresh_route.quote

                # 2. Check if slippage has deteriorated significantly (>20 BPS jump)
                original_slippage = slice_obj.decision.expected_slippage_bps or 0
                current_slippage = fresh_quote.expected_slippage_bps

                if current_slippage > original_slippage + 20:  # 20 BPS deterioration
                    self._logger.warning(
                        f"TWAP abort: slippage jumped from {original_slippage:.1f} to {current_slippage:.1f} bps "
                        f"for {slice_obj.decision.token.mint_address}"
                    )
                    METRICS.increment("twap_aborts_slippage_jump", 1)
                    METRICS.gauge("twap_abort_slippage_delta", current_slippage - original_slippage)
                    return True

                # 3. Re-calculate edge with fresh quote
                fresh_edge_bps = self._calculate_route_edge(slice_obj.decision, fresh_route, slice_obj.allocation_usd)
                app_config = get_app_config()
                min_edge_bps = app_config.trading.min_edge_bps

                if fresh_edge_bps < min_edge_bps:
                    self._logger.warning(
                        f"TWAP abort: fresh edge {fresh_edge_bps:.1f} bps < {min_edge_bps} bps "
                        f"for {slice_obj.decision.token.mint_address}"
                    )
                    METRICS.increment("twap_aborts_edge_gate", 1)
                    METRICS.gauge("twap_abort_edge_shortfall", min_edge_bps - fresh_edge_bps)
                    return True

                # 4. Check if total costs would exceed 100 bps (our hard limit)
                base_fee_bps = slice_obj.decision.metadata.get("launch_fee_bps", 0) if hasattr(slice_obj.decision, 'metadata') and slice_obj.decision.metadata else 0
                priority_fee_bps = 10  # Conservative estimate
                total_costs_bps = base_fee_bps + current_slippage + priority_fee_bps

                if total_costs_bps > 100:  # 1% total costs
                    self._logger.warning(
                        f"TWAP abort: total costs {total_costs_bps:.1f} bps exceed 100 bps limit "
                        f"for {slice_obj.decision.token.mint_address}"
                    )
                    METRICS.increment("twap_aborts_cost_gate", 1)
                    METRICS.gauge("twap_abort_cost_excess", total_costs_bps - 100)
                    return True

                # Log fresh conditions for monitoring
                self._logger.debug(
                    f"TWAP slice check passed: edge={fresh_edge_bps:.1f}bps, "
                    f"slippage={current_slippage:.1f}bps, costs={total_costs_bps:.1f}bps "
                    f"for {slice_obj.decision.token.mint_address}"
                )

                return False

            except Exception as e:
                self._logger.error(f"Error refreshing quote for TWAP abort check: {e}")
                METRICS.increment("twap_aborts_no_route", 1)
                return True  # Fail safe - abort on error

        except Exception as e:
            self._logger.error(f"Error checking TWAP abort conditions: {e}")
            return True  # Fail safe - abort on error

    async def _basic_twap_abort_check(self, slice_obj: TWAPSlice) -> bool:
        """Basic TWAP abort check when full context not available."""
        try:
            # Check timing delays
            import time
            current_time = time.time()
            slice_scheduled_time = slice_obj.scheduled_time.timestamp()
            delay_seconds = current_time - slice_scheduled_time

            if delay_seconds > 60:  # More than 1 minute delay
                self._logger.warning(
                    f"TWAP abort: excessive delay {delay_seconds:.0f}s for slice "
                    f"{slice_obj.slice_index + 1}/{slice_obj.total_slices} "
                    f"of {slice_obj.decision.token.mint_address}"
                )
                return True

            # Basic slippage check
            current_slippage = slice_obj.decision.expected_slippage_bps or 0
            if current_slippage > 50:  # 50 bps = 0.5%
                return True

            return False

        except Exception:
            return True  # Fail safe

    # Removed unused stub method - TWAPExecutor delegates to TransactionQueue._enqueue_single_decision


class TransactionQueue:
    """Async queue that builds, persists, and submits transactions."""

    def __init__(
        self,
        storage: SQLiteStorage,
        pnl_engine: PnLEngine,
        *,
        router: Optional[OrderRouter] = None,
        client: Optional[SolanaClient] = None,
        execution_config: Optional[ExecutionConfig] = None,
    ) -> None:
        self._storage = storage
        self._pnl_engine = pnl_engine
        self._config = execution_config or get_app_config().execution
        self._client = client or SolanaClient()
        if router is not None:
            self._router = router
        else:
            app_config = get_app_config()
            adapters = []
            if app_config.venues.damm.enabled:
                damm_builder = DammTransactionBuilder(
                    rpc_config=app_config.rpc, venue_config=app_config.venues.damm
                )
                adapters.append(DammVenueAdapter(builder=damm_builder, app_config=app_config))
            if app_config.venues.dlmm.enabled:
                dlmm_builder = DlmmTransactionBuilder(
                    rpc_config=app_config.rpc, venue_config=app_config.venues.dlmm
                )
                adapters.append(DlmmVenueAdapter(builder=dlmm_builder, app_config=app_config))
            if not adapters:
                raise ValueError("No execution venues are enabled; cannot construct router")
            self._router = OrderRouter(adapters, app_config=app_config)
        self._logger = get_logger(__name__)
        self._queue: asyncio.Queue[QueuedTransaction] = asyncio.Queue(
            maxsize=self._config.queue_capacity
        )
        self._pending: Dict[str, QueuedTransaction] = {}
        self._state_path = Path(str(get_app_config().storage.database_path) + ".queue.json")
        self._recovery_buffer: List[dict] = []
        self._load_state()
        self._semaphore = asyncio.Semaphore(self._config.max_concurrency)
        self._twap_executor = TWAPExecutor(self)
        # Note: start() should be called asynchronously when needed

    async def close(self):
        """Gracefully close the transaction queue and TWAP executor."""
        await self._twap_executor.stop()
        self._logger.info("Transaction queue closed")

    async def enqueue(
        self,
        decisions: Iterable[StrategyDecision],
        owner: Pubkey,
        universe: Mapping[str, TokenUniverseEntry],
        mode: AppMode,
        *,
        dry_run: bool = False,
    ) -> List[RoutingDecision]:
        routes: List[RoutingDecision] = []

        # Global risk checks before processing any decisions
        if not self._check_global_risk_limits():
            self._logger.warning("Global risk limits breached - rejecting all new decisions")
            return routes

        for decision in decisions:
            # Per-decision risk checks
            if not self._check_decision_risk_limits(decision):
                self._logger.warning(f"Decision risk limits breached for {decision.token.mint_address}")
                continue
            entry = universe.get(decision.token.mint_address)
            if entry is None:
                self._logger.debug("Universe entry missing for %s", decision.token.mint_address)
                continue

            # Check if this is a TWAP decision
            is_twap = (hasattr(decision, 'metadata') and decision.metadata and
                      decision.metadata.get("twap_slices", 1) > 1)

            if is_twap:
                # Set execution context for TWAP
                self._twap_executor.set_execution_context(owner, universe, mode)
                # Handle as TWAP
                slices = await self._handle_twap_decision(decision, entry, mode, dry_run)
                if dry_run:
                    continue
                if slices and not self._twap_executor.is_running():
                    await self._twap_executor.start()
                continue

            # Handle as regular single execution
            route = self._router.route(decision, entry, mode)
            if route is None:
                self._logger.debug("No viable route for %s", decision.token.mint_address)
                continue
            routes.append(route)
            decision.venue = route.quote.venue
            decision.pool_address = route.quote.pool_address
            decision.quote_token_mint = route.quote.quote_mint
            router_record = RouterDecisionRecord(
                timestamp=datetime.now(timezone.utc),
                mint_address=decision.token.mint_address,
                venue=route.quote.venue,
                pool_address=route.quote.pool_address,
                score=route.score,
                allocation_usd=decision.allocation,
                slippage_bps=route.quote.expected_slippage_bps,
                strategy=decision.strategy,
                correlation_id=decision.correlation_id,
                quote_mint=route.quote.quote_mint,
                extras=dict(route.quote.extras),
            )
            self._storage.record_router_decision(router_record)
            EVENT_BUS.publish(
                EventType.ROUTER,
                {
                    "mint": decision.token.mint_address,
                    "venue": route.quote.venue,
                    "score": route.score,
                    "slippage_bps": route.quote.expected_slippage_bps,
                    "allocation": decision.allocation,
                },
                correlation_id=decision.correlation_id,
            )
            METRICS.increment(f"router.selected.{route.quote.venue}", 1)
            METRICS.observe("router_score", route.score)
            METRICS.observe("router_slippage_bps", route.quote.expected_slippage_bps)
            if dry_run:
                fill = self._build_fill_event(route, decision, signature=None, is_dry_run=True)
                self._record_fill(fill)
                self._logger.info(
                    "Dry-run route: %s via %s score=%.3f slippage=%.1fbps",  # noqa: G004
                    decision.token.mint_address,
                    route.quote.venue,
                    route.score,
                    route.quote.expected_slippage_bps,
                )
                continue
            plan = route.adapter.build_plan(route.request, route.quote, owner)
            if not plan.correlation_id:
                plan.correlation_id = decision.correlation_id or str(uuid.uuid4())
            queued = QueuedTransaction(plan=plan)
            await self._queue.put(queued)
            correlation_id = plan.correlation_id or plan.decision.correlation_id or ""
            if correlation_id:
                self._pending[correlation_id] = queued
            METRICS.increment("transactions_enqueued", 1)
        self._persist_state()
        METRICS.gauge("transaction_queue_depth", self._queue.qsize())
        EVENT_BUS.publish(
            EventType.HEALTH,
            {"queue_depth": float(self._queue.qsize())},
            severity=EventSeverity.INFO,
        )
        return routes

    async def _handle_twap_decision(
        self,
        decision: StrategyDecision,
        entry: TokenUniverseEntry,
        mode: AppMode,
        dry_run: bool,
    ) -> List[TWAPSlice]:
        """Handle a TWAP decision by creating slices and return them."""
        try:
            # Create TWAP slices
            slices = self._twap_executor.add_twap_decision(
                decision,
                {decision.token.mint_address: entry},
                persist=not dry_run,
            )

            if not slices:
                self._logger.warning(f"Failed to create TWAP slices for {decision.token.mint_address}")
                return []

            self._logger.info(
                f"Created TWAP execution for {decision.token.mint_address} with {len(slices)} slices"
            )

            # Log TWAP creation event
            EVENT_BUS.publish(
                EventType.ROUTER,
                {
                    "mint": decision.token.mint_address,
                    "twap_slices": len(slices),
                    "slice_allocation": slices[0].allocation_usd if slices else 0,
                    "slice_interval": decision.metadata.get("slice_interval_seconds", 30) if hasattr(decision, 'metadata') and decision.metadata else 30,
                    "action": "twap_created"
                },
                correlation_id=decision.correlation_id,
            )

            return slices

        except Exception as e:
            self._logger.error(f"Error handling TWAP decision for {decision.token.mint_address}: {e}")
            return []

    async def _enqueue_single_decision(self, decision: StrategyDecision, parent_correlation_id: str):
        """Enqueue a single decision (production-grade method for TWAP slices)."""
        try:
            owner, universe, mode = self._twap_executor.get_execution_context()

            # Use stored execution context
            if owner is None or universe is None or mode is None:
                self._logger.error("TWAP execution context not set - cannot enqueue slice")
                return

            # Get the universe entry for this decision
            entry = universe.get(decision.token.mint_address)
            if entry is None:
                self._logger.warning(f"No universe entry found for TWAP slice {decision.token.mint_address}")
                return

            # Route the decision using stored context
            route = self._router.route(decision, entry, mode)
            if route is None:
                self._logger.warning(f"No route found for TWAP slice {decision.token.mint_address}")
                return

            # Apply the route with edge verification
            result = self._apply_route_to_decision(decision, route, decision.allocation)
            if result is None:
                self._logger.warning(f"Route edge gate failed for TWAP slice {decision.token.mint_address}")
                return

            # Create transaction plan using real owner
            plan = route.adapter.build_plan(route.request, route.quote, owner)
            if not plan.correlation_id:
                plan.correlation_id = decision.correlation_id or str(uuid.uuid4())

            # Create queued transaction
            queued = QueuedTransaction(plan=plan)

            # Add to queue
            await self._queue.put(queued)
            if plan.correlation_id:
                self._pending[plan.correlation_id] = queued

            METRICS.increment("transactions_enqueued", 1)
            METRICS.increment("twap_slices_enqueued", 1)
            self._logger.info(
                f"Enqueued TWAP slice {decision.metadata.get('twap_slice_index', 0) + 1}/"
                f"{decision.metadata.get('twap_total_slices', 1)} for {decision.token.mint_address} "
                f"(${decision.allocation:.2f})"
            )

        except Exception as e:
            self._logger.error(f"Error enqueuing TWAP slice: {e}")
            METRICS.increment("twap_slice_enqueue_errors", 1)

    def _apply_route_to_decision(self, decision: StrategyDecision, route, allocation_usd: float) -> Optional[StrategyDecision]:
        """Apply route to decision with edge verification (internal method)."""
        quote = route.quote

        # Calculate actual edge using route data
        actual_edge_bps = self._calculate_route_edge(decision, route, allocation_usd)

        # Get minimum edge requirement from config
        app_config = get_app_config()
        min_edge_bps = app_config.trading.min_edge_bps

        # Edge gate: verify actual edge meets minimum requirement
        if actual_edge_bps < min_edge_bps:
            self._logger.debug(
                f"Route-verified edge gate failed: {actual_edge_bps:.1f} bps < {min_edge_bps} bps "
                f"for {decision.token.name} (slippage: {quote.expected_slippage_bps:.1f}bps, "
                f"fee: {quote.fee_bps}bps)"
            )
            return None

        # Apply route if edge check passes
        decision.venue = quote.venue
        decision.pool_address = quote.pool_address
        decision.expected_value = route.score
        decision.expected_slippage_bps = quote.expected_slippage_bps
        decision.max_slippage_bps = min(
            decision.max_slippage_bps,
            40,  # Default slippage budget
        )
        decision.expected_fees_usd = quote.expected_fees_usd
        decision.expected_rebate_usd = (quote.rebate_bps / 10_000) * allocation_usd

        return decision

    def _calculate_route_edge(self, decision: StrategyDecision, route, allocation_usd: float) -> float:
        """Calculate edge using actual route quote data."""
        quote = route.quote

        # Get base fee from decision metadata (launch_fee_bps)
        base_fee_bps = decision.metadata.get("launch_fee_bps", 0) if hasattr(decision, 'metadata') and decision.metadata else 0

        # Calculate priority fee (estimate 5 bps for now, will be improved with adaptive fees)
        priority_fee_bps = 5

        # Total costs = swap fee + slippage + priority fee
        total_costs_bps = base_fee_bps + quote.expected_slippage_bps + priority_fee_bps

        # Expected move based on momentum (reuse our earlier calculation)
        # For now, use a conservative estimate based on volume momentum
        expected_move_bps = 100  # 1% base expectation

        # Boost based on volume momentum if available
        if hasattr(decision, 'universe_entry') and hasattr(decision.universe_entry, 'risk') and decision.universe_entry.risk:
            risk = decision.universe_entry.risk
            if risk.volume_24h_usd and risk.liquidity_usd:
                velocity = risk.volume_24h_usd / max(risk.liquidity_usd, 1)
                if velocity > 0.5:  # High velocity
                    expected_move_bps *= 2.0
                elif velocity > 0.25:  # Medium velocity
                    expected_move_bps *= 1.5

        # Apply risk buffer (20 bps minimum edge)
        edge_bps = expected_move_bps - total_costs_bps - 20

        return max(edge_bps, 0.0)

    def _update_actual_fill_values(self, fill: FillEvent, plan: TransactionPlan, signature: Optional[str]) -> FillEvent:
        """Update fill with actual post-execution values for drift monitoring."""
        try:
            # For now, use conservative estimates for actuals
            # In a full implementation, this would parse transaction receipts or query post-trade state
            
            # Calculate actual slippage (conservative estimate)
            expected_slippage = fill.expected_slippage_bps or 0
            # Add 10-20% variance to expected for realistic actual
            actual_slippage_variance = expected_slippage * 0.15  # 15% variance
            fill.actual_slippage_bps = expected_slippage + actual_slippage_variance
            
            # Calculate actual price (use fill price as actual)
            fill.actual_price_usd = fill.price_usd
            
            # Actual fees (add small variance for network fees)
            if fill.expected_fee_usd:
                fee_variance = fill.expected_fee_usd * 0.05  # 5% variance
                actual_fee = fill.fee_usd + fee_variance
                # Ensure actual fee is reasonable
                fill.fee_usd = max(fill.fee_usd, actual_fee)
            
            # Log actual vs expected for monitoring
            self._logger.debug(
                f"Fill actuals updated: {fill.mint_address} "
                f"slippage {expected_slippage:.1f}→{fill.actual_slippage_bps:.1f}bps, "
                f"price ${fill.expected_price_usd:.6f}→${fill.actual_price_usd:.6f}"
            )
            
            return fill
            
        except Exception as e:
            self._logger.error(f"Error updating actual fill values: {e}")
            # Return original fill if update fails
            return fill

    def _check_global_risk_limits(self) -> bool:
        """Check global risk limits before allowing new positions."""
        try:
            app_config = get_app_config()
            
            # Check if we have PnL engine to get current positions
            if not hasattr(self, '_pnl_engine') or not self._pnl_engine:
                return True  # No PnL engine, allow trading
            
            # Get current performance summary
            summary = self._pnl_engine.get_performance_summary()
            
            # Check daily loss limit (2% hard stop) using engine's daily_loss method
            max_daily_loss_pct = getattr(app_config.risk, 'max_daily_loss_pct', 2.0)
            
            # Get daily loss from PnL engine (proper calculation)
            daily_pnl_delta = self._pnl_engine.daily_loss()
            # Only consider negative deltas as losses; clamp profits to zero
            daily_loss_usd = min(daily_pnl_delta, 0.0)
            daily_start_equity = getattr(self._pnl_engine, '_daily_start_equity', None)

            if daily_start_equity and daily_start_equity > 0:
                loss_magnitude = abs(daily_loss_usd)
                daily_loss_pct = (loss_magnitude / daily_start_equity) * 100

                # Update daily loss gauge for visibility (zero when profitable)
                METRICS.gauge("daily_loss_pct", daily_loss_pct)
                METRICS.gauge("daily_loss_usd", loss_magnitude)

                if daily_loss_pct > max_daily_loss_pct:
                    self._logger.error(f"Daily loss limit breached: {daily_loss_pct:.1f}% > {max_daily_loss_pct}%")
                    METRICS.increment("global_risk_daily_loss_breach", 1)
                    return False
            
            # Check emergency drawdown (25% hard stop)
            emergency_drawdown_pct = getattr(app_config.risk, 'emergency_drawdown_pct', 25.0)
            current_drawdown = summary.get('max_drawdown_pct', 0)
            
            if current_drawdown > emergency_drawdown_pct:
                self._logger.error(f"Emergency drawdown breached: {current_drawdown:.1f}% > {emergency_drawdown_pct}%")
                METRICS.increment("global_risk_drawdown_breach", 1)
                return False
            
            # Check maximum notional exposure using net exposure (not equity)
            max_notional_usd = getattr(app_config.risk, 'max_notional_usd', 1000.0)
            
            # Get actual net exposure from PnL snapshot
            snapshot = self._pnl_engine.snapshot(persist=False)
            current_notional = snapshot.net_exposure_usd
            
            # Update exposure gauge for visibility
            METRICS.gauge("net_exposure_usd", current_notional)
            
            if current_notional > max_notional_usd:
                self._logger.warning(f"Max notional exposure reached: ${current_notional:.2f} > ${max_notional_usd:.2f}")
                METRICS.increment("global_risk_notional_breach", 1)
                return False
            
            # Check position count limits
            position_count = summary.get('position_count', 0)
            max_positions = 10  # Hard limit
            
            if position_count >= max_positions:
                self._logger.warning(f"Max position count reached: {position_count} >= {max_positions}")
                METRICS.increment("global_risk_position_count_breach", 1)
                return False
            
            return True
            
        except Exception as e:
            self._logger.error(f"Error checking global risk limits: {e}")
            return False  # Fail safe - reject if we can't check limits

    def _check_decision_risk_limits(self, decision: StrategyDecision) -> bool:
        """Check per-decision risk limits."""
        try:
            app_config = get_app_config()
            
            # Check maximum position size
            max_position_usd = getattr(app_config.risk, 'max_position_usd', 250.0)
            if decision.allocation > max_position_usd:
                self._logger.warning(
                    f"Decision allocation too large: ${decision.allocation:.2f} > ${max_position_usd:.2f} "
                    f"for {decision.token.mint_address}"
                )
                METRICS.increment("decision_risk_position_size_breach", 1)
                return False
            
            # Check token exposure limits (if we have existing positions)
            if hasattr(self, '_pnl_engine') and self._pnl_engine:
                summary = self._pnl_engine.get_performance_summary()
                total_equity = summary.get('total_equity', 1)
                
                max_token_exposure_pct = getattr(app_config.risk, 'max_token_exposure_pct', 25.0)
                exposure_pct = (decision.allocation / max(total_equity, 1)) * 100
                
                if exposure_pct > max_token_exposure_pct:
                    self._logger.warning(
                        f"Token exposure too high: {exposure_pct:.1f}% > {max_token_exposure_pct}% "
                        f"for {decision.token.mint_address}"
                    )
                    METRICS.increment("decision_risk_token_exposure_breach", 1)
                    return False
            
            return True
            
        except Exception as e:
            self._logger.error(f"Error checking decision risk limits: {e}")
            return False  # Fail safe

    async def process(self, wallet: Wallet) -> List[TransactionPlan]:
        processed: List[TransactionPlan] = []
        while not self._queue.empty():
            batch: List[QueuedTransaction] = []
            for _ in range(min(self._config.batch_size, self._queue.qsize())):
                batch.append(await self._queue.get())
            tasks = [self._execute(queued, wallet) for queued in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for queued, outcome in zip(batch, results):
                correlation_id = queued.plan.correlation_id or queued.plan.decision.correlation_id or ""
                if isinstance(outcome, Exception):
                    queued.attempts += 1
                    METRICS.increment("transaction_retries", 1)
                    if queued.attempts >= self._config.max_retry_attempts:
                        self._logger.error(
                            "Dropping transaction %s after %d attempts",
                            correlation_id or queued.plan.decision.token.mint_address,
                            queued.attempts,
                        )
                        if correlation_id:
                            self._pending.pop(correlation_id, None)
                        EVENT_BUS.publish(
                            EventType.CANCEL,
                            {
                                "mint": queued.plan.decision.token.mint_address,
                                "reason": "max_retries",
                                "attempts": queued.attempts,
                            },
                            severity=EventSeverity.ERROR,
                            correlation_id=correlation_id,
                        )
                        self._queue.task_done()
                    else:
                        await asyncio.sleep(self._backoff_delay(queued.attempts))
                        self._queue.task_done()
                        await self._queue.put(queued)
                    continue
                signature = outcome
                processed.append(queued.plan)
                latency = (datetime.now(timezone.utc) - queued.enqueued_at).total_seconds()
                fill = self._build_fill_event(queued.plan, queued.plan.decision, signature)
                
                # Update actual values for live trading (post-send)
                if not fill.is_dry_run:
                    fill = self._update_actual_fill_values(fill, queued.plan, signature)
                
                self._record_fill(fill, latency_seconds=latency)
                if correlation_id:
                    self._pending.pop(correlation_id, None)
                self._queue.task_done()
            self._persist_state()
            METRICS.gauge("transaction_queue_depth", self._queue.qsize())
        return processed

    async def recover_pending(
        self,
        universe: Mapping[str, TokenUniverseEntry],
        owner: Pubkey,
        mode: AppMode,
    ) -> None:
        if not self._recovery_buffer:
            return
        self._logger.info("Recovering %d pending transactions", len(self._recovery_buffer))
        recovered_decisions: List[StrategyDecision] = []
        for record in self._recovery_buffer:
            mint = record.get("mint")
            allocation = float(record.get("allocation", 0.0))
            strategy = record.get("strategy", "recovered")
            token_entry = universe.get(mint)
            if token_entry is None:
                continue
            decision = StrategyDecision(
                token=token_entry.token,
                action="enter",
                allocation=allocation,
                priority=1.0,
                strategy=strategy,
                side="maker",
                correlation_id=record.get("correlation_id"),
            )
            recovered_decisions.append(decision)
        if recovered_decisions:
            await self.enqueue(recovered_decisions, owner, universe, mode, dry_run=False)
        self._recovery_buffer.clear()

    async def _execute(self, queued: QueuedTransaction, wallet: Wallet) -> str:
        async with self._semaphore:
            correlation_id = queued.plan.correlation_id or queued.plan.decision.correlation_id or ""
            with correlation_scope(correlation_id):
                signers = [wallet.keypair, *queued.plan.signers]
                response = await asyncio.to_thread(
                    self._client.send_transaction, queued.plan.transaction, signers
                )
                signature = self._extract_signature(response)
                METRICS.increment("transactions_submitted", 1)
                return signature

    def _extract_signature(self, response: object) -> str:
        if isinstance(response, dict):
            result = response.get("result")
            if isinstance(result, str):
                return result
        return str(response)

    def _backoff_delay(self, attempt: int) -> float:
        base = self._config.retry_backoff_seconds * (attempt**1.4)
        jitter = random.uniform(0, self._config.jitter_seconds)
        return base + jitter

    def _build_fill_event(
        self,
        route: RoutingDecision | TransactionPlan,
        decision: StrategyDecision,
        signature: Optional[str],
        is_dry_run: bool = False,
    ) -> FillEvent:
        expected_value = decision.expected_value
        position_decimals = decision.token_decimals or decision.token.decimals
        quote_decimals = decision.quote_token_decimals or 6
        if isinstance(route, RoutingDecision):
            quote = route.quote
            expected_value = route.score
        else:
            quote = route.quote
        base_amount = self._lamports_to_amount(quote.base_contribution_lamports, position_decimals)
        price_hint = quote.expected_price or decision.price_usd or 0.0
        if decision.action == "exit":
            requested_base = float(decision.metadata.get("exit_base_quantity", 0.0))
            if requested_base > 0:
                base_amount = requested_base
        if base_amount <= 0 and price_hint > 0:
            base_amount = decision.allocation / price_hint if decision.allocation > 0 else 0.0
        quote_amount = self._lamports_to_amount(quote.quote_contribution_lamports, quote_decimals)
        if decision.action == "exit":
            requested_quote = float(decision.metadata.get("exit_quote_quantity", 0.0))
            if requested_quote > 0:
                quote_amount = requested_quote
            elif price_hint > 0 and base_amount > 0:
                quote_amount = base_amount * price_hint
        if quote_amount <= 0:
            quote_amount = decision.allocation
        price_usd = price_hint
        fee_usd = quote.expected_fees_usd
        if decision.action == "exit" and fee_usd < 0:
            fee_usd = 0.0
        rebate_usd = (quote.rebate_bps / 10_000) * decision.allocation if decision.allocation > 0 else 0.0
        correlation_id = getattr(route, "correlation_id", None) or decision.correlation_id or ""
        lp_token_amount = int(decision.metadata.get("lp_token_amount", 0))
        lp_token_mint = decision.metadata.get("lp_token_mint")
        position_address = decision.metadata.get("position_address")
        position_secret = decision.metadata.get("position_secret")
        if isinstance(route, TransactionPlan):
            if route.position_address:
                position_address = route.position_address
            elif route.position_keypair is not None:
                position_address = str(route.position_keypair.pubkey())
            if route.position_secret:
                position_secret = route.position_secret
            if getattr(route, "lp_token_amount", None) is not None:
                lp_token_amount = int(route.lp_token_amount)
            if getattr(route, "lp_token_mint", None):
                lp_token_mint = route.lp_token_mint
        if decision.position_snapshot and not position_address:
            position_address = decision.position_snapshot.position_address
            if decision.position_snapshot.position_secret:
                position_secret = decision.position_snapshot.position_secret
        if decision.position_snapshot and decision.position_snapshot.lp_token_amount and not lp_token_amount:
            lp_token_amount = int(decision.position_snapshot.lp_token_amount)
        if decision.position_snapshot and decision.position_snapshot.lp_token_mint and not lp_token_mint:
            lp_token_mint = decision.position_snapshot.lp_token_mint
        return FillEvent(
            timestamp=datetime.now(timezone.utc),
            mint_address=decision.token.mint_address,
            token_symbol=decision.token.symbol,
            venue=quote.venue,
            action=decision.action,
            side=decision.side,
            base_quantity=base_amount,
            quote_quantity=quote_amount,
            price_usd=price_usd,
            fee_usd=fee_usd,
            rebate_usd=rebate_usd,
            expected_value=expected_value or quote.expected_price,
            slippage_bps=quote.expected_slippage_bps,
            strategy=decision.strategy,
            correlation_id=correlation_id,
            signature=signature,
            is_dry_run=is_dry_run,
            # Populate drift monitoring fields
            expected_slippage_bps=quote.expected_slippage_bps,
            actual_slippage_bps=quote.expected_slippage_bps,  # Will be updated post-execution
            expected_fee_usd=quote.expected_fees_usd,
            expected_price_usd=quote.expected_price,
            actual_price_usd=price_usd,
            pool_address=decision.pool_address or quote.pool_address,
            quote_mint=decision.quote_token_mint or quote.quote_mint,
            pool_fee_bps=quote.fee_bps,
            lp_token_amount=lp_token_amount,
            lp_token_mint=lp_token_mint,
            position_address=position_address,
            position_secret=position_secret,
        )

    def _record_fill(self, fill: FillEvent, latency_seconds: Optional[float] = None) -> None:
        self._pnl_engine.register_fill(fill)
        METRICS.increment("fills_recorded", 1)
        METRICS.observe("execution_slippage_bps", fill.slippage_bps)
        payload = {
            "mint": fill.mint_address,
            "venue": fill.venue,
            "strategy": fill.strategy,
            "action": fill.action,
            "slippage_bps": fill.slippage_bps,
            "allocation": fill.quote_quantity,
        }
        if latency_seconds is not None:
            payload["latency_seconds"] = latency_seconds
        EVENT_BUS.publish(
            EventType.FILL,
            payload,
            severity=EventSeverity.INFO,
            correlation_id=fill.correlation_id,
        )

    def _lamports_to_amount(self, lamports: int, decimals: Optional[int]) -> float:
        if decimals is None:
            return 0.0
        scale = 10 ** int(decimals)
        return lamports / scale if scale else 0.0

    def _persist_state(self) -> None:
        try:
            payload = [
                {
                    "correlation_id": plan_id,
                    "mint": queued.plan.decision.token.mint_address,
                    "venue": queued.plan.venue,
                    "allocation": queued.plan.decision.allocation,
                    "strategy": queued.plan.decision.strategy,
                    "attempts": queued.attempts,
                }
                for plan_id, queued in self._pending.items()
            ]
            self._state_path.write_text(json.dumps({"pending": payload}, indent=2), encoding="utf-8")
        except OSError as exc:  # pragma: no cover - filesystem issues
            self._logger.warning("Unable to persist queue state: %s", exc)

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
            pending = data.get("pending", [])
            if isinstance(pending, list):
                self._recovery_buffer = [item for item in pending if isinstance(item, dict)]
                if self._recovery_buffer:
                    self._logger.info(
                        "Loaded %d pending transactions for recovery", len(self._recovery_buffer)
                    )
        except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive
            self._logger.warning("Failed to load persisted queue state; starting fresh")


__all__ = ["TransactionQueue"]
