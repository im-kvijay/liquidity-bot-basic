"""Portfolio and PnL accounting utilities."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, Mapping, Optional

from ..config.settings import MarkPriceSource, PnLConfig, get_app_config
from ..datalake.schemas import (
    FillEvent,
    MetricsSnapshot,
    PnLSnapshot,
    PortfolioPosition,
    TokenUniverseEntry,
)
from ..datalake.storage import SQLiteStorage
from ..ingestion.pricing import PriceOracle
from ..monitoring.logger import get_logger
from ..monitoring.metrics import METRICS


@dataclass(slots=True)
class ExposureSummary:
    """Represents the current notional exposure for a token."""

    mint_address: str
    notional_usd: float
    base_quantity: float
    last_price: float


class PnLEngine:
    """Tracks realised and unrealised PnL with attribution."""

    def __init__(
        self,
        storage: SQLiteStorage,
        *,
        price_oracle: Optional[PriceOracle] = None,
        config: Optional[PnLConfig] = None,
    ) -> None:
        self._storage = storage
        self._price_oracle = price_oracle or PriceOracle()
        self._config = config or get_app_config().pnl
        self._logger = get_logger(__name__)
        self._positions: Dict[str, PortfolioPosition] = {}
        self._realized_usd = 0.0
        self._fees_usd = 0.0
        self._rebates_usd = 0.0
        self._venue_realized: Dict[str, float] = defaultdict(float)
        self._pair_realized: Dict[str, float] = defaultdict(float)
        self._strategy_realized: Dict[str, float] = defaultdict(float)
        self._max_equity = 0.0
        self._daily_start_equity: Optional[float] = None
        self._daily_start_time: datetime = datetime.now(timezone.utc)

    @property
    def realized_pnl_usd(self) -> float:
        return self._realized_usd - self._fees_usd + self._rebates_usd

    def get_dry_run_pnl(self, current_prices: Optional[Mapping[str, float]] = None) -> Dict[str, float]:
        """Calculate accurate dry run PnL including simulated fills."""
        if current_prices is None:
            current_prices = {}

        total_unrealized = 0.0
        total_realized = self.realized_pnl_usd
        position_details = {}

        for mint, position in self._positions.items():
            current_price = current_prices.get(mint, position.entry_price)
            if current_price <= 0:
                continue

            # Calculate unrealized PnL for this position
            value = position.base_quantity * current_price
            cost_basis = abs(position.base_quantity) * position.entry_price
            unrealized = value - cost_basis if position.base_quantity > 0 else cost_basis - value

            total_unrealized += unrealized
            position_details[mint] = {
                'unrealized': unrealized,
                'value': value,
                'quantity': position.base_quantity,
                'entry_price': position.entry_price,
                'current_price': current_price,
                'pool': position.pool_address[:8] if position.pool_address else 'N/A',
                'venue': position.venue,
                'strategy': position.strategy
            }

        return {
            'total_unrealized': total_unrealized,
            'total_realized': total_realized,
            'total_equity': total_realized + total_unrealized,
            'position_count': len(self._positions),
            'positions': position_details,
            'performance_metrics': self._calculate_performance_metrics(position_details, total_realized, total_unrealized)
        }

    def get_performance_summary(self) -> Dict[str, float]:
        """Get a comprehensive performance summary for monitoring."""
        dry_run_pnl = self.get_dry_run_pnl()
        metrics = dry_run_pnl['performance_metrics']

        return {
            'total_equity': metrics['total_equity'],
            'realized_pnl': metrics['realized_pnl'],
            'unrealized_pnl': metrics['unrealized_pnl'],
            'position_count': metrics['position_count'],
            'win_rate_pct': metrics['win_rate_pct'],
            'average_position_size': metrics['average_position_size'],
            'max_drawdown_pct': metrics['max_drawdown_pct'],
            'profit_factor': metrics['profit_factor'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'daily_return_pct': self._calculate_daily_return(),
            'hourly_return_pct': self._calculate_hourly_return(),
            'total_fees_paid': self._fees_usd,
            'total_rebates_earned': self._rebates_usd,
            'net_fees': self._fees_usd - self._rebates_usd
        }

    def _calculate_daily_return(self) -> float:
        """Calculate daily return percentage."""
        if self._daily_start_equity is None or self._daily_start_equity == 0:
            return 0.0
        current_equity = self.realized_pnl_usd + self._calculate_total_unrealized()
        return (current_equity / self._daily_start_equity - 1) * 100

    def _calculate_hourly_return(self) -> float:
        """Calculate hourly return percentage."""
        hours_elapsed = (datetime.now(timezone.utc) - self._daily_start_time).total_seconds() / 3600
        if hours_elapsed == 0:
            return 0.0
        daily_return = self._calculate_daily_return()
        return daily_return / max(hours_elapsed, 1) * 24  # Annualized hourly return

    def _calculate_total_unrealized(self) -> float:
        """Calculate total unrealized PnL across all positions."""
        total_unrealized = 0.0
        for mint, position in self._positions.items():
            # Use entry price as fallback if no current price available
            current_price = getattr(position, 'current_price', position.entry_price)
            if current_price <= 0:
                continue
            value = position.base_quantity * current_price
            cost_basis = abs(position.base_quantity) * position.entry_price
            unrealized = value - cost_basis if position.base_quantity > 0 else cost_basis - value
            total_unrealized += unrealized
        return total_unrealized

    def log_performance_summary(self) -> None:
        """Log comprehensive performance summary for monitoring."""
        summary = self.get_performance_summary()

        self._logger.info("=== PERFORMANCE SUMMARY ===")
        self._logger.info(f"Total Equity: ${summary['total_equity']:.2f}")
        self._logger.info(f"Realized PnL: ${summary['realized_pnl']:.2f}")
        self._logger.info(f"Unrealized PnL: ${summary['unrealized_pnl']:.2f}")
        self._logger.info(f"Positions: {summary['position_count']}")
        self._logger.info(f"Win Rate: {summary['win_rate_pct']:.1f}%")
        self._logger.info(f"Average Position: ${summary['average_position_size']:.2f}")
        self._logger.info(f"Max Drawdown: {summary['max_drawdown_pct']:.1f}%")
        self._logger.info(f"Profit Factor: {summary['profit_factor']:.2f}")
        self._logger.info(f"Daily Return: {summary['daily_return_pct']:.2f}%")
        self._logger.info(f"Hourly Return: {summary['hourly_return_pct']:.2f}%")
        self._logger.info(f"Net Fees: ${summary['net_fees']:.2f}")
        self._logger.info("=" * 30)

    def _calculate_performance_metrics(
        self,
        positions: Dict,
        realized: float,
        unrealized: float
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics for profitability analysis."""
        total_equity = realized + unrealized

        # Win rate calculation
        winning_positions = sum(1 for pos in positions.values() if pos['unrealized'] > 0)
        win_rate = (winning_positions / len(positions)) * 100 if positions else 0

        # Average position size
        avg_position_size = sum(abs(pos['value']) for pos in positions.values()) / len(positions) if positions else 0

        # Sharpe ratio approximation (simplified)
        # This would need historical data for accurate calculation
        volatility = 0.0  # Would need historical volatility data
        risk_free_rate = 0.02  # 2% annual risk-free rate
        sharpe_ratio = (total_equity / avg_position_size - risk_free_rate) / volatility if volatility > 0 else 0

        # Maximum drawdown (simplified)
        max_drawdown = 0.0
        current_peak = max(total_equity, self._max_equity)
        if current_peak > 0:
            max_drawdown = (current_peak - total_equity) / current_peak * 100

        # Profit factor (gross profit / gross loss)
        gross_profit = sum(pos['unrealized'] for pos in positions.values() if pos['unrealized'] > 0)
        gross_loss = abs(sum(pos['unrealized'] for pos in positions.values() if pos['unrealized'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        return {
            'win_rate_pct': win_rate,
            'average_position_size': avg_position_size,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'profit_factor': profit_factor,
            'total_equity': total_equity,
            'realized_pnl': realized,
            'unrealized_pnl': unrealized,
            'position_count': len(positions)
        }

    def prime(self, positions: Iterable[PortfolioPosition]) -> None:
        """Seed the engine with any persisted positions."""

        for position in positions:
            self._positions[position.token.mint_address] = position
        # Establish baseline equity for drawdown and daily-loss checks.
        snapshot = self.snapshot()
        equity = snapshot.realized_usd + snapshot.unrealized_usd - snapshot.fees_usd + snapshot.rebates_usd
        self._daily_start_equity = equity
        self._max_equity = max(self._max_equity, equity)

    def register_fill(self, fill: FillEvent) -> PortfolioPosition:
        """Update accounting with a new fill (live or simulated)."""

        position = self._positions.get(fill.mint_address)
        if position is None:
            token = self._storage.get_token(fill.mint_address)
            if token is None:
                raise ValueError(f"Token metadata for {fill.mint_address} missing in storage")
            position = PortfolioPosition(
                token=token,
                pool_address=fill.pool_address or "",
                allocation=0.0,
                entry_price=fill.price_usd,
                created_at=fill.timestamp,
                unrealized_pnl_pct=0.0,
                position_address=None,
                position_secret=None,
                venue=fill.venue,
                strategy=fill.strategy,
            )
            # Initialize peak price for trailing stops
            position.peak_price = fill.price_usd
            position.peak_timestamp = fill.timestamp
        direction = self._classify_direction(fill)
        base_abs = abs(fill.base_quantity)
        previous_quantity = position.base_quantity
        realised_component = 0.0
        position.pool_address = fill.pool_address or position.pool_address
        position.pool_fee_bps = fill.pool_fee_bps or position.pool_fee_bps
        if fill.pool_address:
            position.pool_address = fill.pool_address
        if fill.pool_fee_bps is not None:
            position.pool_fee_bps = fill.pool_fee_bps
        position.venue = fill.venue or position.venue
        position.strategy = fill.strategy or position.strategy
        if fill.position_address:
            position.position_address = fill.position_address
        if fill.position_secret:
            position.position_secret = fill.position_secret
        if fill.lp_token_amount:
            if direction > 0:
                position.lp_token_amount += int(fill.lp_token_amount)
            else:
                position.lp_token_amount = max(position.lp_token_amount - int(fill.lp_token_amount), 0)
        if fill.lp_token_mint:
            position.lp_token_mint = fill.lp_token_mint
        if direction > 0:
            new_quantity = previous_quantity + base_abs
            if new_quantity <= 0:
                new_quantity = 0.0
            if new_quantity > 0:
                weighted_cost = (position.entry_price * previous_quantity) + (fill.price_usd * base_abs)
                position.entry_price = weighted_cost / new_quantity
                # Update peak price if this is a new position or price increased
                current_price = fill.price_usd
                if position.peak_price is None or current_price > position.peak_price:
                    position.peak_price = current_price
                    position.peak_timestamp = fill.timestamp
            position.base_quantity = new_quantity
            notional = base_abs * fill.price_usd
            if notional <= 0:
                notional = fill.quote_quantity
            position.quote_quantity += max(fill.quote_quantity, 0.0)
            position.allocation += max(notional, 0.0)
            position.created_at = fill.timestamp
        else:
            quantity_reduced = min(base_abs, previous_quantity)
            realised_component = (fill.price_usd - position.entry_price) * quantity_reduced
            self._realized_usd += realised_component
            position.realized_pnl_usd += realised_component
            position.base_quantity = max(previous_quantity - quantity_reduced, 0.0)
            notional = quantity_reduced * fill.price_usd
            if notional <= 0:
                notional = fill.quote_quantity
            position.quote_quantity = max(position.quote_quantity - max(fill.quote_quantity, 0.0), 0.0)
            position.allocation = max(position.allocation - max(notional, 0.0), 0.0)
            if position.base_quantity == 0:
                position.entry_price = fill.price_usd
                position.quote_quantity = 0.0
                position.allocation = 0.0
                position.lp_token_amount = 0
                position.position_address = None
                position.position_secret = None
        self._fees_usd += fill.fee_usd
        self._rebates_usd += fill.rebate_usd
        position.fees_paid_usd += fill.fee_usd
        position.rebates_earned_usd += fill.rebate_usd
        key_pair = f"{fill.mint_address}:{fill.venue}"
        self._venue_realized[fill.venue] += realised_component
        self._pair_realized[key_pair] += realised_component
        self._strategy_realized[fill.strategy] += realised_component

        if position.base_quantity <= 0 and direction < 0:
            self._positions.pop(fill.mint_address, None)
            self._storage.delete_position(fill.mint_address)
        else:
            self._positions[fill.mint_address] = position
            self._storage.upsert_position(position)

        self._storage.record_fill(fill)
        METRICS.increment("fills_processed", 1)
        
        # Track slippage and fee drift
        if hasattr(fill, 'expected_slippage_bps') and hasattr(fill, 'actual_slippage_bps'):
            slippage_alert = METRICS.track_trading_metric(
                "slippage", 
                fill.expected_slippage_bps, 
                fill.actual_slippage_bps
            )
            if slippage_alert:
                self._logger.warning(f"Slippage drift alert: {slippage_alert}")
        
        if hasattr(fill, 'expected_fee_usd') and fill.fee_usd > 0:
            fee_alert = METRICS.track_trading_metric(
                "fees", 
                getattr(fill, 'expected_fee_usd', fill.fee_usd), 
                fill.fee_usd
            )
            if fee_alert:
                self._logger.warning(f"Fee drift alert: {fee_alert}")
        
        return position

    def snapshot(
        self,
        universe: Optional[Mapping[str, TokenUniverseEntry]] = None,
        *,
        persist: bool = True,
    ) -> PnLSnapshot:
        """Capture a fresh snapshot and optionally persist it."""

        now = datetime.now(timezone.utc)
        total_unrealised = 0.0
        inventory_value = 0.0
        net_exposure = 0.0
        venue_breakdown: Dict[str, float] = defaultdict(float)
        pair_breakdown: Dict[str, float] = defaultdict(float)
        strategy_breakdown: Dict[str, float] = defaultdict(float)
        mark_source = self._config.mark_price_source
        twap_window = self._config.twap_window_seconds
        vwap_window = self._config.vwap_window_seconds
        universe = universe or {}

        exposures_map: Dict[str, float] = {}
        for mint, position in list(self._positions.items()):
            entry = universe.get(mint)
            fallback_price = position.last_mark_price or (
                entry.liquidity_event.price_usd if entry and entry.liquidity_event else None
            )
            if fallback_price is None:
                fallback_price = position.entry_price
            mark_price = self._resolve_mark_price(
                mint,
                source=mark_source,
                fallback=fallback_price,
                twap_window=twap_window,
                vwap_window=vwap_window,
            )
            if mark_price is None:
                mark_price = position.entry_price
            cost_basis = position.base_quantity * position.entry_price
            market_value = position.base_quantity * mark_price
            unrealised_component = market_value - cost_basis
            position.unrealized_pnl_usd = unrealised_component
            position.unrealized_pnl_pct = (
                ((mark_price / position.entry_price) - 1.0) * 100.0 if position.entry_price else 0.0
            )
            position.last_mark_price = mark_price
            position.last_mark_timestamp = now
            total_unrealised += unrealised_component
            inventory_value += market_value
            net_exposure += abs(market_value)
            venue_breakdown[position.venue or "unknown"] += unrealised_component
            pair_breakdown[f"{mint}:{position.venue or 'unknown'}"] += unrealised_component
            strategy_breakdown[position.strategy or "unspecified"] += unrealised_component
            exposures_map[mint] = market_value

        realised = self.realized_pnl_usd
        equity = realised + total_unrealised
        if self._max_equity < equity:
            self._max_equity = equity
        drawdown = 0.0
        if self._max_equity > 0:
            drawdown = max(0.0, (self._max_equity - equity) / self._max_equity)

        snapshot = PnLSnapshot(
            timestamp=now,
            realized_usd=realised,
            unrealized_usd=total_unrealised,
            fees_usd=self._fees_usd,
            rebates_usd=self._rebates_usd,
            inventory_value_usd=inventory_value,
            net_exposure_usd=net_exposure,
            drawdown_pct=drawdown,
            venue_breakdown=dict(venue_breakdown),
            pair_breakdown=dict(pair_breakdown),
            strategy_breakdown=dict(strategy_breakdown),
        )
        if persist:
            self._storage.record_pnl_snapshot(snapshot)
            for position in self._positions.values():
                self._storage.upsert_position(position)
            self._storage.persist_metrics_snapshot(
                MetricsSnapshot(timestamp=now, data=METRICS.snapshot())
            )
        METRICS.gauge("inventory_value_usd", inventory_value)
        METRICS.gauge("unrealized_pnl_usd", total_unrealised)
        METRICS.gauge("realized_pnl_usd", realised)
        METRICS.gauge("net_exposure_usd", net_exposure)
        METRICS.gauge("drawdown_pct", drawdown)
        METRICS.set_mapping("inventory_exposure_usd", exposures_map)
        return snapshot

    def exposures(self) -> Dict[str, ExposureSummary]:
        """Return the current exposures per mint."""

        summary: Dict[str, ExposureSummary] = {}
        for mint, position in self._positions.items():
            last_price = position.last_mark_price or position.entry_price
            summary[mint] = ExposureSummary(
                mint_address=mint,
                notional_usd=position.base_quantity * last_price,
                base_quantity=position.base_quantity,
                last_price=last_price,
            )
        return summary

    def daily_loss(self) -> float:
        """Return the drawdown relative to the session start."""

        if self._daily_start_equity is None:
            return 0.0
        current_equity = self.realized_pnl_usd + sum(
            position.unrealized_pnl_usd for position in self._positions.values()
        )
        return current_equity - self._daily_start_equity

    def _classify_direction(self, fill: FillEvent) -> int:
        if fill.action in {"exit", "sell", "withdraw", "lp_remove"}:
            return -1
        if fill.side.lower() in {"short", "sell", "exit"}:
            return -1
        return 1

    def _resolve_mark_price(
        self,
        mint: str,
        *,
        source: MarkPriceSource,
        fallback: Optional[float],
        twap_window: int,
        vwap_window: int,
    ) -> Optional[float]:
        if source == MarkPriceSource.TWAP:
            price = self._price_oracle.get_mark_price(mint, source="twap", window_seconds=twap_window)
        elif source == MarkPriceSource.VWAP:
            price = self._price_oracle.get_mark_price(mint, source="vwap", window_seconds=vwap_window)
        elif source == MarkPriceSource.MIDPRICE:
            price = fallback
        else:
            price = self._price_oracle.get_mark_price(mint, source="oracle")
        if price is None:
            price = fallback
        return price


__all__ = ["PnLEngine", "ExposureSummary"]
