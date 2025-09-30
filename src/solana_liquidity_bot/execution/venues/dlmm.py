"""Adapter for Meteora DLMM liquidity venues."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from solders.pubkey import Pubkey

from ...config.settings import AppConfig, AppMode, DlmmVenueConfig, get_app_config
from ...datalake.schemas import DlmmPoolSnapshot, TokenRiskMetrics, TokenUniverseEntry
from ...monitoring.logger import get_logger
from ...utils.constants import STABLECOIN_MINTS
from ..transaction_builder import DlmmTransactionBuilder
from .base import PoolContext, QuoteRequest, TransactionPlan, VenueAdapter, VenueQuote


@dataclass(slots=True)
class _DlmmContext:
    pool: DlmmPoolSnapshot
    liquidity_usd: float
    expected_price: Optional[float]


class DlmmVenueAdapter(VenueAdapter):
    """Quotes and builds execution plans for Meteora DLMM pools."""

    name = "dlmm"

    def __init__(
        self,
        builder: Optional[DlmmTransactionBuilder] = None,
        app_config: Optional[AppConfig] = None,
    ) -> None:
        self._app_config = app_config or get_app_config()
        self._config: DlmmVenueConfig = self._app_config.venues.dlmm
        self._builder = builder or DlmmTransactionBuilder(self._app_config.rpc)
        self._logger = get_logger(__name__)

    def pools(self, entry: TokenUniverseEntry) -> Iterable[PoolContext]:
        for pool in entry.dlmm_pools:
            metadata = {
                "bin_step": float(pool.bin_step or 0),
                "base_virtual": float(pool.base_virtual_liquidity or 0.0),
                "quote_virtual": float(pool.quote_virtual_liquidity or 0.0),
            }
            yield PoolContext(
                venue=self.name,
                address=pool.address,
                base_mint=pool.base_token_mint,
                quote_mint=pool.quote_token_mint,
                base_liquidity=pool.base_token_amount,
                quote_liquidity=pool.quote_token_amount,
                fee_bps=pool.fee_bps,
                tvl_usd=pool.tvl_usd,
                volume_24h_usd=pool.volume_24h_usd,
                price_usd=pool.price_usd,
                is_active=pool.is_active,
                metadata=metadata,
            )

    def quote(self, request: QuoteRequest) -> Optional[VenueQuote]:
        if request.decision.action == "exit":
            return self._quote_exit(request)
        if not request.pool.is_active:
            return None
        allocation = max(request.allocation_usd, 0.0)
        if allocation <= 0:
            return None
        context = self._derive_context(request.pool, request.base_price_hint, request.quote_price_hint)
        if context is None:
            return None
        if context.liquidity_usd < self._config.min_liquidity_usd:
            return None
        expected_price = context.expected_price
        if expected_price is None or expected_price <= 0:
            return None
        slippage_bps = self._estimate_slippage(allocation, context.liquidity_usd)
        liquidity_score = min(context.liquidity_usd / max(allocation * 4.0, 1.0), 1.0)
        depth_score = min(context.liquidity_usd / 75_000.0, 1.0)
        volatility_penalty = self._volatility_penalty(request.risk)
        expected_fees_usd = allocation * (request.pool.fee_bps / 10_000)
        notes: List[str] = []
        if request.pool.metadata.get("bin_step"):
            notes.append(f"bin_step={int(request.pool.metadata['bin_step'])}")
        virtual_base = request.pool.metadata.get("base_virtual")
        virtual_quote = request.pool.metadata.get("quote_virtual")
        if virtual_base:
            notes.append(f"virtual_base={virtual_base:.0f}")
        if virtual_quote:
            notes.append(f"virtual_quote={virtual_quote:.0f}")
        extras = dict(request.pool.metadata)
        extras.update(
            {
                "pool_liquidity_usd": context.liquidity_usd,
                "pool_base_value_usd": self._pool_value_usd(
                    request.pool.base_liquidity, expected_price
                ),
                "pool_quote_value_usd": self._pool_quote_value(
                    request.pool.quote_liquidity, request.quote_price_hint
                ),
            }
        )
        return VenueQuote(
            venue=self.name,
            pool_address=request.pool.address,
            base_mint=request.pool.base_mint,
            quote_mint=request.pool.quote_mint,
            allocation_usd=allocation,
            pool_liquidity_usd=context.liquidity_usd,
            expected_price=expected_price,
            expected_slippage_bps=slippage_bps,
            liquidity_score=liquidity_score,
            depth_score=depth_score,
            volatility_penalty=volatility_penalty,
            fee_bps=request.pool.fee_bps,
            rebate_bps=0.0,
            expected_fees_usd=expected_fees_usd,
            base_contribution_lamports=0,
            quote_contribution_lamports=0,
            notes=notes,
            extras=extras,
        )

    def build_plan(self, request: QuoteRequest, quote: VenueQuote, owner: Pubkey) -> TransactionPlan:
        return self._builder.build_plan(request, quote, owner)

    def _quote_exit(self, request: QuoteRequest) -> Optional[VenueQuote]:
        position = request.position
        if position is None or position.base_quantity <= 0:
            return None
        if not position.position_address and request.mode != AppMode.DRY_RUN:
            return None
        base_decimals = request.decision.token_decimals or request.decision.token.decimals or 0
        price = request.base_price_hint or position.last_mark_price or position.entry_price
        allocation = max(request.allocation_usd, position.base_quantity * (price or 0.0))
        base_lamports = int(max(position.base_quantity, 0.0) * (10**int(base_decimals)))
        return VenueQuote(
            venue=self.name,
            pool_address=request.pool.address,
            base_mint=request.pool.base_mint,
            quote_mint=request.pool.quote_mint,
            allocation_usd=allocation,
            pool_liquidity_usd=request.pool.tvl_usd or allocation,
            expected_price=price or 0.0,
            expected_slippage_bps=0.0,
            liquidity_score=1.0,
            depth_score=1.0,
            volatility_penalty=0.0,
            fee_bps=request.pool.fee_bps,
            rebate_bps=0.0,
            expected_fees_usd=0.0,
            base_contribution_lamports=base_lamports,
            quote_contribution_lamports=0,
            notes=["exit"],
            extras={
                "exit": 1.0,
                "position_address": position.position_address,
            },
        )

    def _derive_context(
        self,
        pool: PoolContext,
        base_price_hint: Optional[float],
        quote_price_hint: Optional[float],
    ) -> Optional[_DlmmContext]:
        price = pool.price_usd or base_price_hint
        if price is None and pool.quote_mint in STABLECOIN_MINTS:
            price = quote_price_hint or 1.0
        if price is None:
            return None
        liquidity_usd = pool.tvl_usd
        if liquidity_usd is None:
            liquidity_usd = self._pool_value_usd(pool.base_liquidity, price) + self._pool_quote_value(
                pool.quote_liquidity, quote_price_hint
            )
        return _DlmmContext(pool=_snapshot_from_context(pool), liquidity_usd=liquidity_usd, expected_price=price)

    def _estimate_slippage(self, allocation: float, liquidity_usd: float) -> float:
        if liquidity_usd <= 0:
            return 10_000.0
        ratio = min(allocation / liquidity_usd, 1.0)
        return min(ratio * 10_000.0, 5_000.0)

    def _volatility_penalty(self, risk: Optional[TokenRiskMetrics]) -> float:
        if risk is None:
            return 0.0
        return min(max(risk.volatility_score / 5.0, 0.0), 1.0)

    def _pool_value_usd(self, base_liquidity: float, price: Optional[float]) -> float:
        if price is None:
            return 0.0
        return max(base_liquidity, 0.0) * max(price, 0.0)

    def _pool_quote_value(self, quote_liquidity: float, quote_price: Optional[float]) -> float:
        price = quote_price or 1.0
        return max(quote_liquidity, 0.0) * max(price, 0.0)


def _snapshot_from_context(pool: PoolContext) -> DlmmPoolSnapshot:
    return DlmmPoolSnapshot(
        address=pool.address,
        base_token_mint=pool.base_mint,
        quote_token_mint=pool.quote_mint,
        base_token_amount=pool.base_liquidity,
        quote_token_amount=pool.quote_liquidity,
        fee_bps=pool.fee_bps,
        bin_step=int(pool.metadata.get("bin_step", 0)),
        base_virtual_liquidity=pool.metadata.get("base_virtual"),
        quote_virtual_liquidity=pool.metadata.get("quote_virtual"),
        tvl_usd=pool.tvl_usd,
        volume_24h_usd=pool.volume_24h_usd,
        price_usd=pool.price_usd,
        is_active=pool.is_active,
    )
