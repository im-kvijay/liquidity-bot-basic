"""Builders that translate strategy decisions into Solana transactions."""

from __future__ import annotations

import base64
from typing import Iterable, Optional, Tuple

import base58
from solders.keypair import Keypair
from solders.pubkey import Pubkey

try:  # pragma: no cover - exercised when the optional dependency is installed
    from solana.rpc.api import Client
except ModuleNotFoundError:  # pragma: no cover - fallback for test environments
    from .solana_compat import Client

try:  # pragma: no cover - exercised when the optional dependency is installed
    from solana.transaction import AccountMeta, Instruction, Transaction
except ModuleNotFoundError:  # pragma: no cover - fallback for test environments
    from .solana_compat import AccountMeta, Instruction, Transaction

try:  # pragma: no cover - optional dependency
    from spl.token._layouts import MINT_LAYOUT
except ModuleNotFoundError:  # pragma: no cover - fallback decoder
    from types import SimpleNamespace

    class _DefaultMintLayout:
        def parse(self, _data: bytes) -> SimpleNamespace:  # type: ignore[override]
            return SimpleNamespace(
                decimals=6,
                mintAuthorityOption=0,
                freezeAuthorityOption=0,
                mintAuthority=b"",
                freezeAuthority=b"",
            )

    MINT_LAYOUT = _DefaultMintLayout()

from ..config.settings import DammVenueConfig, DlmmVenueConfig, RPCConfig, get_app_config
from ..ingestion.pricing import PriceOracle
from ..monitoring.logger import get_logger
from ..utils.constants import STABLECOIN_MINTS
from .node_bridge import NodeBridge, NodeBridgeError
from .venues.base import QuoteRequest, TransactionPlan, VenueQuote


class BaseLiquidityBuilder:
    """Shared utilities for constructing liquidity provision transactions."""

    def __init__(
        self,
        *,
        rpc_config: Optional[RPCConfig] = None,
        price_oracle: Optional[PriceOracle] = None,
        node_bridge: Optional[NodeBridge] = None,
    ) -> None:
        self._rpc_config = rpc_config or get_app_config().rpc
        self._rpc_client = Client(str(self._rpc_config.primary_url), timeout=self._rpc_config.request_timeout)
        self._price_oracle = price_oracle or PriceOracle()
        self._node_bridge = node_bridge or NodeBridge()
        self._decimals_cache: dict[str, int] = {}
        self._logger = get_logger(__name__)

    def _prepare_amounts(self, request: QuoteRequest, quote: VenueQuote) -> Tuple[int, int, int, int, float, float]:
        decision = request.decision
        base_decimals = self._resolve_decimals(decision.token.mint_address, decision.token_decimals)
        quote_decimals = self._resolve_decimals(quote.quote_mint, decision.quote_token_decimals)

        base_price = self._resolve_price(decision.token.mint_address, quote.expected_price or decision.price_usd)
        quote_price = self._resolve_quote_price(quote.quote_mint, request.quote_price_hint)

        base_value_hint = max(quote.extras.get("pool_base_value_usd", 0.0), 0.0)
        quote_value_hint = max(quote.extras.get("pool_quote_value_usd", 0.0), 0.0)

        base_amount, quote_amount = self._determine_raw_amounts(
            allocation=decision.allocation,
            base_price=base_price,
            quote_price=quote_price,
            base_decimals=base_decimals,
            quote_decimals=quote_decimals,
            base_value_hint=base_value_hint,
            quote_value_hint=quote_value_hint,
        )
        quote.base_contribution_lamports = base_amount
        quote.quote_contribution_lamports = quote_amount
        return base_amount, quote_amount, base_decimals, quote_decimals, base_price, quote_price

    def _resolve_decimals(self, mint: str, override: Optional[int]) -> int:
        if override is not None and override > 0:
            return override
        if mint in self._decimals_cache:
            return self._decimals_cache[mint]
        resp = self._rpc_client.get_account_info(Pubkey.from_string(mint))
        value = resp.get("result", {}).get("value")
        if not value:
            raise ValueError(f"Unable to fetch mint info for {mint}")
        data = base64.b64decode(value["data"][0])
        decimals = int(MINT_LAYOUT.parse(data).decimals)
        self._decimals_cache[mint] = decimals
        return decimals

    def _resolve_price(self, mint: str, supplied: Optional[float]) -> float:
        if supplied and supplied > 0:
            return supplied
        price = self._price_oracle.get_price(mint)
        if price is None or price <= 0:
            raise ValueError(f"Unable to resolve market price for {mint}")
        return price

    def _resolve_quote_price(self, mint: str, supplied: Optional[float]) -> float:
        if mint in STABLECOIN_MINTS:
            return 1.0
        if supplied and supplied > 0:
            return supplied
        price = self._price_oracle.get_price(mint)
        if price is None or price <= 0:
            raise ValueError(f"Unable to resolve quote token price for {mint}")
        return price

    def _determine_raw_amounts(
        self,
        *,
        allocation: float,
        base_price: float,
        quote_price: float,
        base_decimals: int,
        quote_decimals: int,
        base_value_hint: float,
        quote_value_hint: float,
    ) -> tuple[int, int]:
        capital = max(allocation, 0.0)
        if capital <= 0:
            return 0, 0
        total_hint = base_value_hint + quote_value_hint
        if total_hint <= 0:
            base_share = 0.5
            quote_share = 0.5
        else:
            base_share = min(max(base_value_hint / total_hint, 0.0), 1.0)
            quote_share = min(max(quote_value_hint / total_hint, 0.0), 1.0)
            normaliser = base_share + quote_share
            if normaliser > 1.0e-6:
                base_share /= normaliser
                quote_share /= normaliser
            else:
                base_share = quote_share = 0.5
        base_capital = capital * base_share
        quote_capital = capital * quote_share
        base_units = int((base_capital / max(base_price, 1e-9)) * (10**base_decimals))
        quote_units = int((quote_capital / max(quote_price, 1e-9)) * (10**quote_decimals))
        if base_units <= 0 and base_capital > 0:
            base_units = 1
        if quote_units <= 0 and quote_capital > 0:
            quote_units = 1
        return base_units, quote_units

    def _convert_instructions(self, entries: Iterable[dict]) -> list[Instruction]:
        instructions: list[Instruction] = []
        for entry in entries:
            program_id = Pubkey.from_string(entry["programId"])
            accounts = [
                AccountMeta(
                    pubkey=Pubkey.from_string(meta["pubkey"]),
                    is_signer=bool(meta["isSigner"]),
                    is_writable=bool(meta["isWritable"]),
                )
                for meta in entry.get("accounts", [])
            ]
            data = base64.b64decode(entry.get("data", ""))
            instructions.append(Instruction(program_id=program_id, data=data, accounts=accounts))
        return instructions


class DlmmTransactionBuilder(BaseLiquidityBuilder):
    """Constructs add-liquidity transactions using the Meteora DLMM helper."""

    def __init__(
        self,
        rpc_config: Optional[RPCConfig] = None,
        venue_config: Optional[DlmmVenueConfig] = None,
        price_oracle: Optional[PriceOracle] = None,
        node_bridge: Optional[NodeBridge] = None,
    ) -> None:
        super().__init__(rpc_config=rpc_config, price_oracle=price_oracle, node_bridge=node_bridge)
        app_config = get_app_config()
        self._config = venue_config or app_config.venues.dlmm

    def build_plan(self, request: QuoteRequest, quote: VenueQuote, owner: Pubkey) -> TransactionPlan:
        decision = request.decision
        if decision.action == "exit":
            return self._build_withdraw_plan(request, quote, owner)
        if decision.action != "enter":
            raise ValueError(f"Unsupported action for DLMM builder: {decision.action}")
        if not quote.pool_address:
            raise ValueError("Quote is missing a pool address")
        base_amount, quote_amount, *_ = self._prepare_amounts(request, quote)
        if base_amount <= 0 and quote_amount <= 0:
            raise ValueError("Capital allocation too small to build a deposit transaction")

        position_keypair = Keypair()
        payload = {
            "rpcUrl": str(self._rpc_config.primary_url),
            "poolAddress": quote.pool_address,
            "userPublicKey": str(owner),
            "positionPublicKey": str(position_keypair.pubkey()),
            "totalXAmount": str(base_amount),
            "totalYAmount": str(quote_amount),
            "strategyType": quote.extras.get("strategy_type", "spot"),
            "binSpan": int(quote.extras.get("bin_span", 2)),
            "slippage": int(quote.extras.get("slippage_bps", self._config.max_slippage_bps)),
        }
        if "min_bin_id" in quote.extras:
            payload["minBinId"] = int(quote.extras["min_bin_id"])
        if "max_bin_id" in quote.extras:
            payload["maxBinId"] = int(quote.extras["max_bin_id"])
        if "single_sided_x" in quote.extras:
            payload["singleSidedX"] = bool(quote.extras["single_sided_x"])

        self._logger.debug("Invoking DLMM Node bridge for pool %s", quote.pool_address)
        try:
            response = self._node_bridge.run("build_dlmm_transaction.mjs", payload)
        except NodeBridgeError as exc:  # pragma: no cover - network path
            raise RuntimeError(f"Failed to build DLMM transaction: {exc}") from exc

        instructions = self._convert_instructions(response.get("instructions", []))
        transaction = Transaction()
        transaction.add(*instructions)
        plan = TransactionPlan(
            decision=decision,
            venue=quote.venue,
            quote=quote,
            transaction=transaction,
            signers=[position_keypair],
            position_keypair=position_keypair,
            position_address=str(position_keypair.pubkey()),
            position_secret=self.encode_position_secret(position_keypair),
        )
        return plan

    def encode_position_secret(self, keypair: Keypair) -> str:
        return base58.b58encode(bytes(keypair.secret())).decode("utf-8")

    def _build_withdraw_plan(
        self,
        request: QuoteRequest,
        quote: VenueQuote,
        owner: Pubkey,
    ) -> TransactionPlan:
        position = request.position
        if position is None or not position.position_address:
            raise ValueError("Exit requested without a persisted DLMM position reference")

        payload = {
            "rpcUrl": str(self._rpc_config.primary_url),
            "poolAddress": quote.pool_address,
            "userPublicKey": str(owner),
            "positionPublicKey": position.position_address,
            "bps": int(request.decision.metadata.get("exit_bps", 10_000)),
            "shouldClaimAndClose": True,
        }
        self._logger.debug(
            "Invoking DLMM withdraw helper for pool %s position %s",
            quote.pool_address,
            position.position_address,
        )
        try:
            response = self._node_bridge.run("build_dlmm_withdraw.mjs", payload)
        except NodeBridgeError as exc:  # pragma: no cover - network path
            raise RuntimeError(f"Failed to build DLMM withdraw transaction: {exc}") from exc

        instructions = self._convert_instructions(response.get("instructions", []))
        if not instructions:
            raise RuntimeError("DLMM withdraw helper returned no instructions")
        transaction = Transaction()
        transaction.add(*instructions)
        plan = TransactionPlan(
            decision=request.decision,
            venue=quote.venue,
            quote=quote,
            transaction=transaction,
            signers=[],
            position_address=position.position_address,
            position_secret=position.position_secret,
        )
        plan.lp_token_amount = position.lp_token_amount
        plan.lp_token_mint = position.lp_token_mint
        return plan


class DammTransactionBuilder(BaseLiquidityBuilder):
    """Constructs add-liquidity transactions for Meteora DAMM v2."""

    def __init__(
        self,
        rpc_config: Optional[RPCConfig] = None,
        venue_config: Optional[DammVenueConfig] = None,
        price_oracle: Optional[PriceOracle] = None,
        node_bridge: Optional[NodeBridge] = None,
    ) -> None:
        super().__init__(rpc_config=rpc_config, price_oracle=price_oracle, node_bridge=node_bridge)
        app_config = get_app_config()
        self._config = venue_config or app_config.venues.damm

    def build_plan(self, request: QuoteRequest, quote: VenueQuote, owner: Pubkey) -> TransactionPlan:
        decision = request.decision
        if decision.action == "exit":
            return self._build_withdraw_plan(request, quote, owner)
        if decision.action != "enter":
            raise ValueError(f"Unsupported action for DAMM builder: {decision.action}")
        if not quote.pool_address:
            raise ValueError("Quote is missing a pool address")
        base_amount, quote_amount, *_ = self._prepare_amounts(request, quote)
        if base_amount <= 0 and quote_amount <= 0:
            raise ValueError("Capital allocation too small to build a deposit transaction")

        payload = {
            "rpcUrl": str(self._rpc_config.primary_url),
            "poolAddress": quote.pool_address,
            "userPublicKey": str(owner),
            "tokenAAmount": str(base_amount),
            "tokenBAmount": str(quote_amount),
            "slippageBps": int(quote.extras.get("slippage_bps", self._config.max_slippage_bps)),
            "programId": self._config.program_id,
            "balanceDeposit": True,
        }
        self._logger.debug("Invoking DAMM Node bridge for pool %s", quote.pool_address)
        try:
            response = self._node_bridge.run("build_damm_transaction.mjs", payload)
        except NodeBridgeError as exc:  # pragma: no cover - network path
            raise RuntimeError(f"Failed to build DAMM transaction: {exc}") from exc

        instructions = self._convert_instructions(response.get("instructions", []))
        transaction = Transaction()
        transaction.add(*instructions)
        plan = TransactionPlan(
            decision=decision,
            venue=quote.venue,
            quote=quote,
            transaction=transaction,
            signers=[],
        )
        quote_payload = response.get("quote", {})
        pool_token_out = quote_payload.get("poolTokenAmountOut")
        if pool_token_out is not None:
            try:
                plan.lp_token_amount = int(pool_token_out)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                plan.lp_token_amount = None
        return plan

    def _build_withdraw_plan(
        self,
        request: QuoteRequest,
        quote: VenueQuote,
        owner: Pubkey,
    ) -> TransactionPlan:
        if not quote.pool_address:
            raise ValueError("Quote is missing a pool address")
        position = request.position
        decision = request.decision
        lp_amount = int(
            decision.metadata.get("lp_token_amount")
            if decision.metadata.get("lp_token_amount")
            else (position.lp_token_amount if position else 0)
        )
        if lp_amount <= 0:
            raise ValueError("Exit requested without LP token balance")

        payload = {
            "rpcUrl": str(self._rpc_config.primary_url),
            "poolAddress": quote.pool_address,
            "userPublicKey": str(owner),
            "programId": self._config.program_id,
            "poolTokenAmount": str(lp_amount),
            "slippageBps": int(decision.metadata.get("slippage_bps", self._config.max_slippage_bps)),
        }
        self._logger.debug("Invoking DAMM withdraw helper for pool %s", quote.pool_address)
        try:
            response = self._node_bridge.run("build_damm_withdraw.mjs", payload)
        except NodeBridgeError as exc:  # pragma: no cover - network path
            raise RuntimeError(f"Failed to build DAMM withdraw transaction: {exc}") from exc

        instructions = self._convert_instructions(response.get("instructions", []))
        transaction = Transaction()
        transaction.add(*instructions)
        plan = TransactionPlan(
            decision=decision,
            venue=quote.venue,
            quote=quote,
            transaction=transaction,
            signers=[],
        )
        plan.lp_token_amount = lp_amount
        if position:
            plan.position_address = position.position_address
            plan.position_secret = position.position_secret
            plan.lp_token_mint = position.lp_token_mint
        return plan
