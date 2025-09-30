"""Fallback implementations for optional Solana client dependencies.

These light-weight stand-ins provide the minimum surface area exercised by the
test-suite when the real :mod:`solana` Python package is unavailable.  The
runtime trading system should install the genuine dependency; however, keeping
stubs locally ensures unit tests and offline analysis can still execute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence


@dataclass(slots=True)
class AccountMeta:
    """Simplified representation of an account metadata entry."""

    pubkey: Any
    is_signer: bool
    is_writable: bool


@dataclass(slots=True)
class Instruction:
    """Minimal transaction instruction container."""

    program_id: Any
    data: bytes
    accounts: Sequence[AccountMeta]


@dataclass(slots=True)
class TxOpts:
    """Transaction options stub."""
    skip_preflight: bool = False
    preflight_commitment: str = "confirmed"


@dataclass(slots=True)
class Pubkey:
    """Public key stub."""
    value: str
    
    @classmethod
    def from_string(cls, s: str) -> "Pubkey":
        return cls(value=s)


@dataclass(slots=True)
class Signature:
    """Signature stub."""
    value: str


class Transaction:
    """Transaction stub that collects instructions for inspection."""

    def __init__(self) -> None:
        self.instructions: List[Instruction] = []

    def add(self, *instructions: Iterable[Instruction]) -> "Transaction":
        for instruction in instructions:
            if isinstance(instruction, Iterable) and not isinstance(instruction, Instruction):
                for nested in instruction:
                    self.add(nested)
            else:
                self.instructions.append(instruction)  # type: ignore[arg-type]
        return self


class Client:
    """Very small mock of :class:`solana.rpc.api.Client`."""

    def __init__(self, endpoint: str, timeout: float | None = None) -> None:
        self.endpoint = endpoint
        self.timeout = timeout

    # The real client exposes several RPC helpers; the tests only rely on a
    # subset being callable.  The implementations below provide deterministic,
    # inert responses that mimic the structure returned by the official
    # Solana client so higher level ingestion code can continue to operate
    # without the optional dependency installed.
    def get_account_info(self, pubkey, *args, **kwargs) -> dict:
        return {"result": {"value": None}}

    def get_balance(self, pubkey) -> dict:
        return {"result": {"value": 0}}

    def get_health(self) -> dict:
        return {"result": "ok"}

    def get_token_supply(self, mint, *_, **__) -> dict:
        return {
            "result": {
                "value": {
                    "amount": "0",
                    "decimals": 6,
                    "uiAmount": 0.0,
                    "uiAmountString": "0",
                }
            }
        }

    def get_token_largest_accounts(self, mint, *_, **__) -> dict:
        return {"result": {"value": []}}

    def get_signatures_for_address(self, address, *_, **__) -> dict:
        return {"result": []}

    def get_program_accounts(self, program_id, *_, **__) -> dict:
        return {"result": []}

    def send_transaction(self, transaction, *signers) -> dict:
        return {"result": "mock-tx", "endpoint": self.endpoint}

    def simulate_transaction(self, transaction, opts=None) -> dict:
        """Mock simulation that returns success."""
        class MockResult:
            def __init__(self):
                self.err = None
                self.units_consumed = 150000
                self.logs = ["Mock simulation log"]
                self.accounts = []
        
        class MockResponse:
            def __init__(self):
                self.value = MockResult()
        
        return MockResponse()

    def get_recent_prioritization_fees(self, accounts=None) -> dict:
        """Mock recent prioritization fees."""
        class MockFeeInfo:
            def __init__(self, fee):
                self.prioritization_fee = fee
        
        class MockResponse:
            def __init__(self):
                self.value = [
                    MockFeeInfo(5000),
                    MockFeeInfo(7500),
                    MockFeeInfo(10000),
                    MockFeeInfo(15000),
                    MockFeeInfo(20000)
                ]
        
        return MockResponse()


__all__ = ["AccountMeta", "Instruction", "Transaction", "Client", "TxOpts", "Pubkey", "Signature"]
