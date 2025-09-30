"""Helpers for managing manual allow/deny decisions for tokens."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Iterable, Optional

from ..config.settings import get_app_config
from ..datalake.schemas import TokenControlDecision
from ..datalake.storage import SQLiteStorage
from ..monitoring.logger import get_logger


MANUAL_SOURCE = "manual"
AUTO_SOURCE = "auto"
VALID_STATUSES = {"allow", "deny", "pause"}


class TokenControlService:
    """Service class exposing allow/deny/paused controls."""

    def __init__(self, storage: SQLiteStorage) -> None:
        self._storage = storage
        self._logger = get_logger(__name__)

    def set_status(self, mint_address: str, status: str, reason: str, source: str) -> TokenControlDecision:
        status_lower = status.lower()
        if status_lower not in VALID_STATUSES:
            raise ValueError(f"Unsupported status '{status}'. Valid options: {sorted(VALID_STATUSES)}")
        decision = TokenControlDecision(
            mint_address=mint_address,
            status=status_lower,
            reason=reason,
            source=source,
            updated_at=datetime.now(timezone.utc),
        )
        self._storage.upsert_token_control(decision)
        self._logger.info("Token %s marked %s (%s)", mint_address, status_lower, reason)
        return decision

    def allow(self, mint_address: str, reason: str, source: str = MANUAL_SOURCE) -> TokenControlDecision:
        return self.set_status(mint_address, "allow", reason, source)

    def deny(self, mint_address: str, reason: str, source: str = MANUAL_SOURCE) -> TokenControlDecision:
        return self.set_status(mint_address, "deny", reason, source)

    def pause(self, mint_address: str, reason: str, source: str = MANUAL_SOURCE) -> TokenControlDecision:
        return self.set_status(mint_address, "pause", reason, source)

    def get(self, mint_address: str) -> Optional[TokenControlDecision]:
        return self._storage.get_token_control(mint_address)

    def list(self) -> Iterable[TokenControlDecision]:
        return self._storage.list_token_controls()

    def clear(self, mint_address: str) -> None:
        self._storage.delete_token_control(mint_address)
        self._logger.info("Cleared token control for %s", mint_address)


def _cli() -> None:  # pragma: no cover - CLI utility
    parser = argparse.ArgumentParser(description="Manage token allow/deny decisions")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("mint", help="Token mint address")
    common.add_argument("--reason", required=True, help="Human readable reason for the decision")

    allow_cmd = sub.add_parser("allow", parents=[common], help="Mark a token as allowed")
    deny_cmd = sub.add_parser("deny", parents=[common], help="Mark a token as denied")
    pause_cmd = sub.add_parser("pause", parents=[common], help="Temporarily pause a token")

    list_cmd = sub.add_parser("list", help="List stored token controls")
    clear_cmd = sub.add_parser("clear", parents=[common], help="Remove an existing control entry")

    args = parser.parse_args()
    config = get_app_config()
    storage = SQLiteStorage(config.storage.database_path)
    service = TokenControlService(storage)

    if args.command == "allow":
        service.allow(args.mint, args.reason)
    elif args.command == "deny":
        service.deny(args.mint, args.reason)
    elif args.command == "pause":
        service.pause(args.mint, args.reason)
    elif args.command == "clear":
        service.clear(args.mint)
    elif args.command == "list":
        for decision in service.list():
            print(
                f"{decision.mint_address}: {decision.status} ({decision.reason}) "
                f"[{decision.source}] updated {decision.updated_at.isoformat()}"
            )


if __name__ == "__main__":  # pragma: no cover - CLI utility
    _cli()


__all__ = [
    "AUTO_SOURCE",
    "MANUAL_SOURCE",
    "TokenControlService",
    "VALID_STATUSES",
]

