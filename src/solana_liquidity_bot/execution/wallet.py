"""Wallet helpers for managing Solana keypairs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import base58
import json
from solders.keypair import Keypair
from solders.pubkey import Pubkey

from ..config.settings import WalletConfig, get_app_config


@dataclass(slots=True)
class Wallet:
    """Wrapper around a Solana keypair."""

    keypair: Keypair

    @property
    def public_key(self) -> Pubkey:
        return self.keypair.pubkey()

    def secret(self) -> bytes:
        return bytes(self.keypair.secret())


def load_wallet(config: Optional[WalletConfig] = None) -> Wallet:
    cfg = config or get_app_config().wallet
    secret_key: Optional[bytes] = None
    if cfg.private_key:
        secret_key = base58.b58decode(cfg.private_key)
    elif cfg.keypair_path:
        path = Path(cfg.keypair_path).expanduser()
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            secret_key = bytes(data)
    if secret_key is None:
        raise ValueError(
            "Wallet configuration error - please check your environment variables"
        )

    keypair = Keypair.from_bytes(secret_key)
    return Wallet(keypair=keypair)


__all__ = ["Wallet", "load_wallet"]
