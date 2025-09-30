"""Client for fetching metadata from the Solana token list service."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import requests
from cachetools import TTLCache

from ..config.settings import DataSourceConfig, get_app_config
from ..monitoring.logger import get_logger


@dataclass(slots=True)
class TokenListEntry:
    """Normalized representation of a Solana token list entry."""

    mint: str
    name: str
    symbol: str
    decimals: int
    tags: List[str]
    verified: bool
    logo_uri: Optional[str] = None
    extensions: Dict[str, Any] = field(default_factory=dict)


class SolanaTokenListClient:
    """Caches and serves metadata from the official Solana token registry."""

    def __init__(
        self,
        config: Optional[DataSourceConfig] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self._config = config or get_app_config().data_sources
        self._session = session or requests.Session()
        self._cache: TTLCache[str, Dict[str, TokenListEntry]] = TTLCache(
            maxsize=1, ttl=self._config.cache_ttl_seconds
        )
        self._logger = get_logger(__name__)

    def _load_registry(self) -> Dict[str, TokenListEntry]:
        if not getattr(self._config, "enable_solana_token_list", True):
            self._cache["registry"] = {}
            return {}
        if "registry" in self._cache:
            return self._cache["registry"]
        try:
            response = self._session.get(
                self._config.solana_token_list_url,
                timeout=self._config.http_timeout,
                headers={"User-Agent": "solana-liquidity-bot/1.0"},
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:  # pragma: no cover - network path
            self._logger.warning("Failed to fetch Solana token list: %s", exc)
            return {}
        tokens = payload.get("tokens") if isinstance(payload, dict) else None
        if not isinstance(tokens, list):
            return {}
        registry: Dict[str, TokenListEntry] = {}
        for item in tokens:
            mint = item.get("address") or item.get("mintAddress")
            if not mint:
                continue
            try:
                entry = TokenListEntry(
                    mint=str(mint),
                    name=str(item.get("name", item.get("symbol", ""))),
                    symbol=str(item.get("symbol", "")),
                    decimals=int(item.get("decimals", 9)),
                    tags=[str(tag) for tag in item.get("tags", []) if tag],
                    verified=bool(item.get("extensions", {}).get("verified"))
                    if isinstance(item.get("extensions"), dict)
                    else bool(item.get("verified")),
                    logo_uri=item.get("logoURI"),
                    extensions={
                        str(key): value
                        for key, value in item.get("extensions", {}).items()
                        if isinstance(key, str)
                    }
                )
            except (TypeError, ValueError):  # pragma: no cover - defensive
                continue
            registry[entry.mint] = entry
        self._cache["registry"] = registry
        return registry

    def get(self, mint: str) -> Optional[TokenListEntry]:
        return self._load_registry().get(mint)

    def lookup(self, mints: Iterable[str]) -> Dict[str, TokenListEntry]:
        registry = self._load_registry()
        return {mint: registry[mint] for mint in mints if mint in registry}

    def sample(
        self,
        limit: int,
        *,
        verified_only: bool = False,
        preferred_tags: Optional[Sequence[str]] = None,
    ) -> List[TokenListEntry]:
        """Return a curated list of registry entries for seeding discovery."""

        registry = self._load_registry()
        if not registry or limit <= 0:
            return []

        normalized_tags: List[str] = []
        if preferred_tags:
            normalized_tags = [tag.lower() for tag in preferred_tags]

        def _priority(entry: TokenListEntry) -> tuple[int, int, int, str]:
            is_verified = 0 if entry.verified else 1
            has_preferred_tag = 1
            if normalized_tags and any(tag.lower() in normalized_tags for tag in entry.tags):
                has_preferred_tag = 0
            symbol_len = len(entry.symbol) if entry.symbol else 99
            symbol_key = entry.symbol.lower() if entry.symbol else entry.name.lower()
            return (is_verified, has_preferred_tag, symbol_len, symbol_key)

        candidates = [
            entry
            for entry in registry.values()
            if not verified_only or entry.verified
        ]
        candidates.sort(key=_priority)
        return candidates[:limit]


__all__ = ["SolanaTokenListClient", "TokenListEntry"]
