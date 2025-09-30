"""Client for interacting with the Axiom API to discover early-stage tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import requests
from cachetools import TTLCache
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

from ..config.settings import DataSourceConfig, get_app_config
from ..datalake.schemas import TokenMetadata

DEFAULT_HEADERS = {"User-Agent": "solana-liquidity-bot/1.0"}


@dataclass(slots=True)
class AxiomTokenResult:
    """Parsed response element from the Axiom discovery API."""

    mint_address: str
    symbol: str
    name: str
    creator: Optional[str]
    project_url: Optional[str]


class AxiomClient:
    """Thin wrapper around the Axiom REST API with caching and retries."""

    def __init__(
        self,
        config: Optional[DataSourceConfig] = None,
        cache_ttl: int = 300,
        session: Optional[requests.Session] = None,
    ) -> None:
        self._config = config or get_app_config().data_sources
        self._cache = TTLCache(maxsize=256, ttl=cache_ttl)
        self._session = session or requests.Session()

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3))
    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        url = f"{self._config.axiom_base_url}{path}"
        headers = dict(DEFAULT_HEADERS)
        if self._config.axiom_api_key:
            headers["Authorization"] = f"Bearer {self._config.axiom_api_key}"
        response = self._session.get(
            url,
            params=params,
            headers=headers,
            timeout=self._config.http_timeout,
        )
        response.raise_for_status()
        return response.json()

    def _normalize(self, payload: dict) -> Iterable[AxiomTokenResult]:
        for item in payload.get("tokens", []):
            yield AxiomTokenResult(
                mint_address=item.get("mintAddress", ""),
                symbol=item.get("symbol", ""),
                name=item.get("name", "Unnamed"),
                creator=item.get("creator"),
                project_url=item.get("website"),
            )

    def list_recent_tokens(self, limit: int = 20) -> list[TokenMetadata]:
        if "recent" in self._cache:
            return self._cache["recent"]

        try:
            payload = self._get("/v1/tokens/recent", params={"limit": limit})
        except (RetryError, requests.HTTPError):
            return []

        results: list[TokenMetadata] = []
        for item in self._normalize(payload):
            if not item.mint_address:
                continue
            social_handles = self._extract_socials(payload, item.mint_address)
            metadata = TokenMetadata(
                mint_address=item.mint_address,
                symbol=item.symbol,
                name=item.name,
                creator=item.creator,
                project_url=item.project_url,
                social_handles=social_handles,
                sources=["axiom"],
            )
            results.append(metadata)

        self._cache["recent"] = results
        return results

    def _extract_socials(self, payload: dict, mint_address: str) -> list[str]:  # pragma: no cover - schema dependent
        socials: list[str] = []
        try:
            social_data = payload.get("socials") or {}
            if isinstance(social_data, dict):
                entries = social_data.get(mint_address)
                if isinstance(entries, dict):
                    socials.extend(str(value) for value in entries.values() if value)
                elif isinstance(entries, list):
                    socials.extend(str(value) for value in entries if value)
            elif isinstance(social_data, list):
                socials.extend(str(value) for value in social_data if value)
        except Exception:  # noqa: BLE001
            return socials
        return socials


__all__ = ["AxiomClient"]
