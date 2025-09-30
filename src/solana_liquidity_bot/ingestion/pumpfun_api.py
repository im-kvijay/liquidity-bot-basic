"""PumpFun API client for tracking newly launched tokens."""

from __future__ import annotations

from typing import Iterable, Optional

import requests
from cachetools import TTLCache
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential

from ..config.settings import DataSourceConfig, get_app_config
from ..datalake.schemas import TokenMetadata

DEFAULT_HEADERS = {"User-Agent": "solana-liquidity-bot/1.0"}


class PumpFunClient:
    """Small helper that fetches trending and new tokens from PumpFun."""

    def __init__(
        self,
        config: Optional[DataSourceConfig] = None,
        cache_ttl: int = 60,
        session: Optional[requests.Session] = None,
    ) -> None:
        self._config = config or get_app_config().data_sources
        self._cache = TTLCache(maxsize=256, ttl=cache_ttl)
        self._session = session or requests.Session()

    @retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3))
    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        url = f"{self._config.pumpfun_base_url}{path}"
        headers = dict(DEFAULT_HEADERS)
        if self._config.pumpfun_api_key:
            headers["Authorization"] = f"Bearer {self._config.pumpfun_api_key}"
        response = self._session.get(
            url,
            params=params,
            headers=headers,
            timeout=self._config.http_timeout,
        )
        response.raise_for_status()
        return response.json()

    def _normalize(self, payload: dict) -> Iterable[TokenMetadata]:
        for item in payload.get("tokens", []):
            yield TokenMetadata(
                mint_address=item.get("mint", ""),
                symbol=item.get("symbol", ""),
                name=item.get("name", item.get("symbol", "Unnamed")),
                creator=item.get("creator"),
                project_url=item.get("website"),
                decimals=item.get("decimals", 9),
                sources=["pumpfun"],
            )

    def list_new_tokens(self, limit: int = 20) -> list[TokenMetadata]:
        cache_key = f"new::{limit}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            payload = self._get("/v1/tokens/new", params={"limit": limit})
        except (RetryError, requests.HTTPError):
            return []

        results = [token for token in self._normalize(payload) if token.mint_address]
        self._cache[cache_key] = results
        return results


__all__ = ["PumpFunClient"]
