"""RocketScan API client for token holder cohort metrics."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Dict, Iterable, Optional

import requests
from cachetools import TTLCache

from ..config.settings import DataSourceConfig, get_app_config
from ..datalake.schemas import RocketscanMetrics
from ..monitoring.logger import get_logger


class RocketscanClient:
    """Fetches token cohort distributions from rocketscan.fun."""

    def __init__(
        self,
        config: Optional[DataSourceConfig] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        app_config = get_app_config()
        self._config = config or app_config.data_sources
        self._enabled = getattr(self._config, "enable_rocketscan", False)
        self._base_url = str(getattr(self._config, "rocketscan_base_url", "https://rocketscan.fun/api")).rstrip("/")
        self._timeout = float(getattr(self._config, "rocketscan_timeout", 8.0))
        self._max_results = int(getattr(self._config, "rocketscan_max_results", 75))
        cache_ttl = int(getattr(self._config, "rocketscan_cache_ttl_seconds", 180))
        self._cache: TTLCache[str, Optional[RocketscanMetrics]] = TTLCache(maxsize=512, ttl=max(cache_ttl, 0))
        self._cache_lock = Lock()
        self._max_workers = max(1, int(getattr(self._config, "rocketscan_max_workers", 8)))
        self._max_age_minutes = int(getattr(self._config, "rocketscan_max_age_minutes", 60))
        self._session = session or requests.Session()
        self._logger = get_logger(__name__)

    def fetch_metrics(self, mint: str) -> Optional[RocketscanMetrics]:
        """Return cached or freshly-fetched holder cohort metrics for a mint."""

        if not self._enabled or not mint:
            return None
        cached = self._get_cached(mint)
        if cached is not None or mint in self._cache:
            return cached

        url = f"{self._base_url}/tokens"
        params = {"mint": mint, "limit": self._max_results}
        headers = {"User-Agent": "solana-liquidity-bot/1.0"}
        try:
            response = self._session.get(url, params=params, headers=headers, timeout=self._timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:  # pragma: no cover - network safety
            self._logger.debug("RocketScan request for %s failed: %s", mint, exc)
            self._set_cached(mint, None)
            return None
        except ValueError as exc:  # pragma: no cover - bad payloads
            self._logger.debug("RocketScan invalid JSON for %s: %s", mint, exc)
            self._set_cached(mint, None)
            return None

        pools = payload.get("pools")
        if not isinstance(pools, list):
            self._set_cached(mint, None)
            return None

        for pool in pools:
            base = pool.get("baseAsset") if isinstance(pool, dict) else None
            if not isinstance(base, dict):
                continue
            if str(base.get("id")) != mint:
                continue
            audit = base.get("audit") if isinstance(base.get("audit"), dict) else {}
            metrics = RocketscanMetrics(
                dev_balance_pct=self._extract_float(audit, "devBalancePercentage", "dev_balance_percentage"),
                snipers_pct=self._extract_float(
                    audit,
                    "snipersHoldingPercentage",
                    "sniperHoldingPercentage",
                    "snipersPercentage",
                ),
                insiders_pct=self._extract_float(
                    audit,
                    "insidersHoldingPercentage",
                    "insiderHoldingPercentage",
                    "insidersPercentage",
                ),
                bundlers_pct=self._extract_float(
                    audit,
                    "bundlersHoldingPercentage",
                    "bundlerHoldingPercentage",
                    "bundleHoldingPercentage",
                ),
                top_holders_pct=self._extract_float(audit, "topHoldersPercentage"),
            )
            self._set_cached(mint, metrics)
            return metrics

        self._set_cached(mint, None)
        return None

    def fetch_metrics_bulk(
        self,
        mints: Iterable[str],
    ) -> Dict[str, Optional[RocketscanMetrics]]:
        """Fetch cohort metrics for a batch of mints concurrently."""

        results: Dict[str, Optional[RocketscanMetrics]] = {}
        if not self._enabled:
            return {mint: None for mint in mints}

        to_fetch: list[str] = []
        for mint in mints:
            cached = self._get_cached(mint)
            if mint in self._cache or cached is not None:
                results[mint] = cached
            else:
                to_fetch.append(mint)

        if not to_fetch:
            return results

        max_workers = min(self._max_workers, len(to_fetch))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(self.fetch_metrics, mint): mint for mint in to_fetch}
            for future in as_completed(future_map):
                mint = future_map[future]
                try:
                    results[mint] = future.result()
                except Exception as exc:  # pragma: no cover - defensive
                    self._logger.debug("RocketScan bulk fetch failed for %s: %s", mint, exc)
                    self._set_cached(mint, None)
                    results[mint] = None

        return results

    def _get_cached(self, mint: str) -> Optional[RocketscanMetrics]:
        with self._cache_lock:
            return self._cache.get(mint)

    def _set_cached(self, mint: str, value: Optional[RocketscanMetrics]) -> None:
        with self._cache_lock:
            self._cache[mint] = value

    @staticmethod
    def _extract_float(audit: dict, *keys: str) -> Optional[float]:
        for key in keys:
            value = audit.get(key)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                try:
                    return float(value)
                except ValueError:
                    continue
        return None


__all__ = ["RocketscanClient", "RocketscanMetrics"]
