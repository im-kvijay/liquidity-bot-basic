"""Helpers for fetching token price data from public APIs."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from time import monotonic, sleep
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple

import requests
from cachetools import TTLCache

from ..config.settings import DataSourceConfig, get_app_config
from ..monitoring.logger import get_logger
from .solana_token_list import SolanaTokenListClient


class PriceOracle:
    """HTTP-based price oracle with multi-provider support (Hermes Pyth + Jupiter)."""

    def __init__(
        self,
        config: Optional[DataSourceConfig] = None,
        session: Optional[requests.Session] = None,
        history_window_seconds: int = 3_600,
    ) -> None:
        self._config = config or get_app_config().data_sources
        self._session = session or requests.Session()
        self._fallback_price = getattr(self._config, "fallback_price_usd", None)
        self._fallback_logged = False
        self._cache: TTLCache[str, float] = TTLCache(
            maxsize=512, ttl=self._config.cache_ttl_seconds
        )
        self._logger = get_logger(__name__)
        self._token_list = (
            SolanaTokenListClient(self._config)
            if getattr(self._config, "enable_solana_token_list", True)
            else None
        )
        self._use_pyth_fallback = bool(
            getattr(self._config, "price_oracle_use_pyth_fallback", True)
        )
        self._jupiter_rate_limit = max(
            1,
            int(
                getattr(
                    self._config,
                    "price_oracle_jupiter_rate_limit_per_minute",
                    55,
                )
            ),
        )
        self._jupiter_request_times: Deque[float] = deque()
        self._mint_to_feed: Dict[str, str] = {}
        self._endpoints = self._build_endpoints()
        self._pyth_base = self._determine_endpoint(
            self._endpoints,
            "hermes.pyth.network",
            default="https://hermes.pyth.network",
        )
        self._pyth_updates_url = f"{self._pyth_base.rstrip('/')}" + "/v2/updates/price/latest"
        self._pyth_catalog_url = f"{self._pyth_base.rstrip('/')}" + "/v2/price_feeds"
        # Build a clean Jupiter price endpoint regardless of any extra path (e.g. '/v4')
        from urllib.parse import urlparse
        candidate = self._determine_endpoint(
            self._endpoints,
            "jup.ag",
            default=str(self._config.price_oracle_url),
            prefer_lite_api=True,
        )
        parsed = urlparse(candidate)
        host = parsed.netloc or (urlparse(str(self._config.price_oracle_url)).netloc or "lite-api.jup.ag")
        if not host or "jup.ag" not in host:
            host = "lite-api.jup.ag"
        scheme = parsed.scheme or "https"
        if scheme not in {"http", "https"}:
            scheme = "https"
        self._jupiter_price_url = f"{scheme}://{host}/price/v3"
        overrides = getattr(self._config, "price_feed_overrides", {})
        self._feed_overrides = {str(k).lower(): str(v) for k, v in overrides.items()}
        self._pyth_catalog = self._fetch_pyth_catalog()
        self._pyth_symbol_map = self._build_pyth_symbol_map(self._pyth_catalog)
        self._history_window = history_window_seconds
        self._price_history: Dict[str, Deque[Tuple[datetime, float, float]]] = defaultdict(deque)

    def _build_endpoints(self) -> Tuple[str, ...]:
        urls: Iterable[str]
        if getattr(self._config, "price_oracle_urls", None):
            urls = tuple(str(url) for url in self._config.price_oracle_urls)  # type: ignore[attr-defined]
        else:
            urls = (str(self._config.price_oracle_url),)
        return tuple(dict.fromkeys(urls))

    def _determine_endpoint(
        self,
        endpoints: Tuple[str, ...],
        keyword: str,
        *,
        default: str,
        prefer_lite_api: bool = False,
    ) -> str:
        """Pick endpoint containing keyword; optionally prefer lite-api.jup.ag."""
        if prefer_lite_api:
            for endpoint in endpoints:
                if "lite-api.jup.ag" in endpoint:
                    return endpoint
        for endpoint in endpoints:
            if keyword in endpoint:
                if "price.jup.ag" in endpoint:
                    return endpoint.replace("price.jup.ag", "lite-api.jup.ag")
                return endpoint
        return default

    def _fetch_pyth_catalog(self) -> List[dict]:
        try:
            response = self._session.get(
                self._pyth_catalog_url,
                timeout=self._config.http_timeout,
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                return data
        except (requests.RequestException, ValueError) as exc:  # pragma: no cover - network
            self._logger.warning("Failed to fetch Pyth catalog: %s", exc)
        return []

    def _build_pyth_symbol_map(self, catalog: List[dict]) -> Dict[str, str]:
        symbol_map: Dict[str, str] = {}
        for entry in catalog:
            if not isinstance(entry, dict):
                continue
            feed_id = entry.get("id")
            attributes = entry.get("attributes", {})
            if not feed_id or not isinstance(attributes, dict):
                continue
            quote = str(attributes.get("quote_currency", "")).upper()
            if quote not in {"USD", "USDC", "USDT"}:
                continue
            candidates: Set[str] = set()
            base = attributes.get("base")
            display = attributes.get("display_symbol")
            generic = attributes.get("generic_symbol")
            for value in (base, display, generic):
                if not isinstance(value, str):
                    continue
                token_part = value.split("/")[0]
                candidates.add(token_part)
            if isinstance(generic, str) and generic.upper().endswith(quote):
                candidates.add(generic[:-len(quote)])
            for candidate in candidates:
                normalized = self._normalize_symbol(candidate)
                if normalized:
                    symbol_map.setdefault(normalized, feed_id)
        return symbol_map

    def _normalize_symbol(self, symbol: str) -> Optional[str]:
        if not symbol:
            return None
        cleaned = "".join(ch for ch in symbol.upper() if ch.isalnum())
        if not cleaned:
            return None
        if cleaned.startswith("W") and len(cleaned) > 2:
            cleaned = cleaned[1:]
        if cleaned.startswith("ST") and len(cleaned) > 3:
            cleaned = cleaned[2:]
        if cleaned.endswith("USD") and len(cleaned) > 3:
            cleaned = cleaned[:-3]
        return cleaned or None

    def _collect_pyth_feed_ids(self, mints: Iterable[str]) -> Dict[str, List[str]]:
        feed_map: Dict[str, List[str]] = {}
        for mint in mints:
            mint_lower = mint.lower()
            feed_id = self._feed_overrides.get(mint_lower)
            if not feed_id:
                feed_id = self._mint_to_feed.get(mint_lower)
            if not feed_id:
                feed_id = self._resolve_feed_id_for_mint(mint)
            if feed_id:
                feed_map.setdefault(feed_id, []).append(mint)
        return feed_map

    def _resolve_feed_id_for_mint(self, mint: str) -> Optional[str]:
        mint_lower = mint.lower()
        entry = self._token_list.get(mint) if self._token_list is not None else None
        if not entry:
            return None
        symbol_candidates: Set[str] = set()
        if entry.symbol:
            symbol_candidates.add(entry.symbol)
        if entry.name:
            symbol_candidates.add(entry.name)
        if entry.extensions.get("coingeckoId"):
            symbol_candidates.add(str(entry.extensions["coingeckoId"]))
        for candidate in list(symbol_candidates):
            normalized = self._normalize_symbol(candidate)
            if not normalized:
                continue
            feed_id = self._pyth_symbol_map.get(normalized)
            if feed_id:
                self._mint_to_feed[mint_lower] = feed_id
                return feed_id
        return None

    def _fetch_from_pyth(self, feed_map: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
        results: Dict[str, Dict[str, float]] = {}
        feed_ids = list(feed_map.keys())
        if not feed_ids:
            return results
        chunk_size = 100
        for idx in range(0, len(feed_ids), chunk_size):
            chunk = feed_ids[idx : idx + chunk_size]
            try:
                params = [("ids[]", feed_id) for feed_id in chunk]
                params.extend([("encoding", "base64"), ("parsed", "true")])
                response = self._session.get(
                    self._pyth_updates_url,
                    params=params,
                    timeout=self._config.http_timeout,
                )
                response.raise_for_status()
                payload = response.json()
            except (requests.RequestException, ValueError) as exc:  # pragma: no cover - network
                self._logger.warning(
                    "Failed to fetch Pyth prices for chunk %s: %s", chunk, exc
                )
                continue

            parsed = payload.get("parsed") if isinstance(payload, dict) else None
            if not isinstance(parsed, list):
                continue
            feed_prices: Dict[str, Tuple[float, float]] = {}
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                feed_id = item.get("id")
                price_data = item.get("price", {})
                price_raw = price_data.get("price")
                expo = price_data.get("expo")
                conf = price_data.get("conf", 0)
                if feed_id is None or price_raw is None or expo is None:
                    continue
                try:
                    scale = 10 ** int(expo)
                    price_value = float(price_raw) * scale
                    confidence = float(conf) * scale if isinstance(conf, (int, float)) else 0.0
                except (TypeError, ValueError):
                    continue
                feed_prices[feed_id] = (price_value, confidence)
            for feed_id, mints in feed_map.items():
                if feed_id in feed_prices:
                    price_value, confidence = feed_prices[feed_id]
                    for mint in mints:
                        results[mint] = {"price": price_value, "confidence": confidence}
        return results

    def _fetch_from_jupiter(self, mints: Iterable[str]) -> Dict[str, Dict[str, float]]:
        if not mints:
            return {}

        unique_mints = list(dict.fromkeys(mints))
        aggregated: Dict[str, Dict[str, float]] = {}
        batch_size = 80  # keep URI well below the 8 KB limit enforced by lite-api.jup.ag

        for idx in range(0, len(unique_mints), batch_size):
            chunk = unique_mints[idx : idx + batch_size]
            if not chunk:
                continue
            self._enforce_jupiter_rate_limit(1)
            prices = self._fetch_jupiter_chunk(chunk)
            aggregated.update(prices)

        return aggregated

    def _fetch_jupiter_chunk(self, mints: List[str]) -> Dict[str, Dict[str, float]]:
        # Primary path: requests with proper Accept header
        try:
            response = self._session.get(
                self._jupiter_price_url,
                params={"ids": ",".join(mints)},
                headers={"Accept": "application/json"},
                timeout=self._config.http_timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            # Fallback path: use http.client as per Jupiter's example
            self._logger.warning(
                "Requests path failed for Jupiter: %s; attempting http.client fallback", exc
            )
            try:
                import http.client
                import json as _json
                from urllib.parse import urlencode, urlparse

                parsed = urlparse(self._jupiter_price_url)
                netloc = parsed.netloc or "lite-api.jup.ag"
                path = parsed.path or "/price/v3"
                query = urlencode({"ids": ",".join(mints)})
                full_path = f"{path}?{query}" if query else path

                conn = http.client.HTTPSConnection(netloc, timeout=self._config.http_timeout)
                conn.request("GET", full_path, headers={"Accept": "application/json"})
                res = conn.getresponse()
                raw = res.read()
                conn.close()
                payload = _json.loads(raw.decode("utf-8")) if raw else {}
            except Exception as exc2:  # pragma: no cover - network
                self._logger.warning(
                    "Jupiter http.client fallback failed for %s: %s", list(mints), exc2
                )
                return {}

        if isinstance(payload, dict) and "data" in payload:
            data = payload.get("data", {})
        else:
            data = payload

        prices: Dict[str, Dict[str, float]] = {}
        if not isinstance(data, dict):
            return prices
        for mint, value in data.items():
            if not isinstance(value, dict):
                continue
            price_field = value.get("price") or value.get("usdPrice")
            if price_field is None:
                continue
            try:
                prices[mint] = {
                    "price": float(price_field),
                    "confidence": float(value.get("confidence", 0.0)),
                }
            except (TypeError, ValueError):
                continue
        return prices

    def _request(self, mints: Iterable[str]) -> Dict[str, Dict[str, float]]:
        requested = tuple(dict.fromkeys(mints))
        if not requested:
            return {}

        prices: Dict[str, Dict[str, float]] = {}

        remaining = list(requested)
        if remaining:
            jupiter_prices = self._fetch_from_jupiter(remaining)
            prices.update(jupiter_prices)
            remaining = [mint for mint in remaining if mint not in prices]

        if remaining and self._use_pyth_fallback:
            feed_map = self._collect_pyth_feed_ids(remaining)
            if feed_map:
                pyth_prices = self._fetch_from_pyth(feed_map)
                prices.update(pyth_prices)
                remaining = [mint for mint in remaining if mint not in prices]

        if remaining and self._fallback_price is not None:
            if not self._fallback_logged:
                self._logger.warning(
                    "Falling back to static price %.4f USD for %d tokens (lite-api.jup.ag unavailable)",
                    self._fallback_price,
                    len(remaining),
                )
                self._fallback_logged = True
            for mint in remaining:
                prices[mint] = {"price": float(self._fallback_price), "confidence": 0.0}
        for mint, value in prices.items():
            if isinstance(value, dict):
                price_f = value.get("price")
                volume_hint = float(value.get("confidence", 0.0)) if "confidence" in value else 0.0
            else:
                price_f = value
                volume_hint = 0.0
            if price_f is None:
                continue
            self._record_price(mint, float(price_f), volume_hint)
        return prices

    def get_price(self, mint: str) -> Optional[float]:
        if mint in self._cache:
            return self._cache[mint]
        prices = self._request([mint])
        if mint in prices:
            value = prices[mint]
            price_value = value.get("price") if isinstance(value, dict) else value
            if price_value is not None:
                self._cache[mint] = float(price_value)
                return float(price_value)
        return None

    def get_prices(self, mints: Iterable[str]) -> Dict[str, float]:
        ordered_unique = list(dict.fromkeys(mints))
        if not ordered_unique:
            return {}
        missing = [mint for mint in ordered_unique if mint not in self._cache]
        if missing:
            prices = self._request(missing)
            for mint, value in prices.items():
                if isinstance(value, dict):
                    price_value = value.get("price")
                else:
                    price_value = value
                if price_value is not None:
                    self._cache[mint] = float(price_value)
        results: Dict[str, float] = {}
        for mint in ordered_unique:
            if mint in self._cache:
                results[mint] = self._cache[mint]
        return results

    def get_last_update(self, mint: str) -> Optional[datetime]:
        history = self._price_history.get(mint)
        if not history:
            return None
        return history[-1][0]

    def get_twap(self, mint: str, window_seconds: int) -> Optional[float]:
        return self._compute_weighted_average(mint, window_seconds, use_volume=False)

    def get_vwap(self, mint: str, window_seconds: int) -> Optional[float]:
        return self._compute_weighted_average(mint, window_seconds, use_volume=True)

    def get_mark_price(
        self,
        mint: str,
        *,
        source: str,
        window_seconds: Optional[int] = None,
        fallback: Optional[float] = None,
    ) -> Optional[float]:
        window = window_seconds or self._history_window
        if source == "twap":
            mark = self.get_twap(mint, window)
        elif source == "vwap":
            mark = self.get_vwap(mint, window)
        else:
            mark = self.get_price(mint)
        if mark is None:
            mark = fallback
        return mark

    def _record_price(self, mint: str, price: float, volume_hint: float) -> None:
        history = self._price_history[mint]
        history.append((datetime.now(timezone.utc), price, max(volume_hint, 0.0)))
        self._prune_history(history)

    def _prune_history(self, history: Deque[Tuple[datetime, float, float]]) -> None:
        if not history:
            return
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self._history_window)
        while history and history[0][0] < cutoff:
            history.popleft()

    def _compute_weighted_average(
        self, mint: str, window_seconds: int, *, use_volume: bool
    ) -> Optional[float]:
        history = self._price_history.get(mint)
        if not history:
            return None
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)
        weighted_price = 0.0
        total_weight = 0.0
        for timestamp, price, volume in reversed(history):
            if timestamp < cutoff:
                break
            weight = volume if use_volume and volume > 0 else 1.0
            weighted_price += price * weight
            total_weight += weight
        if total_weight <= 0:
            return None
        return weighted_price / total_weight

    def _enforce_jupiter_rate_limit(self, request_cost: int = 1) -> None:
        if self._jupiter_rate_limit <= 0:
            return
        window = 60.0
        queue = self._jupiter_request_times
        now = monotonic()
        while queue and now - queue[0] > window:
            queue.popleft()
        while len(queue) + request_cost > self._jupiter_rate_limit:
            oldest = queue[0]
            sleep_time = window - (now - oldest)
            if sleep_time > 0:
                sleep(min(sleep_time, 1.0))
            now = monotonic()
            while queue and now - queue[0] > window:
                queue.popleft()
        for _ in range(request_cost):
            queue.append(now)


__all__ = ["PriceOracle"]
