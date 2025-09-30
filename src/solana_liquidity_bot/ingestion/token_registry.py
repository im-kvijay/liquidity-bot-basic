"""Token registry aggregator with compliance and risk evaluation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Set

from ..config.settings import AppConfig, DataSourceConfig, TokenUniverseConfig, get_app_config
from ..datalake.schemas import (
    DammPoolSnapshot,
    DlmmPoolSnapshot,
    LiquidityEvent,
    TokenControlDecision,
    TokenMetadata,
    TokenOnChainStats,
    TokenRiskMetrics,
    TokenUniverseEntry,
)
from ..datalake.storage import SQLiteStorage
from ..monitoring.logger import get_logger
from ..utils.constants import SOL_MINT, STABLECOIN_MINTS
from .compliance import ComplianceEngine
from .axiom_api import AxiomClient
from .damm_api import DammClient
from .dlmm_api import DlmmClient
from .onchain import SolanaChainAnalyzer
from .pricing import PriceOracle
from .pumpfun_api import PumpFunClient
from .rocketscan_api import RocketscanClient, RocketscanMetrics
from .solana_token_list import SolanaTokenListClient, TokenListEntry
from .token_controls import AUTO_SOURCE, MANUAL_SOURCE


class TokenRegistryAggregator:
    """Aggregates token metadata, enforces compliance, and persists risk metrics."""

    def __init__(
        self,
        storage: SQLiteStorage,
        app_config: Optional[AppConfig] = None,
        axiom_client: Optional[AxiomClient] = None,
        pumpfun_client: Optional[PumpFunClient] = None,
        token_list_client: Optional[SolanaTokenListClient] = None,
        damm_client: Optional[DammClient] = None,
        dlmm_client: Optional[DlmmClient] = None,
        price_oracle: Optional[PriceOracle] = None,
        onchain_analyzer: Optional[SolanaChainAnalyzer] = None,
        rocketscan_client: Optional[RocketscanClient] = None,
    ) -> None:
        self._app_config = app_config or get_app_config()
        self._storage = storage
        data_config: DataSourceConfig = self._app_config.data_sources
        self._token_config: TokenUniverseConfig = self._app_config.token_universe
        self._axiom_client = axiom_client or AxiomClient(data_config)
        self._pumpfun_client = pumpfun_client or PumpFunClient(data_config)
        if token_list_client is not None:
            self._token_list_client = token_list_client
        elif getattr(data_config, "enable_solana_token_list", True):
            self._token_list_client = SolanaTokenListClient(data_config)
        else:
            self._token_list_client = None
        if damm_client is not None:
            self._damm_client = damm_client
        elif self._app_config.venues.damm.enabled:
            self._damm_client = DammClient(data_config, app_config=self._app_config)
        else:
            self._damm_client = None
        if dlmm_client is not None:
            self._dlmm_client = dlmm_client
        elif self._app_config.venues.dlmm.enabled:
            self._dlmm_client = DlmmClient(data_config, app_config=self._app_config)
        else:
            self._dlmm_client = None
        self._price_oracle = price_oracle or PriceOracle(data_config)
        self._onchain = onchain_analyzer or SolanaChainAnalyzer(self._app_config.rpc)
        self._compliance = ComplianceEngine(self._token_config)
        self._logger = get_logger(__name__)
        if rocketscan_client is not None:
            self._rocketscan_client = rocketscan_client
        elif getattr(data_config, "enable_rocketscan", False):
            self._rocketscan_client = RocketscanClient(data_config)
        else:
            self._rocketscan_client = None

    def build_universe(self, limit: int) -> List[TokenUniverseEntry]:
        batch_size = max(limit, self._token_config.discovery_batch_size)
        candidates = self._collect_candidates(batch_size)
        metadata: Dict[str, TokenListEntry] = {}
        if (
            self._token_list_client is not None
            and getattr(self._app_config.data_sources, "enable_solana_token_list", True)
        ):
            metadata = self._token_list_client.lookup(token.mint_address for token in candidates)
            if metadata:
                self._apply_token_list_metadata(candidates, metadata)
        stats = self._onchain.collect_stats(candidates)
        stats_by_mint = {stat.token.mint_address: stat for stat in stats}
        damm_pools = (
            self._damm_client.fetch_pools_for_tokens(candidates)
            if self._damm_client is not None
            else {}
        )
        if self._dlmm_client is not None:
            dlmm_targets = [
                token
                for token in candidates
                if token.mint_address not in damm_pools
            ]
            dlmm_pools = (
                self._dlmm_client.fetch_pools_for_tokens(dlmm_targets)
                if dlmm_targets
                else {}
            )
        else:
            dlmm_pools = {}
        prices = self._price_oracle.get_prices(token.mint_address for token in candidates)

        rocketscan_data: Dict[str, Optional[RocketscanMetrics]] = {}
        preexisting_metrics: Dict[str, RocketscanMetrics] = {}
        rocketscan_ttl = int(getattr(self._app_config.data_sources, "rocketscan_cache_ttl_seconds", 180))
        refresh_cutoff = datetime.now(timezone.utc) - timedelta(seconds=max(rocketscan_ttl, 60))
        existing_risk_metrics = self._storage.get_token_risk_metrics_bulk(
            token.mint_address for token in candidates
        )
        if self._rocketscan_client is not None and candidates:
            max_age_minutes = int(getattr(self._app_config.data_sources, "rocketscan_max_age_minutes", 60))
            now = datetime.now(timezone.utc)
            eligible: List[str] = []
            for token in candidates:
                stats_entry = stats_by_mint.get(token.mint_address)
                minted_at = stats_entry.minted_at if stats_entry and stats_entry.minted_at else None
                stored_risk = existing_risk_metrics.get(token.mint_address)
                if (
                    stored_risk
                    and stored_risk.last_updated >= refresh_cutoff
                    and (
                        stored_risk.dev_holding_pct is not None
                        or stored_risk.sniper_holding_pct is not None
                        or stored_risk.insider_holding_pct is not None
                        or stored_risk.bundler_holding_pct is not None
                    )
                ):
                    preexisting_metrics[token.mint_address] = RocketscanMetrics(
                        dev_balance_pct=stored_risk.dev_holding_pct,
                        snipers_pct=stored_risk.sniper_holding_pct,
                        insiders_pct=stored_risk.insider_holding_pct,
                        bundlers_pct=stored_risk.bundler_holding_pct,
                        top_holders_pct=stored_risk.top_holder_pct,
                    )
                    continue
                if minted_at is not None:
                    age_minutes = (now - minted_at).total_seconds() / 60
                    if age_minutes > max_age_minutes:
                        continue
                eligible.append(token.mint_address)
            if eligible:
                rocketscan_data = self._rocketscan_client.fetch_metrics_bulk(eligible)

        entries: List[TokenUniverseEntry] = []
        for token in candidates:
            token_stats = stats_by_mint.get(token.mint_address)
            all_damm_pools = damm_pools.get(token.mint_address, [])
            damm = [
                pool
                for pool in all_damm_pools
                if token.mint_address in {pool.base_token_mint, pool.quote_token_mint}
            ]
            dlmm = dlmm_pools.get(token.mint_address, [])


            price = prices.get(token.mint_address)
            rocketscan_metric = preexisting_metrics.get(token.mint_address)
            if token.mint_address in rocketscan_data:
                rocketscan_metric = rocketscan_data.get(token.mint_address)
            risk_metrics = self._build_risk_metrics(
                token,
                token_stats,
                damm,
                dlmm,
                price,
                rocketscan_metric,
            )
            self._storage.upsert_token(token)
            decision = self._determine_control(
                token, token_stats, metadata.get(token.mint_address), risk_metrics
            )
            self._storage.upsert_token_risk_metrics(risk_metrics)
            liquidity_event = build_liquidity_event(
                token,
                token_stats,
                damm,
                dlmm,
                self._price_oracle,
            )
            entry = TokenUniverseEntry(
                token=token,
                stats=token_stats,
                damm_pools=damm,
                dlmm_pools=dlmm,
                control=decision,
                risk=risk_metrics,
                liquidity_event=liquidity_event,
            )
            entries.append(entry)

        entries.sort(
            key=lambda item: (item.risk.liquidity_usd if item.risk else 0.0, item.token.symbol),
            reverse=True,
        )
        if len(entries) > limit:
            return entries[:limit]
        return entries

    def _collect_candidates(self, limit: int) -> List[TokenMetadata]:
        existing_tokens = list(self._storage.list_tokens())
        seen: Set[str] = set()
        candidates: List[TokenMetadata] = []
        now = datetime.now(timezone.utc)

        def add_candidate(token: TokenMetadata) -> bool:
            mint = token.mint_address
            if not mint or mint in seen:
                return False
            seen.add(mint)
            candidates.append(token)
            return True

        def load_from(source: Iterable[TokenMetadata]) -> None:
            for token in source:
                if add_candidate(token) and len(candidates) >= limit:
                    break

        # PRIORITIZE DAMM pools first - tokens with actual pools!
        if self._damm_client is not None:
            damm_limit = min(limit * 2, getattr(self._app_config.data_sources, "damm_page_limit", 50))
            recent_pools = self._damm_client.list_recent_pools(damm_limit)
            for pool in recent_pools:
                base_fee = pool.fee_scheduler_min_bps or pool.fee_bps or 0
                if base_fee < self._app_config.strategy.launch.min_base_fee_bps:
                    continue
                if pool.created_at is not None:
                    # Handle timezone-aware vs naive datetime comparison
                    pool_created_at = pool.created_at
                    if pool_created_at.tzinfo is None:
                        # Convert naive datetime to aware
                        pool_created_at = pool_created_at.replace(tzinfo=timezone.utc)
                    age_minutes = (now - pool_created_at).total_seconds() / 60
                    if age_minutes > self._app_config.strategy.launch.max_age_minutes * 3:
                        continue
                candidate = self._pool_to_metadata(pool)
                if not candidate:
                    continue
                if add_candidate(candidate) and len(candidates) >= limit:
                    break

        # Only use other sources if we haven't found enough candidates with pools
        if len(candidates) < limit:
            remaining = limit - len(candidates)

            # Prioritise fresh discovery feeds before recycling stored tokens.
            if self._token_config and self._app_config.data_sources.enable_axiom:
                if remaining > 0:
                    load_from(self._axiom_client.list_recent_tokens(limit=remaining))

            if (
                self._token_config
                and self._app_config.data_sources.enable_pumpfun
                and len(candidates) < limit
            ):
                remaining = limit - len(candidates)
                if remaining > 0:
                    load_from(self._pumpfun_client.list_new_tokens(limit=remaining))

        # Last resort: registry tokens
        if len(candidates) < limit:
            remaining = limit - len(candidates)
            registry_tokens = self._sample_registry_tokens(remaining, seen)
            load_from(registry_tokens)

        if len(candidates) > limit:
            return candidates[:limit]
        return candidates

    def _sample_registry_tokens(self, remaining: int, seen: Set[str]) -> List[TokenMetadata]:
        if remaining <= 0:
            return []

        preferred_tags = ["stablecoin", "verified", "dex", "defi"]
        sample_method = getattr(self._token_list_client, "sample", None)
        if callable(sample_method):
            entries = sample_method(
                remaining * 5,
                verified_only=not self._token_config.allow_unverified_token_list,
                preferred_tags=preferred_tags,
            )
        else:
            registry_dict = getattr(self._token_list_client, "_entries", {})
            entries = list(registry_dict.values()) if isinstance(registry_dict, dict) else []

        deny_mints = {mint.lower() for mint in self._token_config.deny_mints}
        suspicious_keywords = [keyword.lower() for keyword in self._token_config.suspicious_keywords]

        sampled: List[TokenMetadata] = []
        for entry in entries:
            mint = entry.mint
            if not mint or mint in seen or mint.lower() in deny_mints:
                continue
            symbol_lower = entry.symbol.lower() if entry.symbol else ""
            name_lower = entry.name.lower() if entry.name else ""
            if any(keyword in symbol_lower or keyword in name_lower for keyword in suspicious_keywords):
                continue

            if entry.decimals < self._token_config.min_decimals or entry.decimals > self._token_config.max_decimals:
                continue

            metadata = TokenMetadata(
                mint_address=mint,
                symbol=entry.symbol,
                name=entry.name,
                decimals=entry.decimals,
                sources=["solana-token-list"],
            )

            control = self._storage.get_token_control(mint)
            if control and control.status == "deny":
                continue

            sampled.append(metadata)
            seen.add(mint)
            if len(sampled) >= remaining:
                break

        return sampled

    def _select_pool_mint(
        self, pool: DammPoolSnapshot
    ) -> Optional[tuple[str, Optional[str]]]:
        def _is_non_stable(mint: str) -> bool:
            return mint not in STABLECOIN_MINTS and mint != SOL_MINT

        options: list[tuple[str, Optional[str]]] = []
        if pool.base_token_mint:
            options.append((pool.base_token_mint, getattr(pool, "base_symbol", None)))
        if pool.quote_token_mint:
            options.append((pool.quote_token_mint, getattr(pool, "quote_symbol", None)))
        for mint, symbol in options:
            if _is_non_stable(mint):
                return mint, symbol
        return options[0] if options else None

    def _pool_to_metadata(self, pool: DammPoolSnapshot) -> Optional[TokenMetadata]:
        selection = self._select_pool_mint(pool)
        if selection is None:
            return None
        mint, symbol = selection
        if not mint:
            return None
        name = symbol or mint
        return TokenMetadata(
            mint_address=mint,
            symbol=symbol or "",
            name=name,
            decimals=9,
            sources=["damm"],
        )

    def _apply_token_list_metadata(
        self, tokens: Iterable[TokenMetadata], metadata: Dict[str, TokenListEntry]
    ) -> None:
        for token in tokens:
            entry = metadata.get(token.mint_address)
            if not entry:
                continue
            if entry.symbol and not token.symbol:
                token.symbol = entry.symbol
            if entry.name and not token.name:
                token.name = entry.name
            token.decimals = entry.decimals
            sources = set(token.sources)
            sources.add("solana-token-list")
            token.sources = sorted(sources)

    def _build_risk_metrics(
        self,
        token: TokenMetadata,
        stats: Optional[TokenOnChainStats],
        damm: List[DammPoolSnapshot],
        dlmm: List[DlmmPoolSnapshot],
        price: Optional[float],
        rocketscan_metrics: Optional[RocketscanMetrics],
    ) -> TokenRiskMetrics:
        liquidity_candidates: List[float] = []
        volume_candidates: List[float] = []
        price_reference = price
        sol_price_cache: Optional[float] = None

        def _quote_price(mint: Optional[str]) -> Optional[float]:
            nonlocal sol_price_cache
            if not mint:
                return None
            if mint in STABLECOIN_MINTS:
                return 1.0
            if mint == SOL_MINT:
                if sol_price_cache is None:
                    sol_price_cache = self._price_oracle.get_price(SOL_MINT)
                return sol_price_cache
            return self._price_oracle.get_price(mint)

        def _liquidity_usd(pool: DammPoolSnapshot, base_price: Optional[float]) -> Optional[float]:
            base_amt = pool.base_token_amount or 0.0
            quote_amt = pool.quote_token_amount or 0.0
            quote_price = _quote_price(pool.quote_token_mint)
            effective_base_price = base_price if base_price and base_price > 0 else None
            if not effective_base_price and quote_price and base_amt > 0:
                effective_base_price = (quote_amt * quote_price) / max(base_amt, 1e-9)
            base_usd = base_amt * effective_base_price if effective_base_price else 0.0
            quote_usd = quote_amt * quote_price if quote_price else 0.0
            total = base_usd + quote_usd
            return total if total > 0 else None
        for pool in damm:
            base_price = pool.price_usd or price_reference or price
            liq_usd = _liquidity_usd(pool, base_price)
            if liq_usd is not None:
                liquidity_candidates.append(liq_usd)
            elif pool.tvl_usd is not None and pool.tvl_usd > 0:
                liquidity_candidates.append(pool.tvl_usd)
            else:
                liquidity_candidates.append(pool.base_token_amount + pool.quote_token_amount)
            if pool.volume_24h_usd is not None and pool.volume_24h_usd > 0:
                volume_candidates.append(pool.volume_24h_usd)
            if base_price is not None and base_price > 0:
                price_reference = base_price
        for pool in dlmm:
            base_price = pool.price_usd or price_reference or price
            liq_usd = _liquidity_usd(pool, base_price)
            if liq_usd is not None:
                liquidity_candidates.append(liq_usd)
            elif pool.tvl_usd is not None and pool.tvl_usd > 0:
                liquidity_candidates.append(pool.tvl_usd)
            else:
                liquidity_candidates.append(pool.base_token_amount + pool.quote_token_amount)
            if pool.volume_24h_usd is not None and pool.volume_24h_usd > 0:
                volume_candidates.append(pool.volume_24h_usd)
            if base_price is not None and base_price > 0:
                price_reference = base_price

        liquidity_usd = max(liquidity_candidates) if liquidity_candidates else 0.0
        volume_24h_usd = max(volume_candidates) if volume_candidates else 0.0

        if liquidity_usd <= 0 and stats and stats.liquidity_estimate:
            if price_reference is None:
                price_reference = price
            liquidity_usd = stats.liquidity_estimate * (price_reference or 0.0) * 2

        holder_count = stats.holder_count if stats else 0
        top_holder_pct = stats.top_holder_pct if stats else 1.0
        top10_holder_pct = stats.top10_holder_pct if stats else 1.0
        volatility_score = 0.0
        if liquidity_usd > 0:
            volatility_score = min(volume_24h_usd / max(liquidity_usd, 1.0), 5.0)

        has_oracle_price = price_reference is not None
        price_confidence_bps = (
            self._app_config.venues.dlmm.required_oracle_confidence_bps if has_oracle_price else 0
        )

        risk_flags: List[str] = []
        min_liquidity = self._token_config.min_liquidity_usd
        min_volume = self._token_config.min_volume_24h_usd
        min_holders = self._token_config.min_holder_count

        if liquidity_usd < min_liquidity:
            risk_flags.append(f"low_liquidity<{min_liquidity}")
        if volume_24h_usd < min_volume:
            risk_flags.append(f"low_volume<{min_volume}")
        if holder_count < min_holders:
            risk_flags.append(f"few_holders<{min_holders}")
        if top_holder_pct > self._token_config.max_holder_concentration_pct:
            risk_flags.append("high_holder_concentration")
        if top10_holder_pct > self._token_config.max_top10_holder_pct:
            risk_flags.append("high_top10_concentration")
        if not has_oracle_price and self._token_config.require_oracle_price:
            risk_flags.append("missing_oracle_price")

        if stats and stats.freeze_authority and self._token_config.deny_freeze_authority:
            risk_flags.append("freeze_authority_set")

        dev_holding_pct: Optional[float] = None
        sniper_holding_pct: Optional[float] = None
        insider_holding_pct: Optional[float] = None
        bundler_holding_pct: Optional[float] = None
        if rocketscan_metrics is not None:
            dev_holding_pct = rocketscan_metrics.dev_balance_pct
            sniper_holding_pct = rocketscan_metrics.snipers_pct
            insider_holding_pct = rocketscan_metrics.insiders_pct
            bundler_holding_pct = rocketscan_metrics.bundlers_pct
            if (
                dev_holding_pct is not None
                and dev_holding_pct > 1.0
                and "dev_hold" not in risk_flags
            ):
                risk_flags.append(f"dev_hold>{dev_holding_pct:.2f}")
            if sniper_holding_pct is not None and sniper_holding_pct > 20.0:
                risk_flags.append(f"snipers>{sniper_holding_pct:.2f}")
            if insider_holding_pct is not None and insider_holding_pct > 20.0:
                risk_flags.append(f"insiders>{insider_holding_pct:.2f}")
            if bundler_holding_pct is not None and bundler_holding_pct > 20.0:
                risk_flags.append(f"bundlers>{bundler_holding_pct:.2f}")

        metrics = TokenRiskMetrics(
            mint_address=token.mint_address,
            liquidity_usd=liquidity_usd,
            volume_24h_usd=volume_24h_usd,
            volatility_score=volatility_score,
            holder_count=holder_count,
            top_holder_pct=top_holder_pct,
            top10_holder_pct=top10_holder_pct,
            has_oracle_price=has_oracle_price,
            price_confidence_bps=price_confidence_bps,
            last_updated=datetime.now(timezone.utc),
            dev_holding_pct=dev_holding_pct,
            sniper_holding_pct=sniper_holding_pct,
            insider_holding_pct=insider_holding_pct,
            bundler_holding_pct=bundler_holding_pct,
            risk_flags=risk_flags,
        )

        return metrics

    def _determine_control(
        self,
        token: TokenMetadata,
        stats: Optional[TokenOnChainStats],
        token_list_entry: Optional[TokenListEntry],
        metrics: TokenRiskMetrics,
    ) -> TokenControlDecision:
        existing = self._storage.get_token_control(token.mint_address)
        manual_override = False
        if existing:
            if existing.source == MANUAL_SOURCE and self._token_config.allow_manual_override:
                return existing
            if existing.source not in {AUTO_SOURCE, MANUAL_SOURCE}:
                manual_override = True
        if manual_override:
            return existing

        status = "allow"
        reasons: List[str] = []
        min_liquidity = self._token_config.min_liquidity_usd
        min_volume = self._token_config.min_volume_24h_usd
        pause_liquidity_threshold = min_liquidity * (1 - self._token_config.autopause_liquidity_buffer)
        pause_volume_threshold = min_volume * (1 - self._token_config.autopause_volume_buffer)

        token_age_minutes = None
        if stats and stats.minted_at:
            token_age_minutes = (datetime.now(timezone.utc) - stats.minted_at).total_seconds() / 60

        if "freeze_authority_set" in metrics.risk_flags:
            status = "deny"
            reasons.append("Auto deny: freeze authority enabled")
        elif not metrics.has_oracle_price and self._token_config.require_oracle_price:
            status = "pause"
            reasons.append("Auto pause: oracle price unavailable")
        elif metrics.liquidity_usd < pause_liquidity_threshold:
            status = "pause"
            reasons.append(
                f"Auto pause: liquidity {metrics.liquidity_usd:.0f} < {pause_liquidity_threshold:.0f}"
            )
        elif metrics.volume_24h_usd < pause_volume_threshold:
            status = "pause"
            reasons.append(
                f"Auto pause: volume {metrics.volume_24h_usd:.0f} < {pause_volume_threshold:.0f}"
            )
        elif metrics.holder_count < self._token_config.min_holder_count:
            status = "pause"
            reasons.append(
                f"Auto pause: holders {metrics.holder_count} < {self._token_config.min_holder_count}"
            )
        elif metrics.top_holder_pct > self._token_config.max_holder_concentration_pct:
            status = "pause"
            reasons.append("Auto pause: holder concentration exceeds threshold")
        elif metrics.top10_holder_pct > self._token_config.max_top10_holder_pct:
            status = "pause"
            reasons.append("Auto pause: top10 concentration exceeds threshold")

        compliance_findings = self._compliance.evaluate(token, stats, metrics)
        deny_messages: List[str] = []
        warn_messages: List[str] = []
        for finding in compliance_findings:
            if finding.level in {"deny", "warn"}:
                flag = f"compliance_{finding.code}"
                if flag not in metrics.risk_flags:
                    metrics.risk_flags.append(flag)
            if finding.level == "deny":
                deny_messages.append(finding.message)
            elif finding.level == "warn":
                warn_messages.append(finding.message)

        if deny_messages:
            status = "deny"
            for message in deny_messages:
                if message not in reasons:
                    reasons.append(message)
        elif warn_messages:
            for message in warn_messages:
                if message not in reasons:
                    reasons.append(message)
            if status == "allow":
                status = "pause"

        if (
            status == "allow"
            and token_list_entry is None
            and not self._token_config.allow_unverified_token_list
        ):
            status = "pause"
            reasons.append("Auto pause: token not present in trusted registry")

        if (
            status == "allow"
            and token_age_minutes is not None
            and token_age_minutes > self._token_config.max_token_age_minutes
        ):
            status = "pause"
            reasons.append("Auto pause: token older than discovery window")

        if status == "allow" and not reasons:
            reasons.append("Auto approve: metrics healthy")

        decision = TokenControlDecision(
            mint_address=token.mint_address,
            status=status,
            reason="; ".join(reasons),
            source=AUTO_SOURCE,
            updated_at=datetime.now(timezone.utc),
        )

        self._storage.upsert_token_control(decision)

        return decision


def build_liquidity_event(
    token: TokenMetadata,
    stats: Optional[TokenOnChainStats],
    damm: List[DammPoolSnapshot],
    dlmm: List[DlmmPoolSnapshot],
    oracle: PriceOracle,
) -> LiquidityEvent:
    """Create a liquidity event from DAMM/DLMM snapshots or fall back to heuristics."""

    def best_pool() -> Optional[
        tuple[
            str,
            float,
            float,
            int,
            Optional[str],
            Optional[float],
            Optional[datetime],
            Optional[str],
        ]
    ]:
        candidates: List[
            tuple[
                str,
                float,
                float,
                int,
                Optional[str],
                Optional[float],
                Optional[datetime],
                Optional[str],
            ]
        ] = []
        for pool in damm:
            liquidity = pool.tvl_usd or (pool.base_token_amount + pool.quote_token_amount)
            candidates.append(
                (
                    pool.address,
                    pool.base_token_amount,
                    pool.quote_token_amount,
                    pool.fee_bps,
                    pool.quote_token_mint,
                    pool.price_usd,
                    pool.created_at,
                    pool.launchpad,
                )
            )
        for pool in dlmm:
            liquidity = pool.tvl_usd or (pool.base_token_amount + pool.quote_token_amount)
            candidates.append(
                (
                    pool.address,
                    pool.base_token_amount,
                    pool.quote_token_amount,
                    pool.fee_bps,
                    pool.quote_token_mint,
                    pool.price_usd,
                    getattr(pool, "created_at", None),
                    None,
                )
            )
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[1] + item[2], reverse=True)
        return candidates[0]

    pool_data = best_pool()
    price_usd: Optional[float] = None
    quote_token_mint: Optional[str] = None
    base_liquidity = 0.0
    quote_liquidity = 0.0
    fee_bps = 30
    created_at: Optional[datetime] = None
    launchpad: Optional[str] = None
    if pool_data is not None:
        (
            pool_address,
            base_liquidity,
            quote_liquidity,
            fee_bps,
            quote_token_mint,
            price_usd,
            created_at,
            launchpad,
        ) = pool_data
    else:
        pool_address = f"heuristic-{token.mint_address[:8]}"

    if price_usd is None and quote_token_mint:
        if quote_token_mint in STABLECOIN_MINTS:
            price_usd = quote_liquidity / base_liquidity if base_liquidity else None
        else:
            quote_price = oracle.get_price(quote_token_mint)
            if quote_price is not None and base_liquidity > 0:
                price_usd = (quote_liquidity * quote_price) / max(base_liquidity, 1e-9)

    if price_usd is None:
        price_usd = oracle.get_price(token.mint_address)

    if base_liquidity == 0 and stats:
        base_liquidity = stats.liquidity_estimate
    if quote_liquidity == 0 and price_usd is not None:
        quote_liquidity = base_liquidity * price_usd

    tvl_usd = None
    if price_usd is not None:
        tvl_usd = (base_liquidity * price_usd) + quote_liquidity

    timestamp = created_at or datetime.now(timezone.utc)

    return LiquidityEvent(
        timestamp=timestamp,
        token=token,
        pool_address=pool_address,
        base_liquidity=base_liquidity,
        quote_liquidity=quote_liquidity,
        pool_fee_bps=fee_bps,
        tvl_usd=tvl_usd,
        volume_24h_usd=None,
        price_usd=price_usd,
        quote_token_mint=quote_token_mint,
        source="meteora" if pool_data else "onchain-heuristic",
        launchpad=launchpad,
    )


__all__ = ["TokenRegistryAggregator", "build_liquidity_event"]
