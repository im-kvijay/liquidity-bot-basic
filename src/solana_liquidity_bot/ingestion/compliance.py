"""Compliance heuristics for token universe vetting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from ..config.settings import TokenUniverseConfig
from ..datalake.schemas import TokenMetadata, TokenOnChainStats, TokenRiskMetrics


@dataclass(slots=True)
class ComplianceFinding:
    """Outcome of a compliance rule evaluation."""

    level: str  # "deny", "warn", or "info"
    code: str
    message: str


class ComplianceEngine:
    """Evaluates tokens against deny lists and AML-style heuristics."""

    def __init__(self, config: TokenUniverseConfig) -> None:
        self._config = config
        self._deny_mints = {mint.lower() for mint in config.deny_mints}
        self._deny_creators = {creator.lower() for creator in config.deny_creators}
        self._deny_authorities = {authority.lower() for authority in config.deny_authorities}
        self._suspicious_keywords = {keyword.lower() for keyword in config.suspicious_keywords}
        self._sanctioned_accounts = {account.lower() for account in config.aml_sanctioned_accounts}

    def evaluate(
        self,
        token: TokenMetadata,
        stats: Optional[TokenOnChainStats],
        metrics: TokenRiskMetrics,
    ) -> List[ComplianceFinding]:
        """Return a list of compliance findings ordered by severity."""

        findings: List[ComplianceFinding] = []
        mint_address = token.mint_address.lower()
        if mint_address in self._deny_mints:
            findings.append(
                ComplianceFinding(
                    level="deny",
                    code="deny_mint",
                    message=f"Mint {token.mint_address} present in deny list",
                )
            )

        creator = (token.creator or "").lower()
        if creator and creator in self._deny_creators:
            findings.append(
                ComplianceFinding(
                    level="deny",
                    code="deny_creator",
                    message=f"Creator {token.creator} is blocked",
                )
            )

        decimals = stats.decimals if stats else token.decimals
        if decimals < self._config.min_decimals:
            findings.append(
                ComplianceFinding(
                    level="deny",
                    code="unsupported_decimals_low",
                    message=f"Decimals {decimals} below minimum {self._config.min_decimals}",
                )
            )
        if decimals > self._config.max_decimals:
            findings.append(
                ComplianceFinding(
                    level="deny",
                    code="unsupported_decimals_high",
                    message=f"Decimals {decimals} above maximum {self._config.max_decimals}",
                )
            )

        combined_text = f"{token.symbol or ''} {token.name or ''}".lower()
        for keyword in self._suspicious_keywords:
            if keyword and keyword in combined_text:
                findings.append(
                    ComplianceFinding(
                        level="deny",
                        code="keyword",
                        message=f"Keyword '{keyword}' indicates potential scam",
                    )
                )
                break

        if len(token.social_handles) < self._config.min_social_links:
            findings.append(
                ComplianceFinding(
                    level="warn",
                    code="missing_socials",
                    message="Insufficient verified social profiles",
                )
            )

        if stats:
            authorities = [stats.freeze_authority, stats.mint_authority]
            for authority in authorities:
                if not authority:
                    continue
                authority_lower = authority.lower()
                if authority_lower in self._deny_authorities:
                    findings.append(
                        ComplianceFinding(
                            level="deny",
                            code="blocked_authority",
                            message=f"Authority {authority} present in deny list",
                        )
                    )
                if authority_lower in self._sanctioned_accounts:
                    findings.append(
                        ComplianceFinding(
                            level="deny",
                            code="sanctioned_authority",
                            message=f"Authority {authority} flagged by AML list",
                        )
                    )

            if stats.minted_at is not None:
                age_minutes = (datetime.now(timezone.utc) - stats.minted_at).total_seconds() / 60
                if age_minutes < self._config.aml_min_token_age_minutes:
                    findings.append(
                        ComplianceFinding(
                            level="warn",
                            code="recent_mint",
                            message=(
                                f"Token minted {age_minutes:.1f} minutes ago (threshold "
                                f"{self._config.aml_min_token_age_minutes})"
                            ),
                        )
                    )

            if stats.holder_count < self._config.aml_min_holder_count:
                findings.append(
                    ComplianceFinding(
                        level="warn",
                        code="low_holder_count",
                        message=(
                            f"Holder count {stats.holder_count} below AML threshold "
                            f"{self._config.aml_min_holder_count}"
                        ),
                    )
                )

            if stats.top_holder_pct > self._config.aml_max_single_holder_pct:
                findings.append(
                    ComplianceFinding(
                        level="warn",
                        code="holder_concentration",
                        message=(
                            f"Top holder controls {stats.top_holder_pct:.2f} of supply "
                            f"(max {self._config.aml_max_single_holder_pct:.2f})"
                        ),
                    )
                )

        # Surface risk flags as informational findings so the dashboard can display them.
        for flag in metrics.risk_flags:
            findings.append(
                ComplianceFinding(
                    level="info",
                    code=f"risk_flag:{flag}",
                    message=f"Risk flag present: {flag}",
                )
            )

        return findings


__all__ = ["ComplianceEngine", "ComplianceFinding"]
