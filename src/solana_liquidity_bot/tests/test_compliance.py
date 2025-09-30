from datetime import datetime, timezone, timedelta

from solana_liquidity_bot.config import settings
from solana_liquidity_bot.datalake.schemas import TokenMetadata, TokenOnChainStats, TokenRiskMetrics
from solana_liquidity_bot.ingestion.compliance import ComplianceEngine


def _risk_metrics(mint: str) -> TokenRiskMetrics:
    return TokenRiskMetrics(
        mint_address=mint,
        liquidity_usd=50_000.0,
        volume_24h_usd=80_000.0,
        volatility_score=0.2,
        holder_count=500,
        top_holder_pct=0.2,
        top10_holder_pct=0.4,
        has_oracle_price=True,
        price_confidence_bps=10,
        last_updated=datetime.now(timezone.utc),
        risk_flags=[],
    )


def test_compliance_deny_list_blocks_token() -> None:
    config = settings.TokenUniverseConfig(
        deny_mints=["BlockedMint"],
        suspicious_keywords=["rug"],
    )
    engine = ComplianceEngine(config)
    token = TokenMetadata(mint_address="BlockedMint", symbol="BLK", name="Blocked")
    findings = engine.evaluate(token, None, _risk_metrics(token.mint_address))
    codes = {finding.code for finding in findings if finding.level == "deny"}
    assert "deny_mint" in codes


def test_compliance_warns_on_recent_mints() -> None:
    config = settings.TokenUniverseConfig(aml_min_token_age_minutes=60)
    engine = ComplianceEngine(config)
    token = TokenMetadata(mint_address="MintWarn", symbol="WARN", name="Warned")
    stats = TokenOnChainStats(
        token=token,
        total_supply=1_000_000,
        decimals=6,
        holder_count=10,
        top_holder_pct=0.9,
        top10_holder_pct=0.95,
        liquidity_estimate=1_000,
        minted_at=datetime.now(timezone.utc) - timedelta(minutes=15),
        last_activity_at=datetime.now(timezone.utc),
        mint_authority="Authority",
        freeze_authority=None,
    )
    findings = engine.evaluate(token, stats, _risk_metrics(token.mint_address))
    codes = {finding.code for finding in findings if finding.level == "warn"}
    assert "recent_mint" in codes
    assert "holder_concentration" in codes


def test_compliance_blocks_suspicious_keyword() -> None:
    config = settings.TokenUniverseConfig(suspicious_keywords=["honeypot"])
    engine = ComplianceEngine(config)
    token = TokenMetadata(mint_address="MintKeyword", symbol="HONEY", name="Honeypot Token")
    findings = engine.evaluate(token, None, _risk_metrics(token.mint_address))
    assert any(finding.code == "keyword" for finding in findings)
