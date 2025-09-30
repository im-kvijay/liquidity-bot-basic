# Security Notes

## Key management

* Never commit private keys or wallet mnemonics to the repository. Load signing keys from an encrypted secrets store and expose
  them via environment variables (`WALLET__PRIVATE_KEY`) or secure file paths.
* When running in production, disable `wallet.allow_test_keypair` and enforce HSM-based signing whenever possible.
* Rotate bearer tokens used by the dashboard and alerting integrations on a regular schedule.

## Dependency integrity

* Python dependencies are pinned in `pyproject.toml` and mirrored with hashes in `requirements.lock`. Re-install with
  `pip install --require-hashes -r requirements.lock` to enforce checksum verification.
* Node dependencies are pinned via `package.json` and `package-lock.json`. Use `npm ci` in CI/CD pipelines for deterministic
  installs.
* Security overrides enforce patched releases of `@solana/web3.js`, `cross-fetch`, `node-fetch`, and a vendored `bigint-buffer`
  build; keep these entries aligned with upstream advisories when bumping dependencies.
* Enable automatic alerts for CVEs (e.g., Dependabot) and audit the pinned versions quarterly.

## Secrets handling & observability

* Sensitive configuration (RPC keys, webhook tokens) should be set through environment variables or secret management platforms.
* Structured logs omit secret values; avoid injecting secrets into event payloads. Review any custom logging extensions before
  deployment.
* Configure alerting webhooks to use HTTPS endpoints and rotate credentials if delivery fails repeatedly.

## Operational safeguards

* The compliance engine enforces deny lists and AML heuristics, but manual reviews remain necessary. Monitor `/api/token-controls`
  for unexpected pauses or denials.
* Keep the SQLite database on encrypted storage and schedule periodic backups (at least hourly while trading live).
* Run the soak test harness (`python scripts/soak_test.py --minutes 60 --dry-run`) on staging infrastructure after significant
  strategy or dependency changes to detect regressions.
