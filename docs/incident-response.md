# Incident response and rollback runbook

Scope: demonstration service and future staging deployments. This runbook must be adapted to the real hosting provider before production use.

## Severity and ownership

- **SEV1:** incorrect BLOCK/APPROVE decisions, sensitive-data exposure, widespread scoring outage, compromised model artifact.
- **SEV2:** material latency increase, drift alarm, stale velocity state, partial deployment failure.
- **SEV3:** isolated malformed requests, non-critical dashboard or documentation issue.

Assign an incident commander, engineering owner and decision/risk owner before a production pilot. Preserve UTC timestamps and affected release/model identifiers.

## Triage

1. Identify environment, image digest, commit SHA, model version, threshold settings and time of onset.
2. Check `/health` and `/ready`, application logs and `/metrics` (latency, decision mix, rule hits and score distribution).
3. Compare recent deployment, configuration and upstream payment event changes. Do **not** paste raw account identifiers or transaction bodies into tickets.
4. For elevated BLOCK rates, suspend automated enforcement using an approved payment-operations contingency; **do not silently change thresholds**.
5. If a model artifact is suspected, quarantine it, preserve its digest and redeploy the last verified artifact.

## Rollback procedure

1. Stop promotion to production; record the current container image digest and deployment revision.
2. Select the last known-good, CI-tested image **by immutable digest**, not the mutable `:latest` tag.
3. Roll back through the hosting provider's deployment mechanism; require the configured environment approval.
4. Verify `/health`, `/ready`, model version and a non-sensitive scoring smoke test.
5. Observe latency, error rate, decision distribution and queue capacity after rollback.
6. Document any velocity-state incompatibility. Rolling back code does not automatically roll back Redis state or processed events.

If scoring is unavailable, payment policy must define whether to queue, route to manual review or decline. This repository does not make that business decision.

## Evidence and follow-up

Capture GitHub Actions run URLs, release image digest, model artifact checksum, feature-schema version, monitoring snapshots and a redacted incident timeline. Create a postmortem with root cause, customer impact, detection gaps, action owners and regression tests. Avoid committing credentials, private datasets or personal financial information.
