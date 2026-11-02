# Security and operational boundaries

This repository is a portfolio demonstration, **not a production fraud decisioning service**.

## Current safeguards

- Pydantic rejects invalid, non-finite or non-positive amounts and unexpected input fields.
- Only pre-decision features are accepted; target labels and post-event balances are rejected.
- The Docker image runs as a non-root user.
- No credentials, raw datasets or transaction histories are committed.
- GitHub Actions uses read-only repository permissions for CI.

## Known gaps

- No authentication, authorization, rate limiting or request-size enforcement beyond the app's standard stack.
- No TLS termination (use an ingress or reverse proxy).
- No audit storage, idempotency, replay prevention or transaction signing.
- No secrets management, vulnerability scanning, deployment approval gates or SBOM yet.
- No logging redaction policy enforced at the infrastructure layer.
- Dependencies are bounded but not locked to exact hashes.

## Deployment checklist (future work)

Use private network boundaries and HTTPS; authenticate callers with scoped credentials; apply WAF/rate limits and body-size limits; restrict CORS; protect logs from PII and account identifiers; implement structured audit trails with retention controls; rotate secrets via a managed store; scan images/dependencies; benchmark latency; define alerts, on-call runbooks and rollback procedures.

## Responsible use

Risk scores are arbitrary additive rule points, **not calibrated probabilities**. Do not automatically reject real customer payments using this prototype. Use synthetic data only unless appropriate permissions, data minimization and privacy safeguards are in place.

## Reporting vulnerabilities

Do not publish secrets or exploit details in public issues. Contact the repository owner privately through GitHub to coordinate remediation.
