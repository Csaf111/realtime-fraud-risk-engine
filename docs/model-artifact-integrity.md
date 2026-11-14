# Model artifact integrity and trust boundary

The baseline is loaded from a local JSON artifact. It is **not** an authenticated or signed model registry: filesystem access and artifact provenance must be controlled by deployment infrastructure.

## Runtime validation

At model construction, the loader checks the exact ordered feature schema and vector lengths, then rejects NaN and positive/negative infinity in means, scales, coefficients and intercept. This prevents malformed artifacts from silently producing non-finite risk scores or decisions. Existing zero-variance scale handling remains compatible: finite nonpositive scales are normalised to 1.0, as in the original implementation.

Gradient-boosted tree artifacts (`model_type: "gradient_boosted_trees"`, see `fraud_engine/trees.py`) are checked tree by tree: equal-length node arrays, finite thresholds and leaf values, finite positive covers, split features inside the feature schema, child indices that point strictly forward (so no cycles or out-of-range jumps), child covers that sum to the parent's (required for TreeSHAP additivity), at least one tree and a maximum depth of 32 (bounding recursion and per-request latency). Unknown `model_type` values are rejected; artifacts without one are read as logistic baselines for backwards compatibility. Training additionally refuses to write a GBT artifact whose pure-Python predictions disagree with scikit-learn's.

At inference, the model rejects non-finite input feature values before standardisation. The API independently validates incoming monetary values.

## Remaining risks

- JSON artifacts have no signature, checksum pinning or provenance attestation.
- The v1 scoring rules and feature pipeline still use post-transaction balances; **do not use v1 for live payment authorisation**.
- Model validation does not establish predictive accuracy, fairness, calibration or robustness to dataset shift.
- A finite but extremely large coefficient or feature may still cause numerical overflow in a contribution; bounded feature contracts and numerical stress tests remain future work.

## Suggested deployment controls

Store versioned artifacts in an access-controlled registry, pin a verified SHA-256 digest in deployment configuration, review training provenance, run shadow scoring and canary checks, and define rollback thresholds. Never load untrusted artifacts from request input.
