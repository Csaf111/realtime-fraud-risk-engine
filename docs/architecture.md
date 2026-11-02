# Architecture

## 1. System context

```mermaid
flowchart LR
    client["Payment / wallet service"] -- "POST /v1/score<br/>(transaction JSON)" --> api
    subgraph engine["realtime-fraud-risk-engine (container)"]
        api["FastAPI app"] --> core["RiskEngine"]
        core --> state[("VelocityStore<br/>in-memory")]
        core --> rules["Rulebook R001–R008"]
        core --> model["Logistic baseline<br/>(optional JSON artifact)"]
    end
    api -- "score, decision,<br/>reasons" --> client
    api -- "/metrics" --> prom["Prometheus / monitoring"]
    analyst["Fraud analyst"] -- "REVIEW queue<br/>+ reason codes" --> client
```

## 2. Request flow

```mermaid
sequenceDiagram
    autonumber
    participant C as Client
    participant A as API (FastAPI)
    participant E as RiskEngine
    participant S as VelocityStore
    participant R as Rules
    participant M as Model (optional)

    C->>A: POST /v1/score {transaction}
    A->>A: Pydantic validation (rejects isFraud / unknown fields → 422)
    A->>E: score(txn)
    E->>S: snapshot(nameOrig, nameDest, step)
    Note right of S: Only events BEFORE this one
    S-->>E: VelocitySnapshot
    E->>E: compute_features(txn, snapshot)
    E->>R: evaluate_rules(txn, snapshot)
    R-->>E: rules_score, hits
    opt model loaded
        E->>M: explain(features)
        M-->>E: probability, top contributions
        E->>E: risk = w·model + (1−w)·rules
    end
    E->>E: decision via thresholds (APPROVE / REVIEW / BLOCK)
    E->>S: record(txn)  (after scoring)
    E-->>A: RiskResult
    A-->>C: 200 {risk_score, decision, reasons, rule_hits, top_features, latency_ms}
```

## 3. Offline training pipeline (leakage-safe)

```mermaid
flowchart TD
    csv[("PaySim CSV<br/>data/raw/paysim.csv")] --> load["load_paysim<br/>+ validate_schema"]
    load --> drop["Drop isFlaggedFraud"]
    drop --> sort["Sort by step (stable)"]
    sort --> replay["Stream replay<br/>VelocityStore + compute_features<br/>(same code as API)"]
    replay --> split["temporal_split<br/>train | gap | test"]
    split --> checks{"leakage_checks"}
    checks -- violation --> fail["LeakageError ✗"]
    checks -- ok --> scale["fit_standardiser(train only)"]
    scale --> fit["LogisticRegression<br/>class_weight=balanced"]
    fit --> eval["Evaluate on test<br/>model vs rules-only"]
    eval --> art[("models/baseline.json<br/>+ metrics.json")]
    art -. "FRE_MODEL_PATH" .-> serve["API / container"]
```

## 4. Scoring and explainability

```mermaid
flowchart LR
    txn["Transaction"] --> F["17 features"]
    txn --> RB["Rules<br/>points → rules_score ∈ [0,1]"]
    F --> LR["Logistic regression<br/>contribution_i = coef_i · z_i"]
    LR --> P["p(fraud)"]
    RB --> B(("Blend<br/>w = FRE_MODEL_WEIGHT"))
    P --> B
    B --> D{"Thresholds"}
    D -- "< review" --> AP["APPROVE"]
    D -- "≥ review" --> RV["REVIEW"]
    D -- "≥ block" --> BL["BLOCK"]
    RB --> X["reasons[]"]
    LR --> X
```

The model is linear in standardised features. Each feature's log-odds contribution is exact,
not a SHAP-style approximation. Rule hits carry stable codes that analysts can look up via
`GET /v1/rules`.

## 5. CI/CD and branch promotion

```mermaid
flowchart LR
    dev["feature branch"] -- PR --> v1["v1<br/>(integration, default)"]
    v1 -- "Promote PR" --> stg["staging"]
    stg -- "Promote PR" --> prod["prod"]

    v1 -. push .-> CI["CI<br/>lint · mypy · tests 3.11–3.13<br/>synthetic training · Docker smoke"]
    stg -. push .-> CDs["CD: CI gate → GHCR :staging → deploy staging"]
    prod -. push .-> CDp["CD: CI gate → GHCR :prod/:latest/:vX → approval → deploy production → release"]
```

## 6. Scaling notes

| Concern | Today (starter) | Next step |
|---|---|---|
| Velocity state | In-process dict of deques, one worker per container | Redis sorted sets keyed by account (`ZADD`/`ZRANGEBYSCORE`), same interface |
| Model | JSON coefficients, pure-Python inference (~µs) | Gradient-boosted trees + SHAP. Keep the rules as a floor |
| Throughput | Single uvicorn worker | Horizontal replicas behind a load balancer, once state is shared |
| Ingestion | Synchronous HTTP | Kafka consumer that scores events and emits decisions |
| Feedback | None | Label ingestion (chargebacks) → delayed-label retraining → shadow deployment |
| Monitoring | Counters + latency histogram | Score-distribution drift, alert rate, per-rule hit rate |
