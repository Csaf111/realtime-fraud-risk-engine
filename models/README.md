# models/

Trained artifacts land here (`baseline.json` + `baseline.metrics.json`). They are git-ignored.

- `make train` trains on the real PaySim CSV.
- `make train-synthetic` trains on synthetic data. Use it only to test the pipeline; its metrics mean nothing.

To serve a model, set `FRE_MODEL_PATH=/app/models/baseline.json` (Docker) or a local path.
Without it, the engine runs in rules-only mode.
