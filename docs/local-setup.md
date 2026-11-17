# Running locally on a new machine

This guide starts from a machine with nothing installed and ends with the API scoring
transactions at `http://localhost:8000`. It covers macOS, Linux and Windows. You can run the
project natively with Python (best for development) or with Docker (no Python needed).

| Path | Use it when | Time |
|---|---|---|
| [A. Python (native)](#a-python-native) | You want to develop, run tests or train models | ~5 min |
| [B. Docker](#b-docker) | You only want to run the API | ~3 min |

Everything below was checked from a fresh clone in a clean virtual environment: install,
tests, synthetic training, serving with a model, the smoke test and the Redis backend.

---

## 1. Prerequisites

You need **Git** for both paths, **Python 3.11, 3.12 or 3.13** for path A, and **Docker**
for path B. `make` is optional (every `make` target has a plain command next to it below).

### macOS

```bash
xcode-select --install                    # git + make
brew install python@3.12                  # https://brew.sh if you don't have Homebrew
# Path B only: install Docker Desktop (https://www.docker.com/products/docker-desktop/)
```

Apple Silicon (M1–M4) works natively; all dependencies ship arm64 wheels.

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install -y git make python3 python3-venv python3-pip curl
python3 --version                         # must be 3.11+; otherwise: sudo apt install python3.12 python3.12-venv
# Path B only: https://docs.docker.com/engine/install/ubuntu/ (includes the compose plugin)
```

### Fedora / RHEL

```bash
sudo dnf install -y git make python3 python3-pip curl
```

### Windows

Two options:

- **WSL 2 (recommended).** In an admin PowerShell run `wsl --install`, reboot, open
  "Ubuntu" from the Start menu and follow the Ubuntu steps above. Everything in this guide,
  including `make` and the bash smoke test, then works unchanged.
- **Native PowerShell.** Install Git (`winget install Git.Git`) and Python
  (`winget install Python.Python.3.12`, tick "Add to PATH"). There is no `make`, so use the
  plain commands shown under each step. Docker Desktop covers path B.

Check what you have:

```bash
git --version
python3 --version      # Windows PowerShell: py --version
docker --version       # path B only
```

---

## 2. Get the code

```bash
git clone https://github.com/Csaf111/realtime-fraud-risk-engine.git
cd realtime-fraud-risk-engine
```

`v1` is the default (integration) branch. `prod` is what is running in production; use
`git checkout prod` if you want exactly that.

---

## A. Python (native)

### A1. Create a virtual environment and install

macOS / Linux / WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
make install                 # = pip install -e ".[dev]"
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1   # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

`[dev]` installs the serving dependencies plus training (pandas, NumPy, scikit-learn),
Redis client, tests and linters. To install only what serving needs, use `pip install -e .`.

### A2. Check the install

```bash
make test-fast               # = pytest -m "not slow"   (a few seconds)
make test                    # = pytest --cov --cov-report=term-missing   (full suite)
```

### A3. Run the API (rules-only)

```bash
make run                     # = uvicorn fraud_engine.api:app --reload --app-dir src
```

Open <http://localhost:8000/docs> for the interactive API. Without a model the engine runs
the explainable rules only; that is a valid mode, not an error.

### A4. Train a model and serve it

You don't need the real dataset to try this. Synthetic data proves the pipeline end to end
(its metrics mean nothing):

```bash
make train-synthetic         # = python -m fraud_engine.train --synthetic 20000 --out models/baseline.json
```

This writes into `models/` (git-ignored, never commit it):

- `baseline.json`: the model
- `baseline.metrics.json`: evaluation metrics
- `baseline.score_reference.json`, `baseline.feature_reference.json`: drift references

Serve it:

```bash
# macOS / Linux / WSL
export FRE_MODEL_PATH=models/baseline.json
export FRE_SCORE_REFERENCE_PATH=models/baseline.score_reference.json
export FRE_FEATURE_REFERENCE_PATH=models/baseline.feature_reference.json
make run
```

```powershell
# Windows PowerShell
$env:FRE_MODEL_PATH = "models/baseline.json"
$env:FRE_SCORE_REFERENCE_PATH = "models/baseline.score_reference.json"
$env:FRE_FEATURE_REFERENCE_PATH = "models/baseline.feature_reference.json"
uvicorn fraud_engine.api:app --reload --app-dir src
```

`curl localhost:8000/ready` should now report `"mode":"rules+model"` and both drift
references as `"loaded"`.

To train on the **real PaySim data**, download the CSV to `data/raw/paysim.csv` (see
[dataset-paysim.md](dataset-paysim.md)) and run `make train`. All training options are in
the README's "Training on PaySim" section.

---

## B. Docker

Needs Docker Desktop (macOS / Windows) or Docker Engine with the compose plugin (Linux).
Make sure Docker is running first (`docker info` should not error).

```bash
docker compose up --build            # rules-only API on http://localhost:8000
```

Or without compose:

```bash
make docker-build && make docker-run
# = docker build -t realtime-fraud-risk-engine:local . && docker run --rm -p 8000:8000 realtime-fraud-risk-engine:local
```

The image holds only the serving dependencies, so it cannot train. To serve a model,
train it with path A (step A4) first; `models/` is mounted into the container read-only.
Then uncomment `FRE_MODEL_PATH` in `docker-compose.yml`:

```yaml
    environment:
      FRE_MODEL_PATH: /app/models/baseline.json
```

and run `docker compose up --build` again. Stop with `Ctrl+C`, or `docker compose down` if
you started it with `-d`.

---

## 3. Verify it works

With the API running (either path), in a second terminal from the repo root:

```bash
make smoke                           # = ./scripts/smoke_test.sh http://localhost:8000
```

It waits for `/health`, then checks that the benign sample is `APPROVE`, the fraud sample
is `BLOCK`, and `/metrics` is served. It ends with `Smoke test passed`. The script needs
bash and curl; on native Windows run it from Git Bash, or call the API by hand:

```bash
curl -s localhost:8000/health
curl -s localhost:8000/v1/score -H 'content-type: application/json' \
  -d @scripts/sample_fraud.json | python -m json.tool
```

```powershell
Invoke-RestMethod http://localhost:8000/v1/score -Method Post -ContentType 'application/json' `
  -InFile scripts/sample_fraud.json
```

Useful endpoints: `/docs` (Swagger UI), `/ready` (mode and model version), `/v1/rules`
(rule catalogue), `/metrics` (Prometheus).

---

## 4. Optional extras

### Shared velocity state with Redis

The default velocity store is in-process. To try the multi-replica backend locally:

```bash
docker run -d --name fre-redis -p 6379:6379 redis:7   # or: brew/apt install redis && redis-server
export FRE_VELOCITY_BACKEND=redis
export FRE_REDIS_URL=redis://localhost:6379/0
make run
```

The `redis` client is already installed by `[dev]`; for a serving-only install use
`pip install -e ".[redis]"`.

### Streaming mode

Replay a file of transactions instead of calling the API:

```bash
python -m fraud_engine.stream --input txns.jsonl --output results.jsonl --dead-letter dlq.jsonl
```

Kafka mode needs `pip install confluent-kafka` and a broker. See [streaming.md](streaming.md).

### Configuration

Every setting is an `FRE_*` environment variable; the full table is in the README's
"Configuration" section. A `.env` file is git-ignored if you want to keep local values in
one place (load it with `set -a; source .env; set +a`).

---

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `python3: command not found` / version below 3.11 | Install 3.11+ (step 1); on Windows use `py -3.12` |
| `error: externally-managed-environment` from pip | You're outside the venv. Run `source .venv/bin/activate` (or `.venv\Scripts\Activate.ps1`) first |
| `make: command not found` | Use the plain command shown next to each `make` target, or install make (step 1) |
| `ModuleNotFoundError: fraud_engine` | Run from the repo root, inside the venv, after `pip install -e ".[dev]"` |
| `Address already in use` on port 8000 | Stop the other process, or run on another port: `uvicorn fraud_engine.api:app --app-dir src --port 8001` (and `./scripts/smoke_test.sh http://localhost:8001`) |
| `/ready` says `rules-only` after training | `FRE_MODEL_PATH` isn't set in *this* terminal, or the path is wrong. In Docker it must be `/app/models/...` |
| API refuses to start with a `ValueError` | A setting is out of range, e.g. `FRE_REVIEW_THRESHOLD` above `FRE_BLOCK_THRESHOLD`. Unset it or fix the value |
| `failed to connect to the docker API` | Docker isn't running. Start Docker Desktop, or `sudo systemctl start docker` on Linux |
| `scripts/smoke_test.sh: Permission denied` | `chmod +x scripts/smoke_test.sh`, or run `bash scripts/smoke_test.sh` |
| `Connection refused` to Redis | Start Redis (section 4) or unset `FRE_VELOCITY_BACKEND` |

## Updating an existing checkout

```bash
git pull
source .venv/bin/activate
make install                 # picks up new or changed dependencies
make test-fast
```
