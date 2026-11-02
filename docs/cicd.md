# CI/CD

## Branches

| Branch | Role | What runs on push |
|---|---|---|
| `v1` | Default / integration. Feature PRs merge here | **CI** |
| `staging` | Pre-production. Updated only by promotion PR from `v1` | **CD → staging** |
| `prod` | Production. Updated only by promotion PR from `staging` | **CD → production** + GitHub release |

Pull requests into any of the three run CI.

## Workflows

### `ci.yml`: quality gate
1. **Lint & type-check**: `ruff check`, `ruff format --check`, `mypy`.
2. **Tests**: pytest on Python 3.11 / 3.12 / 3.13. Coverage gate ≥ 85%. Coverage and JUnit
   reports are uploaded as artifacts.
3. **Dependency audit**: `pip-audit` (report-only for now).
4. **Training pipeline**: trains the baseline on synthetic data, then loads it into the
   engine and checks a known-fraud transaction is blocked.
5. **Docker**: builds the image, runs it, checks it runs as non-root, and smoke-tests it
   in rules-only mode and again with the trained model mounted.

### `cd.yml`: build, publish, deploy
1. Calls `ci.yml` as a gate. Nothing ships unless CI passes.
2. Builds and pushes to GitHub Container Registry:
   - staging: `ghcr.io/csaf111/realtime-fraud-risk-engine:staging`, `:staging-<sha>`
   - prod: `:prod`, `:prod-<sha>`, `:latest`, `:v<version>`
3. **Deploy** job runs in a GitHub *environment* (`staging` or `production`):
   - If the environment secret `DEPLOY_WEBHOOK_URL` exists, it POSTs
     `{"image": "<ref@digest>", "environment": ..., "sha": ...}` to it. This works with
     Render/Railway deploy hooks, Coolify, or your own endpoint.
   - If the environment variable `APP_URL` exists, it smoke-tests the deployed service.
   - With neither set, the image is still published and the job leaves a notice.
4. **Release** (prod only): creates a GitHub release tagged `v<version>-build.<run>`
   that points at the image digest.

### `promote.yml`: one-click promotion
Run it from **Actions → Promote → Run workflow** and pick `staging` or `prod`. It opens
the PR `v1 → staging` or `staging → prod`.

## One-time repository setup

1. **Default branch**: Settings → General → Default branch → `v1`.
2. **Environments**: Settings → Environments → create `staging` and `production`.
   - On `production`, add yourself under **Required reviewers**. Prod deploys then wait
     for approval.
   - Optional: add the secret `DEPLOY_WEBHOOK_URL` and the variable `APP_URL` to each one.
3. **Branch protection** (Settings → Branches) for `v1`, `staging`, `prod`: require a PR
   and require the status checks `Lint & type-check` and `Tests (py3.12)`.
4. **Actions permissions**: Settings → Actions → General → Workflow permissions → enable
   *Allow GitHub Actions to create and approve pull requests* (needed by `promote.yml`).
5. **Package visibility**: after the first CD run, make the GHCR package public under
   *Packages* if you want anonymous `docker pull`.

## Pulling the image

```bash
docker pull ghcr.io/csaf111/realtime-fraud-risk-engine:staging
docker run --rm -p 8000:8000 ghcr.io/csaf111/realtime-fraud-risk-engine:staging
```
