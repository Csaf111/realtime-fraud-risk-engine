# Contributing

## Local development

Use Python 3.11+ and create an isolated virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
ruff check .
pytest -q
```

On Windows, activate with `.venv\Scripts\activate`.

## Change guidelines

- Make focused changes on a feature branch and open a pull request.
- Keep scoring rules deterministic, versioned and explainable.
- Add unit tests for scoring boundaries and API tests for contract changes.
- Do not add labels, post-event balances or future-derived features to online scoring.
- Update the README and relevant documentation with each meaningful change.
- Never commit datasets, account information, credentials, local `.env` files or trained models without reviewing rights and security.

## PR checklist

- [ ] Tests and Ruff pass locally.
- [ ] API changes documented and backward compatibility considered.
- [ ] New scoring rules include reason codes and threshold tests.
- [ ] Data-leakage and privacy implications reviewed.
- [ ] No credentials or large data artifacts committed.
