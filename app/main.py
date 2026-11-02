"""FastAPI entry point for the explainable risk scoring service."""

from fastapi import FastAPI

from app.schemas import ScoreRequest, ScoreResponse
from app.scoring import score_transaction

app = FastAPI(
    title="Realtime Fraud Risk Engine",
    description=(
        "Educational, deterministic risk scoring. Scores are not fraud "
        "probabilities or production payment decisions."
    ),
    version="0.1.0",
)


@app.get("/health", tags=["operations"])
def health() -> dict[str, str]:
    """Liveness probe (does not check dependencies)."""
    return {"status": "ok"}


@app.post("/v1/score", response_model=ScoreResponse, tags=["risk"])
def score(request: ScoreRequest) -> ScoreResponse:
    """Return a heuristic score and explainable rule contributions."""
    return score_transaction(request)
