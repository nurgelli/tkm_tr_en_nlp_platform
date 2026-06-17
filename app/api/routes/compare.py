import json

from fastapi import APIRouter, HTTPException

from app.api.schemas import ComparisonResult, MetricsResponse, TextRequest
from app.core.paths import BASELINE_MODEL_DIR, EMBEDDING_MODEL_DIR
from app.models.baseline.tfidf_classifier import predict_baseline
from app.preprocessing.cleaner import clean_text
from app.services.classifier_service import predict_embedding

router = APIRouter()


def _load_metrics(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@router.get("", response_model=MetricsResponse)
@router.get("/", response_model=MetricsResponse, include_in_schema=False)
def compare_metrics():
    return MetricsResponse(
        baseline=_load_metrics(BASELINE_MODEL_DIR / "metrics.json"),
        embedding=_load_metrics(EMBEDDING_MODEL_DIR / "metrics.json"),
    )


@router.post("", response_model=ComparisonResult)
@router.post("/", response_model=ComparisonResult, include_in_schema=False)
def compare_methods(request: TextRequest):
    cleaned = clean_text(request.text)
    try:
        baseline = predict_baseline(cleaned)
        embedding = predict_embedding(cleaned)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ComparisonResult(
        text=request.text[:150],
        baseline_prediction=baseline["prediction"],
        baseline_latency_ms=baseline["latency_ms"],
        embedding_prediction=embedding["prediction"],
        embedding_latency_ms=embedding["latency_ms"],
        agree=baseline["prediction"] == embedding["prediction"],
    )
