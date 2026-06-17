from fastapi import APIRouter, HTTPException

from app.api.schemas import ClassifyRequest, ClassifyResponse
from app.models.baseline.tfidf_classifier import predict_baseline
from app.preprocessing.cleaner import clean_text
from app.preprocessing.language_detector import detect_language
from app.services.classifier_service import predict_embedding

router = APIRouter()


@router.post("/", response_model=ClassifyResponse, include_in_schema=False)
def classify(request: ClassifyRequest):
    cleaned = clean_text(request.text)
    lang = request.language or detect_language(request.text)

    try:
        if request.method == "tfidf":
            result = predict_baseline(cleaned)
        else:
            result = predict_embedding(cleaned)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return ClassifyResponse(
        text_preview=request.text[:100] + ("..." if len(request.text) > 100 else ""),
        detected_language=lang,
        category=result["prediction"],
        probabilities=result["probabilities"],
        method=request.method,
        latency_ms=result["latency_ms"],
    )
