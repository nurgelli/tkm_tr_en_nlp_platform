from typing import Literal, Optional

from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=10)
    language: Optional[str] = None
    method: Literal["embedding", "tfidf"] = "embedding"


class ClassifyResponse(BaseModel):
    text_preview: str
    detected_language: str
    category: str
    probabilities: dict
    method: str
    latency_ms: float


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=5)
    top_k: int = Field(5, ge=1, le=20)
    language_filter: Optional[str] = None


class SearchResult(BaseModel):
    text: str
    language: str
    category: str
    similarity: float
    latency_ms: float


class ComparisonResult(BaseModel):
    text: str
    baseline_prediction: str
    baseline_latency_ms: float
    embedding_prediction: str
    embedding_latency_ms: float
    agree: bool


class TextRequest(BaseModel):
    text: str = Field(..., min_length=10)


class MetricsResponse(BaseModel):
    baseline: dict
    embedding: dict
