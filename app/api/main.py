from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import classify, compare, languages, search
from app.core.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from app.models.embedding.embedder import load_model

        load_model()
        logger.info("Multilingual embedding model ready")
    except Exception as exc:
        logger.warning("Embedding model preload failed: %s", exc)
    yield


app = FastAPI(lifespan=lifespan, title="Multilingual NLP Platform", version="1.0.0")

app.include_router(classify.router, prefix="/classify", tags=["Classification"])
app.include_router(search.router, prefix="/search", tags=["Semantic Search"])
app.include_router(compare.router, prefix="/compare", tags=["Model Comparison"])
app.include_router(languages.router, prefix="/languages", tags=["Languages"])


@app.get("/health")
def health():
    return {"status": "healthy"}
