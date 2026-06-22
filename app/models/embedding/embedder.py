import os
import time

import numpy as np
from dotenv import load_dotenv

from app.core.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

HF_TOKEN = os.getenv("HF_TOKEN")

_model = None


def load_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME, token=HF_TOKEN)
        logger.info("Embedding model loaded")
    return _model


def embed_texts(texts: list, batch_size: int = 64) -> np.ndarray:
    model = load_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
        normalize_embeddings=True,
    )
    return embeddings


def embed_single(text: str) -> tuple:
    model = load_model()
    start = time.time()
    embedding = model.encode([text], normalize_embeddings=True)
    latency_ms = (time.time() - start) * 1000
    return embedding[0], latency_ms
