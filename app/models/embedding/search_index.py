import numpy as np
import pandas as pd

from app.core.logger import get_logger
from app.core.paths import EMBEDDING_MODEL_DIR
from app.data_utils import SUPPORTED_LANGUAGE_CODES, normalize_corpus_schema
from app.models.embedding.embedder import embed_single, embed_texts

logger = get_logger(__name__)

EMBEDDINGS_PATH = EMBEDDING_MODEL_DIR / "embeddings.npy"
METADATA_PATH = EMBEDDING_MODEL_DIR / "index_metadata.csv"


def build_search_index(df: pd.DataFrame) -> dict:
    df = normalize_corpus_schema(df)
    logger.info(f"Building search index for {len(df)} texts...")

    texts = df["text_clean"].tolist()
    embeddings = embed_texts(texts)

    EMBEDDING_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    df[["text", "text_clean", "language", "category", "source"]].to_csv(
        METADATA_PATH, index=False
    )

    logger.info(f"Search index built: {embeddings.shape}")
    return {"texts": len(texts), "dimensions": embeddings.shape[1]}


def cosine_search(query: str, top_k: int = 5, lang_filter: str = None) -> list:
    if not EMBEDDINGS_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Search index not found. Run scripts/build_embeddings.py first."
        )
    embeddings = np.load(EMBEDDINGS_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    if lang_filter and lang_filter in SUPPORTED_LANGUAGE_CODES:
        mask = metadata["language"] == lang_filter
        embeddings = embeddings[mask]
        metadata = metadata[mask].reset_index(drop=True)
        if metadata.empty:
            return []

    query_vec, latency_ms = embed_single(query)

    scores = np.dot(embeddings, query_vec)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append(
            {
                "text": str(metadata.iloc[idx]["text"])[:200] + "...",
                "language": metadata.iloc[idx]["language"],
                "category": metadata.iloc[idx]["category"],
                "similarity": round(float(scores[idx]), 4),
                "latency_ms": round(latency_ms, 3),
            }
        )

    return results
