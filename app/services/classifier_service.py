import joblib

from app.core.paths import EMBEDDING_MODEL_DIR

MODEL_PATH = EMBEDDING_MODEL_DIR / "embedding_classifier.joblib"
_embedding_classifier = None


def load_embedding_classifier():
    global _embedding_classifier
    if _embedding_classifier is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Embedding classifier not found: {MODEL_PATH}. Run scripts/build_embeddings.py first."
            )
        _embedding_classifier = joblib.load(MODEL_PATH)
    return _embedding_classifier


def predict_embedding(text: str):
    from app.models.embedding.embedder import embed_single

    clf = load_embedding_classifier()
    vec, latency = embed_single(text)
    pred = clf.predict([vec])[0]
    proba = clf.predict_proba([vec])[0]
    return {
        "prediction": pred,
        "probabilities": dict(zip(clf.classes_, [round(float(p), 4) for p in proba])),
        "latency_ms": round(latency, 3),
    }
