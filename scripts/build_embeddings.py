import json
import time

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from app.core.paths import EMBEDDING_MODEL_DIR
from app.data_utils import normalize_corpus_schema
from app.models.embedding.embedder import embed_texts
from app.models.embedding.search_index import build_search_index


def train_embedding_classifier(df: pd.DataFrame) -> dict:
    df = normalize_corpus_schema(df)
    X = df["text_clean"].tolist()
    y = df["category"]
    embeddings = embed_texts(X)

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y, test_size=0.2, random_state=42, stratify=stratify
    )

    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    start = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - start

    start = time.time()
    y_pred = clf.predict(X_test)
    inference_time_ms = (time.time() - start) / len(X_test) * 1000
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)

    EMBEDDING_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, EMBEDDING_MODEL_DIR / "embedding_classifier.joblib")

    metrics = {
        "model": "SentenceTransformer + Logistic Regression",
        "accuracy": round(accuracy, 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "train_time_sec": round(train_time, 2),
        "inference_ms": round(inference_time_ms, 3),
    }
    (EMBEDDING_MODEL_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return metrics


if __name__ == "__main__":
    df = pd.read_csv("data/processed/corpus_cleaned.csv")
    index_result = build_search_index(df)
    metrics = train_embedding_classifier(df)
    print({"index": index_result, "classifier": metrics})
