import json
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.core.logger import get_logger
from app.core.paths import BASELINE_MODEL_DIR
from app.data_utils import normalize_corpus_schema

logger = get_logger(__name__)

MODEL_PATH = BASELINE_MODEL_DIR / "tfidf_pipeline.joblib"
METRICS_PATH = BASELINE_MODEL_DIR / "metrics.json"
_pipeline = None


def train_baseline(df: pd.DataFrame) -> dict:
    df = normalize_corpus_schema(df)
    X = df["text_clean"]
    y = df["category"]

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=10000, ngram_range=(1, 2), sublinear_tf=True, min_df=1
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
        ]
    )

    start = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start

    start = time.time()
    y_pred = pipeline.predict(X_test)
    inference_time_ms = (time.time() - start) / len(X_test) * 1000

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    BASELINE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    logger.info(f"Baseline accuracy: {accuracy:.4f}")

    metrics = {
        "model": "TF-IDF + Logistic Regression",
        "accuracy": round(accuracy, 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "train_time_sec": round(train_time, 2),
        "inference_ms": round(inference_time_ms, 3),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_baseline_model():
    global _pipeline
    if _pipeline is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Baseline model not found: {MODEL_PATH}. Run scripts/train_baseline.py first."
            )
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


def predict_baseline(text: str) -> dict:
    pipeline = load_baseline_model()
    start = time.time()
    prediction = pipeline.predict([text])[0]
    proba = pipeline.predict_proba([text])[0]
    latency_ms = (time.time() - start) * 1000

    classes = pipeline.classes_
    return {
        "prediction": prediction,
        "probabilities": dict(zip(classes, [round(p, 4) for p in proba])),
        "latency_ms": round(latency_ms, 3),
    }
