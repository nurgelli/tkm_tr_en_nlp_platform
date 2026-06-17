import pandas as pd

SUPPORTED_LANGUAGE_CODES = ("tr", "tk", "en")
SUPPORTED_LANGUAGES = {
    "tr": "Turkish",
    "tk": "Turkmen",
    "en": "English",
}


def normalize_corpus_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "language" not in df.columns:
        for candidate in ("language_final", "language_original", "language_detected"):
            if candidate in df.columns:
                df["language"] = df[candidate]
                break
    if "source" not in df.columns:
        df["source"] = "unknown"
    required = {"text", "text_clean", "language", "category"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Corpus is missing required columns: {', '.join(missing)}")
    df["text"] = df["text"].fillna("").astype(str)
    df["text_clean"] = df["text_clean"].fillna("").astype(str)
    df["language"] = df["language"].fillna("unknown").astype(str).str.lower()
    df["category"] = df["category"].fillna("unknown").astype(str)
    df["source"] = df["source"].fillna("unknown").astype(str)
    return df
