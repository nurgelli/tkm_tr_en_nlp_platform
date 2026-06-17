import argparse
from pathlib import Path
import pandas as pd
from app.core.logger import get_logger
from app.preprocessing.cleaner import clean_dataframe
from app.preprocessing.language_detector import detect_language
from app.data_utils import normalize_corpus_schema

logger = get_logger(__name__)


def preprocess(input_csv: str, output_csv: str) -> pd.DataFrame:
    input_path = Path(input_csv)
    output_path = Path(output_csv)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info(f"Loading raw data from: {input_path}")
    df = pd.read_csv(input_path)

    if "text" not in df.columns:
        logger.error("Input CSV must contain a 'text' column")
        raise ValueError("Input CSV must contain a 'text' column")

    # Clean texts and remove short / empty rows
    df_clean = clean_dataframe(df, text_col="text")

    # Detect language on the original text (not the cleaned one) to preserve metadata
    logger.info("Detecting languages for cleaned corpus...")
    df_clean["language_detected"] = df_clean["text"].apply(lambda t: detect_language(t))

    # Preserve original language column if present and compute a final language column
    if "language" in df_clean.columns:
        df_clean["language_original"] = df_clean["language"]
        df_clean["language_final"] = df_clean["language_original"].fillna(df_clean["language_detected"])
        # Mark as valid if detected matches original (with turkmen->turkish tolerance)
        def _valid(row):
            orig = str(row.get("language_original", "")).lower()
            det = str(row.get("language_detected", "")).lower()
            if orig == "tk" and det == "tr":
                return True
            return orig == det
        df_clean["language_valid"] = df_clean.apply(_valid, axis=1)
    else:
        df_clean["language_final"] = df_clean["language_detected"]

    # Ensure output folder exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Select sensible output columns
    out_cols = [c for c in ["text", "text_clean", "language_final", "language_detected", "language_original", "language_valid", "category", "source"] if c in df_clean.columns]

    df_clean.to_csv(output_path, index=False, columns=out_cols)
    df_normalized = normalize_corpus_schema(df_clean)
    df_normalized.to_csv(output_path, index=False)
    logger.info(f"Processed corpus saved to: {output_path} ({len(df_clean)} rows)")

    return df_normalized


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw multilingual corpus CSV")
    parser.add_argument("--input", "-i", default="data/raw/multilingual_corpus.csv", help="Path to raw CSV")
    parser.add_argument("--output", "-o", default="data/processed/corpus_cleaned.csv", help="Path to write processed CSV")

    args = parser.parse_args()

    try:
        df = preprocess(args.input, args.output)
        print(f"Done. Processed {len(df)} rows -> {args.output}")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise
