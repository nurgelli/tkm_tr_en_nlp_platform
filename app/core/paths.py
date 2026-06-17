from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
APP_DIR = ROOT_DIR / "app"
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = APP_DIR / "models"
BASELINE_MODEL_DIR = MODEL_DIR / "baseline"
EMBEDDING_MODEL_DIR = MODEL_DIR / "embedding"
PROCESSED_CORPUS_PATH = DATA_DIR / "processed" / "corpus_cleaned.csv"
