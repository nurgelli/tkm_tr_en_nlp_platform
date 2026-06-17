import re
import unicodedata
import pandas as pd
from html import unescape
from app.core.logger import get_logger

logger = get_logger(__name__)

def remove_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", unescape(text))

def remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", " ", text)

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)

def clean_text(text: str, lowercase: bool = True) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    text = remove_html(text)
    text = remove_urls(text)
    text = normalize_unicode(text)

    text = re.sub(r"[^\w\s\-\.,!\?]", " ", text, flags=re.UNICODE)

    if lowercase:
        text = text.lower()

    text = normalize_whitespace(text)
    return text


def clean_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    df = df.copy()
    df[f"{text_col}_clean"] = df[text_col].apply(clean_text)

    original_len = len(df)
    df = df[df[f"{text_col}_clean"].str.len() > 20].reset_index(drop=True)

    logger.info(f"Cleaned: {original_len} → {len(df)} texts ({original_len-len(df)} removed)")
    return df
