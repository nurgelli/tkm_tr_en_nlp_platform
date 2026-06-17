from langdetect import detect, LangDetectException
from langdetect import DetectorFactory
from app.core.logger import get_logger
from app.data_utils import SUPPORTED_LANGUAGES

logger = get_logger(__name__)

DetectorFactory.seed = 42

def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        return lang if lang in SUPPORTED_LANGUAGES else "other"
    except LangDetectException:
        return "unknown"

def validate_language(text: str, expected_lang: str) -> bool:
    detected = detect_language(text)
    if expected_lang == "tk" and detected == "tr":
        return True
    return detected == expected_lang
