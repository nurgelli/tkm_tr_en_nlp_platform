import logging
import sys
from pathlib import Path

from app.core.config import settings


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(level)

    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        Path("logs").mkdir(exist_ok=True)
        file_handler = logging.FileHandler("logs/app.log")
        file_handler.setLevel(logging.WARNING)

        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
