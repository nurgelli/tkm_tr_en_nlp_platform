from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# BASE_DIR = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    api_host: str = Field("0.0.0.0", alias="API_HOST")
    api_port: int = Field(8000, alias="API_PORT")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    model_dir: Path = Field(Path("app/models"), alias="MODEL_DIR")
    data_path: Path = Field(
        Path("data/processed/corpus_cleaned.csv"), alias="DATA_PATH"
    )
    api_base_url: str = Field("http://fastapi:8000", alias="API_BASE_URL")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    # model_cache_dir: Path = BASE_DIR / ".cache" / "huggingface"


settings = Settings()
