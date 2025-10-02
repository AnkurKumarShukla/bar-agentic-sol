# src/agentic_backend/config.py
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Agentic Backend"

    # Keys
    OPENAI_API_KEY: str | None = None
    SERPAPI_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None
    GOOGLE_CX: str | None = None

    # Persistence
    RUN_SAVE_DIR: Path = Path("runs")  # default folder

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

