__all__ = ["Settings", "get_settings"]

from functools import lru_cache
import os

from dotenv import load_dotenv

from .defaults import CORS_ORIGINS, GEMINI_MODEL, HTTP_TIMEOUT

load_dotenv()  # loads .env from project root


class Settings:
    gemini_api_key: str
    gemini_model: str
    http_timeout: float
    cors_origins: list[str]

    def __init__(self) -> None:
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.gemini_model = os.getenv("GEMINI_MODEL", GEMINI_MODEL)
        self.http_timeout = float(os.getenv("HTTP_TIMEOUT", HTTP_TIMEOUT))
        self.cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o] or CORS_ORIGINS


@lru_cache
def get_settings() -> Settings:
    return Settings()
