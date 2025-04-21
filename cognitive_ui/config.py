# --- LLM Models ---
MAX_TOKENS = 1000

DEFAULT_GEMINI_MODEL = "gemini-1.5-flash-001"
DEFAULT_GEMINI_MAX_TOKENS: int | None = None


# --- Visualization ---
VIZ_NO_DATA: float = -9999  # FIXME
VIZ_NO_DATA_FLOAT = 0.0001
VIZ_PERCENTILES = (0.1, 99.9)
