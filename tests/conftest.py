from pathlib import Path

from dotenv import load_dotenv

# Resolve repo root via your config helper if available; else fall back.
try:
    from dta.config import ROOT_DIR

    ROOT = Path(ROOT_DIR)
except Exception:
    # repo root assumed = parent of tests/
    ROOT = Path(__file__).resolve().parents[1]

load_dotenv(ROOT / ".env")
