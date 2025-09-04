from __future__ import annotations

import os
from pathlib import Path


def run() -> None:
    """Run the API using uvicorn if available."""

    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover
        raise SystemExit("Uvicorn is required to run the API. Install with `[api]` extras.") from exc

    app_path = "server.app:app_factory"
    host = os.environ.get("DT4LC_HOST", "127.0.0.1")
    port = int(os.environ.get("DT4LC_PORT", "8000"))
    reload = os.environ.get("DT4LC_RELOAD", "false").lower() == "true"

    uvicorn.run(app_path, host=host, port=port, reload=reload, factory=True)


if __name__ == "__main__":
    run()
