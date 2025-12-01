"""CLI entry point for the DT4LC API server."""

import os


def run() -> None:
    """Run the FastAPI server using uvicorn."""
    import uvicorn

    host = os.getenv("DT4LC_HOST", "0.0.0.0")
    port = int(os.getenv("DT4LC_PORT", "8000"))
    reload = os.getenv("DT4LC_RELOAD", "").lower() in ("1", "true", "yes")
    log_level = os.getenv("DT4LC_LOG_LEVEL", "info").lower()

    uvicorn.run(
        "server.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )


if __name__ == "__main__":
    run()
