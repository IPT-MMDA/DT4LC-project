"""Centralized logging configuration for the server package."""

import logging

_configured = False


def configure_logging() -> None:
    global _configured

    if _configured:
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    _configured = True
