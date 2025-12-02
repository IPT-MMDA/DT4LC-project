"""Logging configuration for DTA.

Provides structured logging with correlation IDs and performance tracking.
Logs are written to both console and .logs/ directory.
"""

import logging
import logging.config
import logging.handlers
from pathlib import Path
import sys
from typing import Any

# Determine logs directory (project root/.logs/)
_LOGS_DIR = Path(__file__).parent.parent.parent.parent / ".logs"
_LOGS_DIR.mkdir(exist_ok=True)


def setup_logging(level: str = "INFO", format_type: str = "standard") -> None:
    """Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type ("standard" or "json")
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    log_file = str(_LOGS_DIR / "backend.log")

    if format_type == "json":
        # JSON structured logging (for production)
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "class": "logging.Formatter",
                    "format": '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s", "module": "%(module)s", "function": "%(funcName)s", "line": %(lineno)d}',
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": log_level,
                    "formatter": "json",
                    "stream": sys.stdout,
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": log_level,
                    "formatter": "json",
                    "filename": log_file,
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5,
                    "encoding": "utf-8",
                },
            },
            "root": {"level": log_level, "handlers": ["console", "file"]},
            "loggers": {
                "dta": {"level": log_level, "propagate": True},
                "server": {"level": log_level, "propagate": True},
            },
        }
    else:
        # Standard human-readable logging (for development)
        config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": log_level,
                    "formatter": "standard",
                    "stream": sys.stdout,
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": log_level,
                    "formatter": "detailed",
                    "filename": log_file,
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5,
                    "encoding": "utf-8",
                },
            },
            "root": {"level": log_level, "handlers": ["console", "file"]},
            "loggers": {
                "dta": {"level": log_level, "propagate": True},
                "server": {"level": log_level, "propagate": True},
                "uvicorn": {"level": "INFO", "propagate": True},
            },
        }

    logging.config.dictConfig(config)


class CorrelationLogger:
    """Logger with correlation ID support for request tracking."""

    def __init__(self, name: str) -> None:
        """Initialize correlation logger.

        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(name)
        self._correlation_id: str | None = None

    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for current context.

        Args:
            correlation_id: Unique identifier for request/operation
        """
        self._correlation_id = correlation_id

    def clear_correlation_id(self) -> None:
        """Clear correlation ID."""
        self._correlation_id = None

    def _add_correlation(self, msg: str) -> str:
        """Add correlation ID to message.

        Args:
            msg: Log message

        Returns:
            Message with correlation ID
        """
        if self._correlation_id:
            return f"[{self._correlation_id}] {msg}"
        return msg

    def debug(self, msg: str, **kwargs: Any) -> None:
        """Log debug message."""
        self.logger.debug(self._add_correlation(msg), **kwargs)

    def info(self, msg: str, **kwargs: Any) -> None:
        """Log info message."""
        self.logger.info(self._add_correlation(msg), **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        """Log warning message."""
        self.logger.warning(self._add_correlation(msg), **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        """Log error message."""
        self.logger.error(self._add_correlation(msg), **kwargs)

    def critical(self, msg: str, **kwargs: Any) -> None:
        """Log critical message."""
        self.logger.critical(self._add_correlation(msg), **kwargs)


# Initialize default logging
setup_logging()
