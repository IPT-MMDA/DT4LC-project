"""Tests for logging infrastructure.

Tests logging setup and correlation logger.
"""

from dta.dti.logging_config import CorrelationLogger, setup_logging


class TestLoggingSetup:
    """Tests for logging setup."""

    def test_setup_logging_standard(self) -> None:
        """Test standard logging setup."""
        setup_logging(level="INFO", format_type="standard")

    def test_setup_logging_json(self) -> None:
        """Test JSON logging setup."""
        setup_logging(level="DEBUG", format_type="json")


class TestCorrelationLogger:
    """Tests for correlation logger."""

    def test_basic_logging(self) -> None:
        """Test basic logging without correlation ID."""
        logger = CorrelationLogger("test")
        logger.info("Test message")

    def test_logging_with_correlation_id(self) -> None:
        """Test logging with correlation ID."""
        logger = CorrelationLogger("test")

        logger.set_correlation_id("test-123")
        logger.info("Test with ID")
        logger.debug("Debug message")
        logger.warning("Warning message")
        logger.error("Error message")

    def test_clear_correlation_id(self) -> None:
        """Test clearing correlation ID."""
        logger = CorrelationLogger("test")

        logger.set_correlation_id("test-123")
        logger.clear_correlation_id()
        logger.info("After clear")
