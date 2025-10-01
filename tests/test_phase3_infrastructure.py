"""Phase 3 Tests - Infrastructure (Logging, Metrics, Cache, Models)."""

from pathlib import Path
import time

import numpy as np
import pytest

from dta.dti.cache import LRUCache, ResultCache, get_result_cache
from dta.dti.exceptions import ValidationError
from dta.dti.logging_config import CorrelationLogger, setup_logging
from dta.dti.metrics import ExecutionMetrics, LLMMetrics, MetricsCollector, get_metrics_collector
from dta.dti.models.prithvi import PrithviModel
from dta.dti.models.registry import ModelRegistry, get_model_registry
from dta.dti.registry import load_registry
from dta.dti.schemas import ExecutionPlan, PlanStep
from dta.dti.validation import InputValidator, PlanValidator


# Logging Tests
def test_setup_logging_standard() -> None:
    """Test standard logging setup."""
    setup_logging(level="INFO", format_type="standard")
    # Should not raise


def test_setup_logging_json() -> None:
    """Test JSON logging setup."""
    setup_logging(level="DEBUG", format_type="json")
    # Should not raise


def test_correlation_logger() -> None:
    """Test correlation logger."""
    logger = CorrelationLogger("test")

    # No correlation ID
    logger.info("Test message")

    # With correlation ID
    logger.set_correlation_id("test-123")
    logger.info("Test with ID")
    logger.debug("Debug message")
    logger.warning("Warning message")
    logger.error("Error message")

    # Clear
    logger.clear_correlation_id()
    logger.info("After clear")


# Metrics Tests
def test_execution_metrics() -> None:
    """Test execution metrics."""
    metrics = ExecutionMetrics(
        plan_id="test-plan",
        start_time=time.time(),
        steps_total=5,
    )

    assert metrics.plan_id == "test-plan"
    assert metrics.status == "running"
    assert metrics.steps_completed == 0
    assert metrics.duration > 0  # Should have some duration


def test_llm_metrics() -> None:
    """Test LLM metrics."""
    metrics = LLMMetrics(
        provider="gemini",
        model="gemini-2.0-flash-exp",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        cost=0.001,
        duration=2.5,
    )

    assert metrics.provider == "gemini"
    assert metrics.total_tokens == 150
    assert metrics.cost == 0.001


def test_metrics_collector() -> None:
    """Test metrics collector."""
    collector = MetricsCollector()

    # Start execution
    collector.start_execution("plan-1", steps_total=3)
    assert "plan-1" in collector.executions

    # Update progress
    collector.update_execution("plan-1", steps_completed=2)
    assert collector.executions["plan-1"].steps_completed == 2

    # Complete
    collector.complete_execution("plan-1", status="success")
    assert collector.executions["plan-1"].status == "success"

    # Record LLM call
    collector.record_llm_call(
        provider="gemini",
        model="gemini-2.0-flash-exp",
        prompt_tokens=100,
        completion_tokens=50,
        cost=0.001,
        duration=1.5,
    )

    # Get stats
    stats = collector.get_stats()
    assert stats.total_executions == 1
    assert stats.successful_executions == 1
    assert stats.total_llm_calls == 1
    assert stats.total_llm_tokens == 150


def test_metrics_collector_global() -> None:
    """Test global metrics collector."""
    collector = get_metrics_collector()
    collector.clear()

    collector.start_execution("test", 1)
    collector.complete_execution("test", "success")

    stats = collector.get_stats()
    assert stats.total_executions >= 1


# Cache Tests
def test_lru_cache() -> None:
    """Test LRU cache."""
    cache = LRUCache(max_size=3, default_ttl=10)

    # Set values
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    cache.set("key3", "value3")

    # Get values
    assert cache.get("key1") == "value1"
    assert cache.get("key2") == "value2"
    assert cache.get("key3") == "value3"

    # Add one more (should evict key1)
    cache.set("key4", "value4")
    assert cache.get("key1") is None  # Evicted
    assert cache.get("key4") == "value4"

    # Check size
    assert cache.size == 3


def test_lru_cache_ttl() -> None:
    """Test cache TTL expiration."""
    cache = LRUCache(max_size=10, default_ttl=1)

    cache.set("key", "value", ttl=1)
    assert cache.get("key") == "value"

    # Wait for expiration
    time.sleep(1.1)
    assert cache.get("key") is None  # Expired


def test_lru_cache_stats() -> None:
    """Test cache statistics."""
    cache = LRUCache(max_size=10)

    cache.set("key1", "value1")
    cache.get("key1")  # Hit
    cache.get("key2")  # Miss

    stats = cache.get_stats()
    assert stats["size"] == 1
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5


def test_result_cache() -> None:
    """Test result cache."""
    cache = ResultCache(max_size=10)

    # Generate key
    key = cache.generate_key("algorithm/ndvi", {"raster": "test.tif"})
    assert isinstance(key, str)
    assert len(key) == 16  # SHA256 hash truncated

    # Cache result
    result = {"ndvi": np.array([0.5, 0.6, 0.7])}
    cache.set("algorithm/ndvi", {"raster": "test.tif"}, result)

    # Retrieve
    cached = cache.get("algorithm/ndvi", {"raster": "test.tif"})
    assert cached is not None
    assert "ndvi" in cached


def test_result_cache_global() -> None:
    """Test global result cache."""
    cache = get_result_cache()
    cache.clear()

    cache.set("test", {"input": 1}, {"output": 2})
    result = cache.get("test", {"input": 1})
    assert result == {"output": 2}


# Validation Tests
def test_input_validator_file_path(tmp_path: Path) -> None:
    """Test file path validation."""
    # Valid file
    test_file = tmp_path / "test.txt"
    test_file.write_text("test")

    validated = InputValidator.validate_file_path(test_file)
    assert validated.exists()

    # Non-existent file
    with pytest.raises(ValidationError, match="does not exist"):
        InputValidator.validate_file_path(tmp_path / "missing.txt")


def test_input_validator_raster_path(tmp_path: Path) -> None:
    """Test raster path validation."""
    # Valid raster
    raster = tmp_path / "test.tif"
    raster.write_text("fake tif")

    validated = InputValidator.validate_raster_path(raster)
    assert validated.suffix == ".tif"

    # Invalid extension
    invalid = tmp_path / "test.txt"
    invalid.write_text("not a raster")

    with pytest.raises(ValidationError, match="Invalid raster extension"):
        InputValidator.validate_raster_path(invalid)


def test_input_validator_parameter() -> None:
    """Test parameter validation."""
    # Valid
    InputValidator.validate_parameter(5, "test", expected_type=int, min_value=0, max_value=10)

    # Wrong type
    with pytest.raises(ValidationError, match="must be int"):
        InputValidator.validate_parameter("5", "test", expected_type=int)

    # Out of range
    with pytest.raises(ValidationError, match="must be >= 0"):
        InputValidator.validate_parameter(-5, "test", min_value=0)

    with pytest.raises(ValidationError, match="must be <= 10"):
        InputValidator.validate_parameter(15, "test", max_value=10)


def test_plan_validator() -> None:
    """Test plan validation."""
    registry = load_registry()
    validator = PlanValidator(registry)

    # Valid plan
    plan = ExecutionPlan(
        flow="test",
        steps=[
            PlanStep(uses="input/file"),
            PlanStep(uses="algorithms/ndvi"),
        ],
        outputs=["publish: chat"],
    )

    validator.validate_plan(plan)  # Should not raise

    # Empty plan
    empty_plan = ExecutionPlan(flow="test", steps=[], outputs=[])
    with pytest.raises(ValidationError, match="no steps"):
        validator.validate_plan(empty_plan)


def test_plan_validator_invalid_component() -> None:
    """Test plan validation with invalid component."""
    registry = load_registry()
    validator = PlanValidator(registry)

    # Invalid component
    plan = ExecutionPlan(
        flow="test",
        steps=[PlanStep(uses="invalid/component")],
        outputs=[],
    )

    with pytest.raises(ValidationError, match="not found in registry"):
        validator.validate_plan(plan)


def test_plan_validator_resources() -> None:
    """Test resource checking."""
    registry = load_registry()
    validator = PlanValidator(registry)

    plan = ExecutionPlan(
        flow="test",
        steps=[PlanStep(uses="algorithms/ndvi")],
        outputs=[],
    )

    resources = validator.check_resources(plan)
    assert "steps" in resources
    assert "estimated_time_seconds" in resources
    assert "estimated_memory_mb" in resources
    assert resources["steps"] == 1


# Model Tests
def test_prithvi_model() -> None:
    """Test Prithvi model wrapper."""
    model = PrithviModel()

    assert model.name == "prithvi"
    assert model.version == "v1.0"
    assert "Raster" in model.required_inputs
    assert "Features" in model.outputs
    assert model.is_available()


def test_prithvi_prediction() -> None:
    """Test Prithvi prediction."""
    model = PrithviModel()

    result = model.predict({"raster_path": "test.tif"})

    assert "features" in result
    assert "embeddings" in result
    assert "model" in result
    assert result["model"] == "prithvi"


def test_prithvi_extract_features() -> None:
    """Test feature extraction."""
    model = PrithviModel()

    raster = np.random.rand(10, 10, 6)
    features = model.extract_features(raster)

    assert isinstance(features, np.ndarray)
    assert len(features) > 0


def test_model_registry() -> None:
    """Test model registry."""
    registry = ModelRegistry()

    # Register model
    model = PrithviModel()
    registry.register(model, metadata={"gpu_required": False})

    # Get model
    retrieved = registry.get("prithvi")
    assert retrieved.name == "prithvi"

    # List available
    available = registry.list_available()
    assert len(available) > 0

    # Check requirements
    reqs = registry.check_requirements("prithvi:v1.0")
    assert reqs["name"] == "prithvi"
    assert reqs["available"] is True


def test_model_registry_global() -> None:
    """Test global model registry."""
    registry = get_model_registry()

    # Should have Prithvi registered by default
    available = registry.list_available()
    assert any("prithvi" in m for m in available)
