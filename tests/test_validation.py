"""Tests for input and plan validation.

Tests file path validation, raster validation, parameter validation, and plan validation.
"""

from pathlib import Path

import pytest

from dta.dti.exceptions import ValidationError
from dta.dti.registry import load_registry
from dta.dti.schemas import ExecutionPlan, PlanStep
from dta.dti.validation import InputValidator, PlanValidator


class TestInputValidator:
    """Tests for input validation."""

    def test_validate_file_path_valid(self, tmp_path: Path) -> None:
        """Test file path validation with valid file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")

        validated = InputValidator.validate_file_path(test_file)
        assert validated.exists()

    def test_validate_file_path_not_found(self, tmp_path: Path) -> None:
        """Test file path validation with non-existent file."""
        with pytest.raises(ValidationError, match="does not exist"):
            InputValidator.validate_file_path(tmp_path / "missing.txt")

    def test_validate_raster_path_valid(self, tmp_path: Path) -> None:
        """Test raster path validation with valid extension."""
        raster = tmp_path / "test.tif"
        raster.write_text("fake tif")

        validated = InputValidator.validate_raster_path(raster)
        assert validated.suffix == ".tif"

    def test_validate_raster_path_invalid_extension(self, tmp_path: Path) -> None:
        """Test raster path validation with invalid extension."""
        invalid = tmp_path / "test.txt"
        invalid.write_text("not a raster")

        with pytest.raises(ValidationError, match="Invalid raster extension"):
            InputValidator.validate_raster_path(invalid)

    def test_validate_parameter_valid(self) -> None:
        """Test parameter validation with valid values."""
        InputValidator.validate_parameter(5, "test", expected_type=int, min_value=0, max_value=10)

    def test_validate_parameter_wrong_type(self) -> None:
        """Test parameter validation with wrong type."""
        with pytest.raises(ValidationError, match="must be int"):
            InputValidator.validate_parameter("5", "test", expected_type=int)

    def test_validate_parameter_min_value(self) -> None:
        """Test parameter validation with value below minimum."""
        with pytest.raises(ValidationError, match="must be >= 0"):
            InputValidator.validate_parameter(-5, "test", min_value=0)

    def test_validate_parameter_max_value(self) -> None:
        """Test parameter validation with value above maximum."""
        with pytest.raises(ValidationError, match="must be <= 10"):
            InputValidator.validate_parameter(15, "test", max_value=10)


class TestPlanValidator:
    """Tests for plan validation."""

    def test_validate_plan_valid(self) -> None:
        """Test plan validation with valid plan."""
        registry = load_registry()
        validator = PlanValidator(registry)

        plan = ExecutionPlan(
            flow="test",
            steps=[
                PlanStep(uses="input/file"),
                PlanStep(uses="algorithms/ndvi"),
            ],
            outputs=["publish: chat"],
        )

        validator.validate_plan(plan)  # Should not raise

    def test_validate_plan_empty(self) -> None:
        """Test plan validation with empty plan."""
        registry = load_registry()
        validator = PlanValidator(registry)

        empty_plan = ExecutionPlan(flow="test", steps=[], outputs=[])
        with pytest.raises(ValidationError, match="no steps"):
            validator.validate_plan(empty_plan)

    def test_validate_plan_invalid_component(self) -> None:
        """Test plan validation with invalid component."""
        registry = load_registry()
        validator = PlanValidator(registry)

        plan = ExecutionPlan(
            flow="test",
            steps=[PlanStep(uses="invalid/component")],
            outputs=[],
        )

        with pytest.raises(ValidationError, match="not found in registry"):
            validator.validate_plan(plan)

    def test_check_resources(self) -> None:
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
