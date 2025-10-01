"""Validation layer for DTA.

Validates plans, inputs, and resources before execution.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dta.dti.registry import Registry
    from dta.dti.schemas import ExecutionPlan, PlanStep

from .exceptions import ResourceError, ValidationError

logger = logging.getLogger(__name__)


class PlanValidator:
    """Validates execution plans."""

    def __init__(self, registry: Registry) -> None:
        """Initialize validator.

        Args:
            registry: Component registry
        """
        self.registry = registry

    def validate_plan(self, plan: ExecutionPlan) -> None:
        """Validate execution plan.

        Args:
            plan: Plan to validate

        Raises:
            ValidationError: If plan is invalid
        """
        if not plan.steps:
            raise ValidationError("Plan has no steps")

        # Validate each step
        for i, step in enumerate(plan.steps):
            self._validate_step(step, i)

        # Validate type flow
        self._validate_type_flow(plan)

        logger.info(f"Plan validated: {len(plan.steps)} steps")

    def _validate_step(self, step: PlanStep, index: int) -> None:
        """Validate a single step.

        Args:
            step: Step to validate
            index: Step index

        Raises:
            ValidationError: If step is invalid
        """
        from dta.dti.registry import get_item

        # Check component exists
        try:
            item = get_item(self.registry, step.uses)
        except KeyError as e:
            raise ValidationError(f"Step {index}: component '{step.uses}' not found in registry") from e

        # Validate runner type
        if not item.runner:
            raise ValidationError(f"Step {index}: component '{step.uses}' has no runner")

        valid_runners = ["python", "agent", "passthrough"]
        if item.runner.type not in valid_runners:
            raise ValidationError(
                f"Step {index}: invalid runner type '{item.runner.type}'. Must be one of: {', '.join(valid_runners)}"
            )

        # Validate python runner has entrypoint
        if item.runner.type == "python" and not item.runner.entrypoint:
            raise ValidationError(f"Step {index}: python runner requires entrypoint")

    def _validate_type_flow(self, plan: ExecutionPlan) -> None:
        """Validate data type flow through pipeline.

        Args:
            plan: Plan to validate

        Raises:
            ValidationError: If type flow is invalid
        """
        from dta.dti.registry import get_item

        # Track available types
        available: set[str] = set()

        for i, step in enumerate(plan.steps):
            item = get_item(self.registry, step.uses)

            # Check required inputs are available
            for required_input in item.inputs:
                if required_input and required_input not in available:
                    logger.warning(
                        f"Step {i} ({step.uses}) requires '{required_input}' "
                        f"which may not be available yet. Available: {available}"
                    )

            # Add outputs to available set
            available.update(item.outputs)

        logger.debug(f"Type flow validated. Final available types: {available}")

    def check_resources(self, plan: ExecutionPlan) -> dict[str, Any]:
        """Check resource requirements for plan.

        Args:
            plan: Plan to check

        Returns:
            Dictionary with resource estimates

        Raises:
            ResourceError: If resources exceed limits
        """
        from dta.dti.registry import get_item

        # Resource limits (configurable)
        MAX_STEPS = 50
        MAX_ESTIMATED_TIME = 600  # 10 minutes
        MAX_MEMORY_MB = 4096  # 4GB

        if len(plan.steps) > MAX_STEPS:
            raise ResourceError(f"Plan has {len(plan.steps)} steps, exceeds limit of {MAX_STEPS}")

        # Estimate resources
        estimated_time = 0
        estimated_memory = 0

        for step in plan.steps:
            item = get_item(self.registry, step.uses)

            # Rough estimates based on component type
            if item.kind == "algorithm":
                estimated_time += 5  # 5 seconds per algorithm
                estimated_memory += 256  # 256 MB
            elif item.kind == "model":
                estimated_time += 30  # 30 seconds per model
                estimated_memory += 1024  # 1 GB
            else:
                estimated_time += 1
                estimated_memory += 50

        if estimated_time > MAX_ESTIMATED_TIME:
            raise ResourceError(f"Estimated execution time {estimated_time}s exceeds limit of {MAX_ESTIMATED_TIME}s")

        if estimated_memory > MAX_MEMORY_MB:
            raise ResourceError(f"Estimated memory usage {estimated_memory}MB exceeds limit of {MAX_MEMORY_MB}MB")

        return {
            "steps": len(plan.steps),
            "estimated_time_seconds": estimated_time,
            "estimated_memory_mb": estimated_memory,
        }


class InputValidator:
    """Validates inputs."""

    @staticmethod
    def validate_file_path(path: str | Path, must_exist: bool = True) -> Path:
        """Validate file path.

        Args:
            path: Path to validate
            must_exist: Whether file must exist

        Returns:
            Validated Path object

        Raises:
            ValidationError: If path is invalid
        """
        try:
            path_obj = Path(path)

            if must_exist and not path_obj.exists():
                raise ValidationError(f"File does not exist: {path}")

            if must_exist and not path_obj.is_file():
                raise ValidationError(f"Path is not a file: {path}")

            return path_obj

        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Invalid path: {path}. Error: {e}") from e

    @staticmethod
    def validate_raster_path(path: str | Path) -> Path:
        """Validate raster file path.

        Args:
            path: Raster path

        Returns:
            Validated Path object

        Raises:
            ValidationError: If not a valid raster
        """
        path_obj = InputValidator.validate_file_path(path, must_exist=True)

        # Check extension
        valid_extensions = [".tif", ".tiff", ".nc", ".hdf", ".h5"]
        if path_obj.suffix.lower() not in valid_extensions:
            raise ValidationError(
                f"Invalid raster extension '{path_obj.suffix}'. Must be one of: {', '.join(valid_extensions)}"
            )

        return path_obj

    @staticmethod
    def validate_parameter(
        param: Any,
        param_name: str,
        expected_type: type | None = None,
        min_value: float | None = None,
        max_value: float | None = None,
    ) -> None:
        """Validate parameter value.

        Args:
            param: Parameter value
            param_name: Parameter name (for error messages)
            expected_type: Expected type
            min_value: Minimum value (for numeric params)
            max_value: Maximum value (for numeric params)

        Raises:
            ValidationError: If parameter is invalid
        """
        if expected_type and not isinstance(param, expected_type):
            raise ValidationError(
                f"Parameter '{param_name}' must be {expected_type.__name__}, got {type(param).__name__}"
            )

        if min_value is not None and param < min_value:
            raise ValidationError(f"Parameter '{param_name}' must be >= {min_value}, got {param}")

        if max_value is not None and param > max_value:
            raise ValidationError(f"Parameter '{param_name}' must be <= {max_value}, got {param}")
