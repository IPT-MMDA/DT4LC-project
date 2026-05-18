"""Custom exceptions for DTA (Digital Twin Aggregator).

Provides clear, specific error types for different failure modes.
"""


class DTAException(Exception):  # noqa: N818  # historical name; subclasses use Error suffix
    """Base exception for all DTA errors."""

    pass


class PlanningError(DTAException):
    """Error during pipeline planning phase."""

    pass


class ExecutionError(DTAException):
    """Error during pipeline execution."""

    pass


class ValidationError(DTAException):
    """Error during input validation."""

    pass


class ResourceError(DTAException):
    """Error related to resource constraints (memory, disk, GPU)."""

    pass


class ModelError(DTAException):
    """Error related to model loading or inference."""

    pass


class DataError(DTAException):
    """Error related to data loading or processing."""

    pass


class RegistryError(DTAException):
    """Error related to registry operations."""

    pass


class LLMError(DTAException):
    """Error related to LLM operations."""

    pass
