"""LLM Router Dependency Injection.

Provides a centralized, cached LLM router instance using @lru_cache.
This replaces the scattered global _router pattern across multiple modules.
"""

from functools import lru_cache

from .config import create_router_from_env
from .router import LLMRouter


@lru_cache(maxsize=1)
def get_llm_router() -> LLMRouter:
    """Get or create the shared LLM router instance.

    Uses @lru_cache to ensure only one router instance is created
    and reused across all consumers.

    Returns:
        Configured LLM router with fallback chain
    """
    return create_router_from_env()


def reset_llm_router() -> None:
    """Reset the cached router instance.

    Useful for testing or when environment configuration changes.
    The next call to get_llm_router() will create a new instance.
    """
    get_llm_router.cache_clear()
