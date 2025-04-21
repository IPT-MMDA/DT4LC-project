#!/usr/bin/env python3
"""
Utility functions for Cognitive Digital Twin Interactive UI

This module contains general utility functions used throughout the
Cognitive Digital Twin application.
"""

import os
from typing import Any


def debug_info(message: str, data: Any = None) -> None:
    """Write debug information to console.

    Args:
        message: Debug message
        data: Optional data to print
    """
    # Enable debug mode with an environment variable
    if os.environ.get("CDT_DEBUG", "").lower() in ("1", "true", "yes"):
        print(f"DEBUG: {message}")
        if data is not None:
            if isinstance(data, (str, int, float, bool)):
                print(f"       {data}")
            else:
                try:
                    print(f"       {type(data)}: {str(data)}")
                except Exception:
                    print(f"       {type(data)} (cannot convert to string)")


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to a maximum length, respecting sentence boundaries.

    Args:
        text: Text to truncate
        max_length: Maximum length in characters

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text

    # Find the last period before max_length
    last_period = text[:max_length].rfind(".")
    if last_period > 0:
        return text[: last_period + 1]

    # If no period found, truncate at max_length
    return text[:max_length] + "..."
