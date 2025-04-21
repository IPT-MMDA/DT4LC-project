#!/usr/bin/env python3
"""
Command-line interface for the Cognitive Digital Twin UI.

This module provides command-line utilities for launching the Streamlit-based
Cognitive Digital Twin UI application with appropriate configuration.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def run_app() -> int:
    """Launch the Streamlit app with proper configuration.

    This is the entry point defined in pyproject.toml's [project.scripts].
    It parses command-line arguments and launches the Streamlit application
    with the appropriate settings.

    Returns:
        int: Exit code (0 for success, non-zero for errors)
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the Cognitive Digital Twin UI")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    args = parser.parse_args()

    # If --version flag is set, just print version and exit
    if args.version:
        from importlib.metadata import version

        try:
            ver = version("dt4lc-project")
            print(f"Cognitive Digital Twin UI version: {ver}")
            return 0
        except Exception as e:
            print(f"Could not determine version: {e}")
            return 1

    # Set up path to project root
    project_root = Path(__file__).parent.parent.absolute()

    try:
        # Define app path and verify it exists
        app_path = Path(__file__).parent / "app.py"
        if not app_path.exists():
            print(f"Error: App file not found at {app_path}")
            return 1

        # Print status information
        print("=== Starting Cognitive Digital Twin UI ===")
        print(f"App path: {app_path}")
        print(f"Project root: {project_root}")
        print(f"Checking if .streamlit exists: {(project_root / '.streamlit').exists()}")
        print("Streamlit settings loaded from .streamlit/config.toml")

        # Run the Streamlit app with the constructed arguments
        streamlit_cmd = get_streamlit_args(app_path)
        subprocess.run(streamlit_cmd, check=True, cwd=str(project_root))
        return 0

    except KeyboardInterrupt:
        print("\nApp stopped by user.")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\nStreamlit process error: {e}")
        return e.returncode
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback

        traceback.print_exc()
        return 1


def get_streamlit_args(app_path: Path) -> List[str]:
    """Build the Streamlit command line arguments.

    Args:
        app_path: Path to the app.py file

    Returns:
        List[str]: Command line arguments for Streamlit
    """
    # Base arguments for the streamlit CLI
    args = ["streamlit", "run", str(app_path)]

    # Additional arguments can be added here as needed
    # Most settings are now loaded from .streamlit/config.toml

    return args


if __name__ == "__main__":
    sys.exit(run_app())
