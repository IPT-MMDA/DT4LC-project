# Makefile for Python projects using uv

# ----- Project Specific Variables -----
PYTHON_VERSION ?= 3.10
PACKAGE_LIB_DIRS := cognitive_ui digital_twin
PROJECT_NAME := dt4lc-project
# Directories to target for linting, formatting, and type checking.
# Using "." applies to all files recognized by tools, respecting .gitignore/excludes.
# Alternatively, be more specific: $(PACKAGE_LIB_DIRS) tests
SRC_TARGET_DIRS := cognitive_ui digital_twin tests
# ----- End Project Specific Variables -----

VENV_DIR := .venv

.DEFAULT_GOAL := help

# Phony targets to avoid conflicts with filenames
.PHONY: help venv install lint format format-check typecheck test tests coverage build clean all-checks

help:
	@echo "------------------------------------------------------------------------------------"
	@echo " $(PROJECT_NAME) - Makefile Help"
	@echo "------------------------------------------------------------------------------------"
	@echo " Environment:"
	@echo "  venv             Create a virtual environment at '$(VENV_DIR)' using Python $(PYTHON_VERSION)"
	@echo "  install          Install '$(PROJECT_NAME)' in editable mode with all (dev) dependencies"
	@echo "  sync             Effectively re-runs install to ensure env is up-to-date with pyproject.toml"
	@echo ""
	@echo " Quality & Formatting (Individual Checks):"
	@echo "  lint             Run ruff linter on $(SRC_TARGET_DIRS)"
	@echo "  format-check     Check formatting with ruff on $(SRC_TARGET_DIRS) (no changes made)"
	@echo "  format           Apply formatting with ruff to $(SRC_TARGET_DIRS)"
	@echo "  typecheck        Run mypy type checker (configured in pyproject.toml)"
	@echo ""
	@echo " Testing (Individual Check):"
	@echo "  test             Run pytest tests (coverage configured in pyproject.toml)"
	@echo "  coverage         Show coverage report in terminal and open HTML report"
	@echo ""
	@echo " Build & Clean:"
	@echo "  build            Build wheel and sdist for '$(PROJECT_NAME)' into 'dist/'"
	@echo "  clean            Remove virtual environment, build artifacts, and caches"
	@echo ""
	@echo " All Checks (Umbrella Target):"
	@echo "  all-checks       Run: format-check, lint, typecheck, and then test"
	@echo "------------------------------------------------------------------------------------"

# Setup and Installation (Prerequisites for running checks)
venv:
	@echo ">>> Creating virtual environment in '$(VENV_DIR)' with Python $(PYTHON_VERSION)..."
	command -v uv >/dev/null 2>&1 || (echo "Error: uv not found. Please install uv: curl -LsSf https://astral.sh/uv/install.sh | sh" && exit 1)
	uv venv $(VENV_DIR) --python $(PYTHON_VERSION)
	@echo ">>> Virtual environment created. Activate with: source $(VENV_DIR)/bin/activate"
	@echo ">>> Then run 'make install' to install dependencies."

$(VENV_DIR)/pyvenv.cfg:
	@echo "Virtual environment '$(VENV_DIR)' not found."
	@echo "Please run 'make venv' first, then activate it (optional but recommended),"
	@echo "and then run your desired make target (e.g., 'make install')."
	@exit 1

install: $(VENV_DIR)/pyvenv.cfg
	@echo ">>> Installing '$(PROJECT_NAME)' in editable mode with dev dependencies into '$(VENV_DIR)'..."
	uv pip install -e .[dev,ui,models,server,agents]
	@echo ">>> Installation complete."

sync: $(VENV_DIR)/pyvenv.cfg
	@echo ">>> Syncing/Re-installing '$(PROJECT_NAME)' with pyproject.toml into '$(VENV_DIR)'..."
	uv pip install -e .[dev,ui,models]
	@echo ">>> Environment synced/updated."

# --- Individual Check Targets ---

format-check: $(VENV_DIR)/pyvenv.cfg
	@echo "\n>>> Checking formatting with ruff on $(SRC_TARGET_DIRS)..."
	uv run ruff format --check $(SRC_TARGET_DIRS)

lint: $(VENV_DIR)/pyvenv.cfg
	@echo "\n>>> Running ruff linter on $(SRC_TARGET_DIRS)..."
	uv run ruff check $(SRC_TARGET_DIRS)

typecheck: $(VENV_DIR)/pyvenv.cfg
	@echo "\n>>> Running mypy type checker (targets configured in pyproject.toml)..."
	uv run mypy . # Mypy typically reads its include/exclude from pyproject.toml

test tests: $(VENV_DIR)/pyvenv.cfg
	@echo "\n>>> Running pytest tests for '$(PROJECT_NAME)'..."
	uv run pytest

# --- Umbrella "All Checks" Target ---

all-checks: format-check lint typecheck test
	@echo "\n✅ All checks (format-check, lint, typecheck, test) completed successfully for '$(PROJECT_NAME)'!"

# --- Other Useful Targets (Formatting, Coverage, Build, Clean) ---

format: $(VENV_DIR)/pyvenv.cfg
	@echo ">>> Formatting code with ruff in $(SRC_TARGET_DIRS)..."
	uv run ruff format $(SRC_TARGET_DIRS)
	@echo ">>> Applying linter fixes with ruff..."
	uv run ruff check $(SRC_TARGET_DIRS) --fix --exit-zero # Exit zero even if fixes are made

coverage: $(VENV_DIR)/pyvenv.cfg
	@echo ">>> Generating and showing coverage report for '$(PROJECT_NAME)'..."
	uv run pytest # Assumes pytest is configured for coverage in pyproject.toml
	@echo "HTML report often in htmlcov/index.html (if configured)"
	@python -m webbrowser -t htmlcov/index.html 2>/dev/null || echo "Note: Could not auto-open HTML coverage report."

build: $(VENV_DIR)/pyvenv.cfg
	@echo ">>> Building '$(PROJECT_NAME)' wheel and sdist into 'dist/'..."
	uv build
	@echo ">>> Build artifacts created in dist/:"
	@ls -l dist/

clean:
	@echo ">>> Cleaning project artifacts and caches for '$(PROJECT_NAME)'..."
	rm -rf dist/
	rm -rf build/
	rm -f .coverage*
	rm -rf htmlcov/
	find . -maxdepth 1 -type d -name "*.egg-info" -exec rm -rf {} + -print
	find . -type d -name "__pycache__" -exec rm -rf {} + -print
	find . -type d -name ".pytest_cache" -exec rm -rf {} + -print
	find . -type d -name ".mypy_cache" -exec rm -rf {} + -print
	find . -type d -name ".ruff_cache" -exec rm -rf {} + -print
	@echo ">>> Cleanup complete."