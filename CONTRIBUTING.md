# Contributing to DT4LC

Thank you for your interest in contributing to DT4LC. This project is licensed under [AGPL-3.0](LICENSE), and we welcome contributions from the community.

## How to Contribute

### Reporting Bugs

Open a [GitHub issue](https://github.com/IPT-MMDA/DT4LC-project/issues) with:

- Steps to reproduce the problem
- Expected vs actual behavior
- System information (OS, Python version, Docker version if applicable)

### Suggesting Features

Open a GitHub issue describing the feature, its use case, and how it fits the project.

### Submitting Code

1. Fork the repository and create a branch from `dev`
2. Make your changes following the code style below
3. Add or update tests if applicable
4. Open a pull request targeting `dev` (not `master`)

Pull requests are reviewed by maintainers before merging.

## Development Setup

### Backend

```bash
git clone https://github.com/IPT-MMDA/DT4LC-project.git
cd DT4LC-project
cp .env.example .env        # configure LLM API keys
uv sync --extra dev --extra server

# Run backend
uv run uvicorn server.app:app --reload --port 8000

# Run tests
uv run pytest tests/ -v

# Lint, format, type check
uv run ruff check .
uv run ruff format .
uv run mypy dta server
```

### Frontend

```bash
cd cognitive_ui
npm install
npm run dev                 # starts on http://localhost:5173
```

### Docker (full stack)

```bash
cp .env.example .env
docker compose up -d        # backend :8000 + frontend :80
```

## Code Style

- Python: [ruff](https://docs.astral.sh/ruff/) for linting/formatting, line length 119
- Type annotations: [mypy](https://mypy-lang.org/) strict mode
- Frontend: ESLint with TypeScript

## Adding New Algorithms

1. Create `dta/dti/algorithms/your_algo.py` with a `run()` function
2. Register in `dta/registry.yaml` with inputs, outputs, and keywords
3. Add tests in `tests/`

See the [README](README.md#adding-new-algorithms) for detailed instructions.

## Language

The primary language for documentation and code is English. Internal communication (issues, PR discussions between team members) may occasionally be in Ukrainian.

### Pre-commit Setup

We use pre-commit hooks to ensure code quality before pushing changes. Make sure all hooks pass before submitting your work.

Install and enable the hooks:

```bash
pip install pre-commit
pre-commit install
```

To run manually on all files:

```bash
pre-commit run --all-files
```

Hooks included:

- **ruff** – lints and auto-fixes Python and stub files (`--fix` enabled by default)
- **ruff-format** – formats Python and stub files
- **mypy** – static type checking using config from `pyproject.toml`, with stubs for `requests` and `PyYAML`
- **trailing-whitespace** – removes trailing whitespace
- **end-of-file-fixer** – ensures files end with a newline
- **check-yaml** – validates YAML file syntax
- **check-toml** – validates TOML file syntax
- **check-merge-conflict** – detects unresolved merge conflict markers
- **debug-statements** – flags leftover `pdb`, `breakpoint()`, and similar debug calls

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold it. Report unacceptable behavior to <chernyatevich.a@gmail.com>.
