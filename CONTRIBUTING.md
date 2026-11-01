# Contributing to DT4LC

Thank you for your interest in contributing to DT4LC. This project is licensed under [AGPL-3.0](LICENSE), and we welcome contributions from the community.

## How to Contribute

### Getting Support

Ask questions and get help by opening a [GitHub issue](https://github.com/IPT-MMDA/DT4LC-project/issues).

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

### Branches and Releases

- `dev` is the integration branch: feature and fix branches start from it and their pull requests target it.
- `master` holds released code. A release is a pull request from `dev` into `master`; the release commit on `master` is tagged `vX.Y.Z` (for example `v0.4.0`).
- From v0.6.1 on, the release branch `release/vX.Y.Z` is merged into `dev`, and `dev` into `master`, with GitHub's "Create a merge commit", never squashed, so `master` keeps every commit with its author. The detailed history of v0.5.0 to v0.6.0, which were squash-merged, stays on the `release/v0.5.0`, `release/v0.5.1` and `release/v0.6.0` branches, which are kept.
- Before tagging, the version is set to the same `X.Y.Z` in `dta/__init__.py`, `cognitive_ui/package.json` and `CITATION.cff` (with `date-released`).

## Development Setup

Full guide: [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) (environment, structure, tests, algorithms, COE, git).

### Backend

```bash
git clone https://github.com/IPT-MMDA/DT4LC-project.git
cd DT4LC-project
cp .env.example .env        # configure LLM API keys
uv sync --extra dev --extra server

# Run backend
uv run uvicorn server.app:app --reload --port 8000

# Run tests (what CI runs; no API keys, network or model weights needed)
uv run pytest tests/ -m "not slow and not llm"

# Opt-in tests (see README "Running Tests" for what each needs)
uv run pytest tests/ -m llm               # an available LLM provider
uv run pytest tests/ -m network           # GEE_PROJECT_ID + Earth Engine credentials
uv run pytest tests/ -m "slow or gpu"     # downloaded models + NVIDIA GPU

# Lint, format, type check
uv run ruff check .
uv run ruff format .
uv run mypy dta server
```

### Frontend

```bash
cd cognitive_ui
npm install
npm run dev                 # starts on http://localhost:3000, with /v1 proxied to :8000
npm run lint                # ESLint
npm run typecheck           # TypeScript type check (tsc -b --noEmit)
npm test                    # unit and component tests (vitest)
npm run build               # production build into dist/, which the browser tests serve
npm run e2e:smoke           # browser smoke test (Playwright) on dist/ and a real API
```

The browser tests need the backend's Python environment (above) and a Chromium headless shell; see [cognitive_ui/tests/e2e/README.md](cognitive_ui/tests/e2e/README.md). On Linux, if the browser fails to start because shared libraries are missing (`libnss3`, `libnspr4`, `libasound2`), run `cognitive_ui/tests/e2e/setup-libs.sh`: it downloads them and extracts them into `cognitive_ui/tests/e2e/.libroot/` without root, and the test runner adds them to the browser's library path.

### Docker (full stack)

```bash
cp .env.example .env
docker compose --profile ollama up -d   # backend :8000 + frontend :80 + local Ollama (no API key needed)
```

## Code Style

- Python: [ruff](https://docs.astral.sh/ruff/) for linting/formatting, line length 119
- Type annotations: [mypy](https://mypy-lang.org/) strict mode
- Frontend: ESLint with TypeScript

## Adding New Algorithms

1. Create `dta/dti/algorithms/your_algo.py` with a `run()` function
2. Register in `dta/registry.yaml` with inputs, outputs, and keywords
3. Add tests in `tests/`

See [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md#adding-a-new-algorithm) for the step-by-step tutorial.

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
