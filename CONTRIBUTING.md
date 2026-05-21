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

See [Git workflow](#git-workflow) below for branch naming, claiming issues, and the full PR process.

## Git workflow

### Branch naming

Create branches from your fork’s **`dev`** (synced with `upstream/dev`). Use the issue number in the name:

| Prefix | When to use | Example |
|--------|-------------|---------|
| `feature/` | New functionality | `feature/18-send-to-chat` |
| `fix/` | Bug fixes | `fix/42-ndvi-band-mapping` |
| `docs/` | Documentation only | `docs/9-developer-guide` |

Format: `<prefix>/<issue-number>-<short-kebab-desc>` (e.g. `feature/18-map-chat-integration`).

Do **not** use unrelated names (`job-result-19`, `my-branch`) — reviewers need to see which issue the PR addresses.

### Workflow

```text
fork → sync dev → branch → implement → test → PR → review → merge
```

1. **Fork** [IPT-MMDA/DT4LC-project](https://github.com/IPT-MMDA/DT4LC-project) on GitHub.
2. **Sync `dev`** on your fork with upstream:
   ```bash
   git remote add upstream https://github.com/IPT-MMDA/DT4LC-project.git  # once
   git fetch upstream
   git checkout dev
   git merge upstream/dev
   git push origin dev
   ```
3. **Branch** from `dev`:
   ```bash
   git checkout -b feature/18-send-to-chat
   ```
4. **Implement** — one issue per branch; do not mix unrelated tasks.
5. **Test** before opening a PR:
   ```bash
   task test          # or: uv run pytest tests/ -v
   task lint          # Python
   cd cognitive_ui && npm run lint   # frontend, if changed
   ```
6. **Pull request** — base: `IPT-MMDA/dev`, head: `YourFork/feature/18-send-to-chat`.
   - Use a clear title (e.g. `feature: send map region to chat (#18)`).
   - Fill in the [PR template](.github/pull_request_template.md) (description, `Fixes #N`, testing).
   - Compare link: `https://github.com/IPT-MMDA/DT4LC-project/compare/dev...YourFork:feature/18-send-to-chat`
7. **Review** — push fixes to the **same branch**; do not open a second PR for the same issue.
8. **Merge** — maintainers merge into `dev` after approval and CI pass.

Target branch is always **`dev`**, not `master`.

### How to claim an issue

1. Open the [issues list](https://github.com/IPT-MMDA/DT4LC-project/issues) and pick an unassigned item (or one assigned to you).
2. Check there is **no open PR** already linked (`Fixes #N` in existing PRs).
3. Comment on the issue, e.g. *“I’ll work on this”*, so others do not duplicate effort.
4. Create your branch using the naming rules above.
5. Reference the issue in the PR body: `Fixes #18` (or `Related to #18` if it only partly addresses it).

If you stop working on an issue, comment on the issue so it can be reassigned.

### PR checklist

When you open a PR, use the [pull request template](.github/pull_request_template.md) and confirm:

- [ ] Branch name follows `feature|fix|docs/<issue>-<short-desc>`
- [ ] PR targets **`dev`**
- [ ] One issue per PR (no unrelated commits)
- [ ] Description explains what and why
- [ ] `Fixes #<number>` (or explicit “Related to”) in the PR body
- [ ] Tests run (`task test` or documented why N/A for docs-only)
- [ ] Lint/typecheck run if you changed code
- [ ] Review feedback addressed on the **same** branch (no duplicate PRs)

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
