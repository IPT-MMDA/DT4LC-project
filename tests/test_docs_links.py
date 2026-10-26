"""The documentation points at things that exist.

Offline and fast: only file reads, no git, network or subprocess. Each failing case names the
document, the line and the target it could not find.

- Relative links and images in the checked documents resolve to a file or directory.
- Every quoted ``npm run <script>`` is a script in ``cognitive_ui/package.json``.
- Every quoted ``task <name>`` is a task in ``Taskfile.yml``.
- Every ``--profile <p>`` and ``COMPOSE_PROFILES=...`` names a profile a compose service declares.
- The old page names appear in no tracked Markdown file except ``CHANGELOG.md``.
"""

from __future__ import annotations

from collections.abc import Iterator
import functools
import json
from pathlib import Path
import re
from typing import Any, NamedTuple
from urllib.parse import unquote

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

# The documents whose links, scripts, tasks and profiles are checked. CHANGELOG.md is checked once it exists.
CHECKED_TOP_LEVEL = ("README.md", "CONTRIBUTING.md", "CHANGELOG.md")

# Cases that fail on purpose until a documentation change lands. Keyed by (document, target), not by
# line, so an edit above the line doesn't drop the mark. The marks are strict: remove one when its
# case passes.
PRE_EXISTING = "pre-existing, fixed in this run's docs stories"
KNOWN_BROKEN: dict[tuple[str, str], str] = {
    ("THIRD_PARTY_LICENSES.md", "Models page"): PRE_EXISTING,
}

OLD_PAGE_NAMES = ("Jobs page", "Data page", "Models page", "Dashboard page")


class Ref(NamedTuple):
    doc: str
    line: int
    target: str


def checked_docs(root: Path) -> list[Path]:
    docs = [root / name for name in CHECKED_TOP_LEVEL if (root / name).is_file()]
    return docs + sorted((root / "docs").glob("*.md"))


def tracked_markdown(root: Path) -> list[Path]:
    """The tracked Markdown files, without asking git.

    A fixed set of roots that matches ``git ls-files '*.md'``: the top-level files by name (the repo
    root also holds untracked planning notes, so no glob there), ``.github/``, ``docs/``, and
    ``cognitive_ui/`` without the ``node_modules`` and ``dist`` build trees. CHANGELOG.md is left out
    on purpose: it may name the old pages.
    """
    top = ("README.md", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "THIRD_PARTY_LICENSES.md", "paper.md")
    files = [root / name for name in top if (root / name).is_file()]
    for sub in (".github", "docs"):
        files += sorted((root / sub).rglob("*.md"))
    files += sorted(p for p in (root / "cognitive_ui").rglob("*.md") if not {"node_modules", "dist"} & set(p.parts))
    return files


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


# --- Markdown scanning ---------------------------------------------------------------------------

_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_INLINE_CODE = re.compile(r"(`+)(.+?)\1")


def _lines(path: Path) -> Iterator[tuple[int, str, bool]]:
    """Yield (line number, text, inside a fenced code block). Fence lines themselves count as code."""
    fence: str | None = None
    for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        m = _FENCE.match(text)
        if fence is None and m:
            fence = m.group(1)
            yield number, text, True
        elif fence is not None:
            if (
                m
                and m.group(1)[0] == fence[0]
                and len(m.group(1)) >= len(fence)
                and not text.strip()[len(m.group(1)) :]
            ):
                fence = None
            yield number, text, True
        else:
            yield number, text, False


def _code_snippets(path: Path) -> Iterator[tuple[int, str]]:
    """Every piece of code in a document: each fenced line, and each inline code span."""
    for number, text, fenced in _lines(path):
        if fenced:
            yield number, text
        else:
            for m in _INLINE_CODE.finditer(text):
                yield number, m.group(2)


def _all_text(path: Path) -> Iterator[tuple[int, str]]:
    for number, text, _ in _lines(path):
        yield number, text


# --- 1. Links and images -------------------------------------------------------------------------

# `](target)` covers links and images, also when they nest (a badge image inside a link).
_MD_LINK = re.compile(r"\]\(\s*(<[^>]*>|[^)\s]+)(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^)]*\)))?\s*\)")
_REF_DEF = re.compile(r"^ {0,3}\[[^\]]+\]:\s*(<[^>]*>|\S+)")
_HTML_SRC = re.compile(r"<(?:img|source)\b[^>]*?\bsrc\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
_EXTERNAL = ("http://", "https://", "mailto:")


def link_refs(path: Path, root: Path) -> Iterator[Ref]:
    """Relative link and image targets outside code, with the fragment and query stripped."""
    for number, text, fenced in _lines(path):
        if fenced:
            continue
        prose = _INLINE_CODE.sub(lambda m: " " * len(m.group(0)), text)
        targets = [m.group(1) for m in _MD_LINK.finditer(prose)]
        targets += [m.group(1) for m in _REF_DEF.finditer(prose)]
        targets += [m.group(1) for m in _HTML_SRC.finditer(prose)]
        for raw in targets:
            target = raw.strip("<>")
            if target.lower().startswith(_EXTERNAL) or target.startswith("#"):
                continue
            yield Ref(_rel(path, root), number, target)


def _resolve(ref: Ref, root: Path) -> Path:
    target = unquote(re.split(r"[#?]", ref.target, maxsplit=1)[0])
    # A leading slash is the repository root, as GitHub renders it.
    base = root if target.startswith("/") else (root / ref.doc).parent
    return (base / target.lstrip("/")).resolve()


# --- 2. npm scripts ------------------------------------------------------------------------------

# Script names may contain colons (e2e:workspace:guest); a trailing colon, full stop or comma is prose.
_NPM_RUN = re.compile(r"\bnpm\s+run(?:-script)?\s+([A-Za-z0-9_][\w:.-]*)")


def npm_refs(path: Path, root: Path) -> Iterator[Ref]:
    """`npm run <script>` anywhere in the document, prose and code alike: the phrase is unambiguous."""
    for number, text in _all_text(path):
        for m in _NPM_RUN.finditer(text):
            yield Ref(_rel(path, root), number, m.group(1).rstrip(".:,;"))


@functools.cache
def npm_scripts(root: Path) -> set[str]:
    return set(json.loads((root / "cognitive_ui" / "package.json").read_text(encoding="utf-8"))["scripts"])


# --- 3. Task names -------------------------------------------------------------------------------

# "task" is also an English word ("so the task fails"), so a name counts only when it is code and
# at a command position: inside a fenced block line or an inline code span, at its start or after a
# shell separator (&&, ||, ;, |, a "$ " prompt or an opening parenthesis). A "#" comment at the end
# of a fenced line is dropped first. Flags such as `task --list` are not names.
_TASK = re.compile(r"(?:^|&&|\|\||[;|(]|\$\s)\s*task\s+([A-Za-z0-9_][\w:.-]*)")
_SHELL_COMMENT = re.compile(r"(?:^|\s)#.*$")


def task_refs(path: Path, root: Path) -> Iterator[Ref]:
    for number, snippet in _code_snippets(path):
        for m in _TASK.finditer(_SHELL_COMMENT.sub("", snippet)):
            yield Ref(_rel(path, root), number, m.group(1).rstrip(".:,;"))


@functools.cache
def taskfile_tasks(taskfile: Path, prefix: str = "") -> set[str]:
    """Task names and aliases in a Taskfile, with `includes:` namespaced as `<ns>:<task>`."""
    data = yaml.safe_load(taskfile.read_text(encoding="utf-8")) or {}
    names: set[str] = set()
    for name, spec in (data.get("tasks") or {}).items():
        names.add(prefix + str(name))
        if isinstance(spec, dict):
            names.update(prefix + str(alias) for alias in spec.get("aliases") or [])
    for namespace, include in (data.get("includes") or {}).items():
        included = include if isinstance(include, str) else include["taskfile"]
        target = taskfile.parent / included
        if target.is_dir():
            target = target / "Taskfile.yml"
        names |= taskfile_tasks(target, f"{prefix}{namespace}:")
    return names


# --- 4. Compose profiles -------------------------------------------------------------------------

# `--profile <p>` anywhere in a document (the docs also write "add `--profile ollama`" without the
# `docker compose` in front), and each name in `COMPOSE_PROFILES=a,b`.
_PROFILE_FLAG = re.compile(r"--profile[=\s]+([A-Za-z0-9_][\w.-]*)")
_COMPOSE_PROFILES = re.compile(r"\bCOMPOSE_PROFILES=([\w.,-]+)")
COMPOSE_FILES = ("docker-compose.yml", "docker-compose.demo.yml")


def profile_refs(path: Path, root: Path) -> Iterator[Ref]:
    for number, text in _all_text(path):
        for m in _PROFILE_FLAG.finditer(text):
            yield Ref(_rel(path, root), number, m.group(1).rstrip("."))
        for m in _COMPOSE_PROFILES.finditer(text):
            for name in m.group(1).rstrip(".,").split(","):
                if name:
                    yield Ref(_rel(path, root), number, name)


class _ComposeLoader(yaml.SafeLoader):
    """safe_load plus compose's own tags (`!reset`, `!override`): the tag is dropped, the value kept."""


def _construct_tagged(loader: yaml.SafeLoader, suffix: str, node: yaml.Node) -> Any:
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    return loader.construct_scalar(node)


_ComposeLoader.add_multi_constructor("!", _construct_tagged)


@functools.cache
def compose_profiles(root: Path) -> set[str]:
    profiles: set[str] = set()
    for name in COMPOSE_FILES:
        compose = yaml.load((root / name).read_text(encoding="utf-8"), Loader=_ComposeLoader)
        for service in (compose.get("services") or {}).values():
            profiles.update((service or {}).get("profiles") or [])
    return profiles


# --- 5. Old page names ---------------------------------------------------------------------------

_OLD_PAGE = re.compile(r"\b(" + "|".join(n.replace(" ", r"\s+") for n in OLD_PAGE_NAMES) + r")\b", re.IGNORECASE)


def old_page_hits(path: Path, root: Path) -> list[Ref]:
    """Every old page name in the file, code included; a name may wrap across lines."""
    text = path.read_text(encoding="utf-8")
    return [
        Ref(_rel(path, root), text.count("\n", 0, m.start()) + 1, " ".join(m.group(1).split()))
        for m in _OLD_PAGE.finditer(text)
    ]


# --- Parametrization -----------------------------------------------------------------------------


def _params(refs: Iterator[Ref] | list[Ref]) -> list[Any]:
    params = []
    for ref in refs:
        reason = KNOWN_BROKEN.get((ref.doc, ref.target))
        marks = [pytest.mark.xfail(strict=True, reason=reason)] if reason else []
        params.append(pytest.param(ref, id=f"{ref.doc}:{ref.line}:{ref.target}", marks=marks))
    return params


def _collect(extract: Any) -> list[Ref]:
    return [ref for doc in checked_docs(ROOT) for ref in extract(doc, ROOT)]


LINKS = _collect(link_refs)
NPM = _collect(npm_refs)
TASKS = _collect(task_refs)
PROFILES = _collect(profile_refs)
OLD_PAGES = [hit for doc in tracked_markdown(ROOT) for hit in old_page_hits(doc, ROOT)]


def test_the_scanners_find_something() -> None:
    """Guards against a regex that silently matches nothing and leaves a check below empty."""
    assert LINKS and NPM and TASKS and PROFILES, (len(LINKS), len(NPM), len(TASKS), len(PROFILES))


@pytest.mark.parametrize("ref", _params(LINKS))
def test_relative_link_resolves(ref: Ref) -> None:
    resolved = _resolve(ref, ROOT)
    assert resolved.is_relative_to(ROOT), f"{ref.doc}:{ref.line}: link {ref.target!r} points outside the repository"
    assert resolved.exists(), f"{ref.doc}:{ref.line}: link {ref.target!r} → {_rel(resolved, ROOT)} does not exist"


@pytest.mark.parametrize("ref", _params(NPM))
def test_npm_script_exists(ref: Ref) -> None:
    assert ref.target in npm_scripts(ROOT), (
        f"{ref.doc}:{ref.line}: `npm run {ref.target}` is not a script in cognitive_ui/package.json"
    )


@pytest.mark.parametrize("ref", _params(TASKS))
def test_task_exists(ref: Ref) -> None:
    assert ref.target in taskfile_tasks(ROOT / "Taskfile.yml"), (
        f"{ref.doc}:{ref.line}: `task {ref.target}` is not a task in Taskfile.yml"
    )


@pytest.mark.parametrize("ref", _params(PROFILES))
def test_compose_profile_exists(ref: Ref) -> None:
    profiles = compose_profiles(ROOT)
    assert ref.target in profiles, (
        f"{ref.doc}:{ref.line}: compose profile {ref.target!r} is declared by no service in "
        f"{' or '.join(COMPOSE_FILES)} (profiles: {sorted(profiles)})"
    )


# With no hit left the parameter set would be empty (a skip); one empty case keeps it a pass.
@pytest.mark.parametrize("ref", _params(OLD_PAGES) or [pytest.param(None, id="none")])
def test_old_page_name_absent(ref: Ref | None) -> None:
    if ref is not None:
        pytest.fail(
            f"{ref.doc}:{ref.line}: old page name {ref.target!r}; no such page since v0.6.0 (cognitive_ui/src/pages/)"
        )


def test_known_broken_marks_are_still_needed() -> None:
    """A strict xfail can't fail once its case is gone (the link or phrase removed): remove the mark then."""
    collected = {(ref.doc, ref.target) for ref in [*LINKS, *NPM, *TASKS, *PROFILES, *OLD_PAGES]}
    stale = sorted(set(KNOWN_BROKEN) - collected)
    assert not stale, f"KNOWN_BROKEN entries whose case no longer exists; delete them: {stale}"
