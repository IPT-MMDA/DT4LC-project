"""The stored Library ``card`` of a finished job (v0.6.0 Library, ``server/library.py``).

The jobs run the real orchestrator with every LLM provider down (Ollama on a closed port, API
keys removed), so plans, titles and labels are deterministic. ``GET /v1/jobs`` returns each
card without decoding the job's result.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from datetime import datetime, timedelta
import functools
import os
from pathlib import Path
import sqlite3
from typing import Any

import httpx
import pytest
import rasterio

from dta.config import ROOT_DIR
from dta.dti.coe.llm import get_llm_router, reset_llm_router
from dta.dti.data_sources import gee_common
from dta.dti.raster import describe_raster
from server import jobs as jobs_mod
from server import library
from server.jobs import Job, JobQueue, JobStatus, MemoryJobStore, SQLiteJobStore

SAMPLE_0601 = str(Path(ROOT_DIR) / "resources" / "kahovka_data" / "hlsl_20230601.tif")
SAMPLE_0609 = str(Path(ROOT_DIR) / "resources" / "kahovka_data" / "hlsl_20230609.tif")
SAMPLE_PAIR = [SAMPLE_0601, SAMPLE_0609]
_KEY_VARS = ("GEMINI_API_KEY", "GROQ_API_KEY", "OPENAI_API_KEY", "GPT_SECRET", "ANTHROPIC_API_KEY", "MISTRAL_API_KEY")
WATER_PROMPT = "Compare water before and after the dam breach"
NDMI_PROMPT = "Map moisture change with NDMI between the two dates"
REGION = {
    "bbox": [32.388779, 46.455331, 32.691118, 46.663879],
    "name": "Dnipro floodplain",
    "dataset": "HLS L30",
    "start_date": "2023-06-01",
    "end_date": "2023-06-09",
}
EE_MESSAGE = "Earth Engine isn't configured on this server, so I can't fetch imagery for this region."


@pytest.fixture(autouse=True)
def _providers_down(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for var in _KEY_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "ollama")
    monkeypatch.setenv("LLM_ENABLE_OLLAMA", "true")
    monkeypatch.setenv("LLM_STRATEGY", "fallback")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.delenv("GEE_PROJECT_ID", raising=False)
    monkeypatch.delenv("GEE_SERVICE_ACCOUNT_KEY", raising=False)
    monkeypatch.setattr(gee_common, "_ee_initialized", False)
    reset_llm_router()
    get_llm_router()
    yield
    reset_llm_router()


@pytest.fixture
async def queue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[JobQueue]:
    q = JobQueue(max_workers=1, store=SQLiteJobStore(tmp_path / "jobs.db"))
    monkeypatch.setattr(jobs_mod, "_job_queue", q)
    await q.start()
    yield q
    await q.stop()


def _client() -> httpx.AsyncClient:
    from server.app import app

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@functools.cache
def _attachment(path: str) -> dict[str, Any]:
    name = Path(path).name
    with rasterio.open(path) as src:
        meta = describe_raster(src, name, os.path.getsize(path)).model_dump()
    return {"id": name, "filename": name, "mime_type": "image/tiff", "path": path, "metadata": meta}


async def _wait(q: JobQueue, job_id: str, states: set[str], timeout: float = 120) -> None:
    async def poll() -> None:
        while (job := await q.get_job(job_id)) is None or job.status.value not in states:
            await asyncio.sleep(0.05)

    await asyncio.wait_for(poll(), timeout)


async def _run(client: httpx.AsyncClient, q: JobQueue, prompt: str, paths: list[str], **extra: Any) -> str:
    body = {"prompt": prompt, "attachments": [_attachment(p) for p in paths], **extra}
    r = await client.post("/v1/jobs", json=body)
    assert r.status_code == 202, r.text
    job_id = str(r.json()["id"])
    assert r.json()["card"] is None  # not finished yet
    await _wait(q, job_id, {"completed", "failed", "cancelled", "proposal"})
    if (await q.get_job(job_id)).status == JobStatus.PROPOSAL:  # type: ignore[union-attr]
        assert (await client.post(f"/v1/jobs/{job_id}/confirm")).status_code == 202
        await _wait(q, job_id, {"completed", "failed", "cancelled"})
    return job_id


async def _listed_card(client: httpx.AsyncClient, job_id: str) -> dict[str, Any]:
    jobs = (await client.get("/v1/jobs")).json()["jobs"]
    (item,) = [j for j in jobs if j["id"] == job_id]
    record = (await client.get(f"/v1/jobs/{job_id}")).json()
    assert record["card"] == item["card"]
    assert "result" not in item
    return item["card"]  # type: ignore[no-any-return]


def _without_duration(card: dict[str, Any]) -> dict[str, Any]:
    assert isinstance(card["duration_s"], float) and card["duration_s"] >= 0
    return {k: v for k, v in card.items() if k != "duration_s"}


def _thumbnail(job_id: str, name: str) -> str:
    from urllib.parse import quote

    from dta.dti import artifact_files

    return "/v1/download?path=" + quote(str(artifact_files.output_dir(job_id) / name), safe="")


# --- Done ------------------------------------------------------------------------------------


async def test_ndwi_change_card(queue: JobQueue) -> None:
    async with _client() as client:
        job_id = await _run(client, queue, WATER_PROMPT, SAMPLE_PAIR)
        card = await _listed_card(client, job_id)

    assert _without_duration(card) == {
        "title": "Water change · NDWI difference",
        "analysis": {"id": "algorithms/change-detection", "label": "Change detection"},
        "kind": "change",
        "thumbnail_url": _thumbnail(job_id, "2_ChangeMap_change_map_bare.png"),
        "error": None,
    }


async def test_ndvi_single_index_card(queue: JobQueue) -> None:
    async with _client() as client:
        job_id = await _run(client, queue, "calculate ndvi", [SAMPLE_0601])
        card = await _listed_card(client, job_id)

    assert _without_duration(card) == {
        "title": "NDVI · 2023-06-01",
        "analysis": {"id": "algorithms/ndvi", "label": "NDVI"},
        "kind": "index",
        "thumbnail_url": _thumbnail(job_id, "1_NDVIMap_ndvi_map_bare.png"),
        "error": None,
    }


async def test_statistics_card(queue: JobQueue) -> None:
    async with _client() as client:
        job_id = await _run(client, queue, "Give me statistics for every band", [SAMPLE_0609])
        card = await _listed_card(client, job_id)

    assert _without_duration(card) == {
        "title": "Raster statistics · 2023-06-09",
        "analysis": {"id": "algorithms/statistics", "label": "Statistics"},
        "kind": "statistics",
        "thumbnail_url": None,  # statistics draw no map
        "error": None,
    }


async def test_ndmi_composed_card(queue: JobQueue) -> None:
    async with _client() as client:
        job_id = await _run(client, queue, NDMI_PROMPT, SAMPLE_PAIR)
        card = await _listed_card(client, job_id)

    assert _without_duration(card) == {
        "title": "Moisture change · NDMI difference",
        "analysis": {"id": "algorithms/band-math-change", "label": "Band math"},
        "kind": "composed",
        "thumbnail_url": _thumbnail(job_id, "2_BandMathChangeMap_change_map_bare.png"),
        "error": None,
    }


# --- Failed and cancelled ---------------------------------------------------------------------


async def test_region_job_without_earth_engine_card(queue: JobQueue) -> None:
    async with _client() as client:
        job_id = await _run(client, queue, WATER_PROMPT, [], region=REGION)
        card = await _listed_card(client, job_id)

    assert _without_duration(card) == {
        "title": WATER_PROMPT,  # no plan was made: the prompt
        "analysis": {"id": "algorithms/change-detection", "label": "Change detection"},  # what the intent announced
        "kind": "change",
        "thumbnail_url": None,
        "error": {"code": "EE_NOT_CONFIGURED", "message": EE_MESSAGE},
    }


async def test_band_missing_card(queue: JobQueue) -> None:
    async with _client() as client:
        job_id = await _run(client, queue, "Map burn severity", SAMPLE_PAIR)
        card = await _listed_card(client, job_id)

    error = card.pop("error")
    assert _without_duration(card) == {
        "title": "Map burn severity",
        "analysis": {"id": "algorithms/band-math-change", "label": "Band math"},
        "kind": "composed",
        "thumbnail_url": None,
    }
    assert error["code"] == "BAND_MISSING"
    assert error["message"].startswith("Burn severity uses NBR = (NIR − SWIR2)/(NIR + SWIR2), which needs band B7")


async def test_cancelled_pending_job_card() -> None:
    q = JobQueue(max_workers=1, store=MemoryJobStore())  # not started: the job stays pending
    try:
        job_id = await q.submit_job("calculate ndvi " + "on the whole image " * 5)
        assert await q.cancel_job(job_id)
        job = await q.get_job(job_id)
    finally:
        q._thread_pool.shutdown(wait=False)
    assert job is not None and job.card == {
        "title": "Calculate ndvi on the whole image on the whole image on…",
        "analysis": {"id": "algorithms/ndvi", "label": "NDVI"},
        "kind": "index",
        "thumbnail_url": None,
        "duration_s": None,  # never started
        "error": None,
    }


def test_short_prompt() -> None:
    assert library.short_prompt("  detect field boundaries\n east of the river ") == "Detect field boundaries"
    assert library.short_prompt("") == "Analysis"
    assert len(library.short_prompt("word " * 40)) <= 60


# --- Storage -----------------------------------------------------------------------------------

_CARD = {
    "title": "Water change · NDWI difference",
    "analysis": {"id": "algorithms/change-detection", "label": "Change detection"},
    "kind": "change",
    "thumbnail_url": "/v1/download?path=x",
    "duration_s": 48.0,
    "error": None,
}


class _UndecodableResult(dict[str, Any]):
    def items(self) -> Any:
        raise AssertionError("listing touched job.result")


def _break_results(store: SQLiteJobStore | MemoryJobStore, tmp_path: Path) -> None:
    """Make every stored ``result`` fail if anything decodes or serialises it."""
    if isinstance(store, SQLiteJobStore):
        con = sqlite3.connect(tmp_path / "jobs.db")
        con.execute("UPDATE jobs SET result = '{not json'")
        con.commit()
        con.close()
    else:
        for job in store.list_all():
            job.result = _UndecodableResult(x=1)


@pytest.mark.parametrize("kind", ["sqlite", "memory"])
def test_list_returns_cards_without_decoding_results(
    kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from server.app import app

    store = SQLiteJobStore(tmp_path / "jobs.db") if kind == "sqlite" else MemoryJobStore()
    base = datetime(2026, 10, 22)
    for i in range(3):
        store.save(
            Job(
                id=f"job{i}",
                status=JobStatus.COMPLETED,
                prompt=WATER_PROMPT,
                result={"big": [0.5] * 1000},
                created_at=base + timedelta(seconds=i),
                started_at=base + timedelta(seconds=i),
                completed_at=base + timedelta(seconds=i + 48),
                card=dict(_CARD) if i else None,
            )
        )
    monkeypatch.setattr(jobs_mod, "_job_queue", JobQueue(max_workers=1, store=store))
    _break_results(store, tmp_path)  # as in test_job_list_summaries.py
    try:
        r = TestClient(app).get("/v1/jobs")
    finally:
        jobs_mod._job_queue._thread_pool.shutdown(wait=False)  # type: ignore[union-attr]

    assert r.status_code == 200, r.text
    assert [(j["id"], j["card"]) for j in r.json()["jobs"]] == [("job2", _CARD), ("job1", _CARD), ("job0", None)]


def test_older_jobs_db_without_card_column_migrates(tmp_path: Path) -> None:
    db = tmp_path / "jobs.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """CREATE TABLE jobs (
            id TEXT PRIMARY KEY, status TEXT NOT NULL, prompt TEXT NOT NULL,
            attachments TEXT NOT NULL DEFAULT '[]', context TEXT, plan TEXT, result TEXT,
            progress REAL NOT NULL DEFAULT 0.0, error TEXT, created_at TEXT NOT NULL,
            started_at TEXT, completed_at TEXT, events TEXT NOT NULL DEFAULT '[]',
            owner TEXT, owner_role TEXT, region TEXT
        )"""
    )
    conn.execute(
        "INSERT INTO jobs (id, status, prompt, created_at, owner) VALUES (?, ?, ?, ?, ?)",
        ("old1", "completed", "calculate ndvi", "2026-10-01T10:00:00", "o"),
    )
    conn.commit()
    conn.close()

    store = SQLiteJobStore(db)  # adds the column

    old = store.get("old1")
    assert old is not None and old.card is None
    assert [j.card for j in store.list_summaries()] == [None]
    old.card = dict(_CARD)
    store.save(old)
    assert SQLiteJobStore(db).get("old1").card == _CARD  # type: ignore[union-attr]
    assert SQLiteJobStore(db).list_summaries()[0].card == _CARD
