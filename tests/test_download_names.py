"""Readable download names: ``download_name`` on artifact references and
``GET /v1/download?path=…&name=…``.

The stored files keep their names (``OUTPUTS_PATH/<job>/2_ChangeMap_change.tif``); ``name`` only
sets ``Content-Disposition``. The names of real sample jobs are tested in ``test_job_cards.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from dta.dti import artifact_files
from dta.dti.registry import load_registry
from dta.dti.schemas import ExecutionPlan, PlanStep
from server import library
from server.app import app
import server.jobs as jobs_mod
from server.jobs import Job, JobQueue, JobStatus, MemoryJobStore
from server.routes.files import attachment_name

NAME = "water-change_ndwi_2023-06-01_2023-06-09.tif"


@pytest.fixture
def queue(monkeypatch: pytest.MonkeyPatch) -> Iterator[JobQueue]:
    """A queue without workers: submitted jobs stay pending."""
    q = JobQueue(max_workers=1, store=MemoryJobStore())
    monkeypatch.setattr(jobs_mod, "_job_queue", q)
    yield q
    q._thread_pool.shutdown(wait=False)


def _client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def _artifact(outputs: Path, job_id: str) -> Path:
    tif = outputs / job_id / "2_ChangeMap_change.tif"
    tif.parent.mkdir(exist_ok=True)
    tif.write_bytes(b"II*\x00 change raster bytes")
    return tif


# --- Content-Disposition ---------------------------------------------------------------------


def test_download_is_named_by_name(queue: JobQueue, _outputs_in_tmp: Path) -> None:
    tif = _artifact(_outputs_in_tmp, "jobA")
    r = _client().get(artifact_files.download_url(str(tif), NAME))

    assert r.status_code == 200
    assert r.headers["content-disposition"] == f"attachment; filename=\"{NAME}\"; filename*=UTF-8''{NAME}"
    assert r.content == tif.read_bytes()
    assert tif.name == "2_ChangeMap_change.tif"  # the stored name is unchanged


def test_download_without_name_keeps_the_stored_name(queue: JobQueue, _outputs_in_tmp: Path) -> None:
    tif = _artifact(_outputs_in_tmp, "jobB")
    r = _client().get("/v1/download", params={"path": str(tif)})

    assert r.status_code == 200
    assert r.headers["content-disposition"] == (
        "attachment; filename=\"2_ChangeMap_change.tif\"; filename*=UTF-8''2_ChangeMap_change.tif"
    )


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("../../etc/passwd.tif", "etc-passwd.tif"),  # no path separators, no leading dots
        ("..\\..\\boot.ini", "boot.ini.tif"),  # backslashes too; the extension is the served file's
        ('evil".tif', "evil-.tif"),  # a quote would end filename="…"
        ("a\r\nSet-Cookie: x=1.tif", "a--Set-Cookie- x=1.tif"),  # control characters cannot split the header
        ("ndwi_2023-06-01", "ndwi_2023-06-01.tif"),  # extension added
        ("NDWI.TIF", "NDWI.TIF"),  # same extension, any case
        ("change.png", "change.png.tif"),  # never another type than the file served
        ("", "2_ChangeMap_change.tif"),
        ("/../.", "2_ChangeMap_change.tif"),  # nothing left: the stored name
        ("x" * 300, "x" * 200 + ".tif"),
    ],
)
def test_attachment_name_is_sanitised(name: str, expected: str) -> None:
    assert attachment_name(name, Path("/o/job/2_ChangeMap_change.tif")) == expected


def test_hostile_name_in_the_header(queue: JobQueue, _outputs_in_tmp: Path) -> None:
    tif = _artifact(_outputs_in_tmp, "jobC")
    r = _client().get("/v1/download", params={"path": str(tif), "name": '../../etc/pass"wd.tif'})

    assert r.status_code == 200
    assert r.headers["content-disposition"] == (
        "attachment; filename=\"etc-pass-wd.tif\"; filename*=UTF-8''etc-pass-wd.tif"
    )
    assert r.content == tif.read_bytes()


def test_non_ascii_name_has_an_ascii_fallback(queue: JobQueue, _outputs_in_tmp: Path) -> None:
    tif = _artifact(_outputs_in_tmp, "jobD")
    r = _client().get("/v1/download", params={"path": str(tif), "name": "вода_2023-06-09.tif"})

    assert r.headers["content-disposition"] == (
        "attachment; filename=\"_____2023-06-09.tif\"; filename*=UTF-8''%D0%B2%D0%BE%D0%B4%D0%B0_2023-06-09.tif"
    )


# --- name never changes which file is served ------------------------------------------------


def test_name_does_not_change_the_file_or_the_owner_check(
    monkeypatch: pytest.MonkeyPatch, queue: JobQueue, _outputs_in_tmp: Path
) -> None:
    monkeypatch.setenv("DEV_ROLE", "guest")
    alice, bob = _client(), _client()
    job_id = alice.post("/v1/jobs", json={"prompt": "Compute NDVI"}).json()["id"]
    tif = _artifact(_outputs_in_tmp, job_id)
    other = _outputs_in_tmp / job_id / "1_NDVIMap_ndvi.tif"
    other.write_bytes(b"II*\x00 another file")

    # A name that is another file of the job, or an absolute path, still serves ``path``.
    for name in (other.name, str(other), "1_NDVIMap_ndvi.tif"):
        r = alice.get("/v1/download", params={"path": str(tif), "name": name})
        assert r.status_code == 200 and r.content == tif.read_bytes(), name
    # Another session gets 404, with or without a name.
    for params in ({"path": str(tif)}, {"path": str(tif), "name": NAME}):
        assert bob.get("/v1/download", params=params).status_code == 404
    # Outside the allowed roots is 403 whatever the name.
    r = alice.get("/v1/download", params={"path": "/etc/passwd", "name": NAME})
    assert r.status_code == 403


# --- name_downloads on a constructed job ----------------------------------------------------


def _job(steps: list[PlanStep], artifacts: dict[str, object]) -> Job:
    return Job(
        id="j",
        status=JobStatus.COMPLETED,
        prompt="p",
        plan=ExecutionPlan(flow="p", steps=steps),
        result={"execution": {"artifacts": artifacts}},
        completed_at=datetime.now(),
    )


def _change_artifact(prefix: str) -> dict[str, object]:
    def ref(kind: str, name: str) -> dict[str, object]:
        return artifact_files.ref(kind, f"/o/j/{prefix}_{name}")

    return {
        "rasters": {
            "change": ref("geotiff", "change.tif"),
            "index_before": ref("geotiff", "index_before.tif"),
            "index_after": ref("geotiff", "index_after.tif"),
        },
        "visualizations": {"change_map": ref("png", "change_map.png"), "missing": None},
    }


def _names(job: Job) -> dict[str, str]:
    return {Path(r["path"]).name: r["download_name"] for r in library._refs(job.result)}


def test_names_are_unique_within_a_job() -> None:
    """Files without a date in their name: the two per-date indices would share a name."""
    job = _job(
        [
            PlanStep(uses="input/file-before", binds={"RasterPathBefore": "/u/before.tif"}),
            PlanStep(uses="input/file-after", binds={"RasterPathAfter": "/u/after.tif"}),
            PlanStep(uses="algorithms/change-detection", binds={"IndexType": "ndwi"}),
        ],
        {
            "ChangeMap": _change_artifact("2_ChangeMap"),
            "Other": {"rasters": {"x": artifact_files.ref("geotiff", "/o/j/9_x.tif")}},
        },
    )
    library.name_downloads(job, load_registry())

    assert _names(job) == {
        "2_ChangeMap_change.tif": "water-change_ndwi.tif",
        "2_ChangeMap_index_before.tif": "ndwi.tif",
        "2_ChangeMap_index_after.tif": "ndwi-2.tif",
        "2_ChangeMap_change_map.png": "water-change_ndwi.png",
        "9_x.tif": "9-x.tif",  # outside an analysis step: its stored name
    }
    ref = job.result["execution"]["artifacts"]["ChangeMap"]["rasters"]["change"]  # type: ignore[index]
    assert ref["url"] == artifact_files.download_url(ref["path"], "water-change_ndwi.tif")
    assert ref["path"] == "/o/j/2_ChangeMap_change.tif"


def test_region_job_names_the_requested_dates() -> None:
    """A region change job fetches ±3-day windows around 2023-06-01 and 2023-06-09: the names say the dates."""
    region = {"dataset": "HLS L30", "bbox": [32.4, 46.4, 32.7, 46.7], "bands": ["B3", "B5"], "cloud_max": 20}
    job = _job(
        [
            PlanStep(
                uses="input/earth-engine-before",
                config={**region, "start_date": "2023-05-29", "end_date": "2023-06-04"},
            ),
            PlanStep(
                uses="input/earth-engine-after",
                config={**region, "start_date": "2023-06-06", "end_date": "2023-06-12"},
            ),
            PlanStep(uses="algorithms/change-detection", binds={"IndexType": "ndwi"}),
        ],
        {"ChangeMap": _change_artifact("2_ChangeMap")},
    )
    library.name_downloads(job, load_registry())

    assert _names(job)["2_ChangeMap_change.tif"] == NAME
    assert _names(job)["2_ChangeMap_index_after.tif"] == "ndwi_2023-06-09.tif"

    single = _job(
        [
            PlanStep(
                uses="input/earth-engine", config={**region, "start_date": "2023-06-01", "end_date": "2023-06-09"}
            ),
            PlanStep(uses="algorithms/ndvi"),
        ],
        {"NDVIMap": {"rasters": {"ndvi": artifact_files.ref("geotiff", "/o/j/1_NDVIMap_ndvi.tif")}}},
    )
    library.name_downloads(single, load_registry())
    assert _names(single) == {"1_NDVIMap_ndvi.tif": "ndvi_2023-06-01_2023-06-09.tif"}  # the composite's window
