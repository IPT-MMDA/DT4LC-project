"""``POST /v1/upload`` metadata: sensor guess, bands, resolution, date, size and display line."""

from pathlib import Path
import shutil
from unittest.mock import patch

from fastapi.testclient import TestClient
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from dta.config import RESOURCES_PATH
from dta.dti.raster import _acquisition_date, _band_summary, describe_raster, format_size
from dta.dti.schemas import Attachment
from server.app import app
import server.routes.files as files_routes

KAKHOVKA = RESOURCES_PATH / "kahovka_data" / "hlsl_20230601.tif"


@pytest.fixture
def uploads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write uploads under a temp dir instead of resources/.cache/uploads."""
    root = tmp_path / "uploads"
    root.mkdir()
    monkeypatch.setattr(files_routes, "UPLOAD_DIR", root)
    return root


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _write(
    path: Path,
    *,
    count: int,
    descriptions: list[str] | None = None,
    crs: str = "EPSG:32636",
    res: float = 10.0,
    dtype: str = "uint16",
    tags: dict[str, str] | None = None,
) -> Path:
    origin = (32.4, 46.6) if crs == "EPSG:4326" else (650_000.0, 5_160_000.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=16,
        height=16,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=from_origin(*origin, res, res),
    ) as dst:
        dst.write((np.arange(count * 256).reshape(count, 16, 16) % 1000 + 1).astype(dtype))
        if descriptions:
            dst.descriptions = tuple(descriptions)
        if tags:
            dst.update_tags(**tags)
    return path


def _upload(client: TestClient, path: Path) -> dict:  # type: ignore[type-arg]
    with path.open("rb") as fh:
        resp = client.post("/v1/upload", files={"file": (path.name, fh, "image/tiff")})
    assert resp.status_code == 200, resp.text
    return resp.json()  # type: ignore[no-any-return]


def test_kakhovka_crop_line(client: TestClient, uploads: Path) -> None:
    meta = _upload(client, KAKHOVKA)["metadata"]
    assert meta["line"] == "HLS L30 · B2–B6 · 30 m · 2023-06-01"
    assert meta["sensor"] == "HLS L30"
    assert [b["code"] for b in meta["bands"]] == ["B2", "B3", "B4", "B5", "B6"]
    assert [b["index"] for b in meta["bands"]] == [1, 2, 3, 4, 5]
    assert meta["dtype"] == "int16"
    assert meta["resolution_m"] == 30.0
    assert meta["date"] == "2023-06-01"
    assert meta["size_bytes"] == KAKHOVKA.stat().st_size
    assert meta["size_label"] == "4.0 MB"
    assert meta["crs"] == "EPSG:32636"
    west, south, east, north = meta["bounds_wgs84"]
    # DESIGN §8: 32.39–32.69 E, 46.46–46.66 N
    assert (round(west, 2), round(east, 2), round(south, 2), round(north, 2)) == (32.39, 32.69, 46.46, 46.66)


def test_synthetic_sentinel2(client: TestClient, uploads: Path, tmp_path: Path) -> None:
    path = _write(
        tmp_path / "S2A_crop_2023-06-09_l2a.tif", count=6, descriptions=["B2", "B3", "B4", "B8", "B11", "B12"]
    )
    meta = _upload(client, path)["metadata"]
    assert meta["sensor"] == "Sentinel-2"
    assert meta["line"] == "Sentinel-2 · B2–B4, B8, B11, B12 · 10 m · 2023-06-09"
    assert meta["dtype"] == "uint16"


def test_no_descriptions_or_date_gives_partial_line(client: TestClient, uploads: Path, tmp_path: Path) -> None:
    path = _write(tmp_path / "scene.tif", count=3, crs="EPSG:4326", res=0.0001, dtype="float32")
    meta = _upload(client, path)["metadata"]
    assert meta["sensor"] is None
    assert meta["date"] is None
    assert [b["code"] for b in meta["bands"]] == [None, None, None]
    assert [b["description"] for b in meta["bands"]] == [None, None, None]
    assert meta["resolution_m"] == 11.1  # 0.0001° of latitude
    assert meta["line"] == "3 bands · 11.1 m"
    assert meta["crs"] == "EPSG:4326"


def test_metadata_failure_does_not_fail_upload(client: TestClient, uploads: Path, tmp_path: Path) -> None:
    path = _write(tmp_path / "hlsl_20230601.tif", count=5, descriptions=["B2", "B3", "B4", "B5", "B6"])
    with patch.object(files_routes, "describe_raster", side_effect=RuntimeError("boom")):
        body = _upload(client, path)
    assert body["metadata"]["size_bytes"] == path.stat().st_size
    assert body["metadata"]["line"] == ""


def test_attachment_carries_metadata(client: TestClient, uploads: Path) -> None:
    body = _upload(client, KAKHOVKA)
    att = Attachment.model_validate(
        {"id": body["id"], "filename": body["filename"], "mime_type": "image/tiff", "metadata": body["metadata"]}
    )
    assert att.metadata is not None and att.metadata.line == body["metadata"]["line"]
    assert att.model_dump()["metadata"] == body["metadata"]


def test_file_listing_carries_metadata(client: TestClient, uploads: Path) -> None:
    # Stored as <id>_<original name>; the id must not be read as a date.
    shutil.copy(KAKHOVKA, uploads / "20991231_hlsl_20230601.tif")
    listed = [f for f in client.get("/v1/files").json()["files"] if f["path"].startswith(str(uploads))]
    assert listed[0]["metadata"]["line"] == "HLS L30 · B2–B6 · 30 m · 2023-06-01"


def test_landsat_collection2_and_tag_date(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "export.tif",
        count=6,
        res=30.0,
        descriptions=["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],
        tags={"DATE_ACQUIRED": "2023-06-09"},
    )
    with rasterio.open(path) as src:
        meta = describe_raster(src, path.name, path.stat().st_size)
    assert meta.sensor == "Landsat 8/9"
    assert meta.line == "Landsat 8/9 · B2–B7 · 30 m · 2023-06-09"


def test_plain_landsat_codes_at_30_m_read_as_hls_l30(tmp_path: Path) -> None:
    path = _write(tmp_path / "crop.tif", count=5, res=30.0, descriptions=["B2", "B3", "B4", "B5", "B6"])
    with rasterio.open(path) as src:
        assert describe_raster(src, path.name).sensor == "HLS L30"


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("hlsl_20230601.tif", "2023-06-01"),
        ("kakhovka_2023-06-09_crop.tif", "2023-06-09"),
        ("scene_2023_06_01.tif", "2023-06-01"),
        ("tile_120230601.tif", None),  # inside a longer digit run
        ("scene_20231301.tif", None),  # month 13
        ("scene.tif", None),
    ],
)
def test_date_from_filename(filename: str, expected: str | None) -> None:
    assert _acquisition_date({}, filename) == expected


def test_tag_date_wins_over_filename() -> None:
    assert _acquisition_date({"acquisition_date": "2023:06:09 08:35:59"}, "hlsl_20230601.tif") == "2023-06-09"


@pytest.mark.parametrize(
    ("codes", "expected"),
    [
        (["B2", "B3", "B4", "B5", "B6"], "B2–B6"),
        (["B2", "B3", "B4", "B8", "B8A", "B11", "B12"], "B2–B4, B8, B8A, B11, B12"),
        (["B4", "B8"], "B4, B8"),
        (["B2", None], "2 bands"),
        ([None], "1 band"),
    ],
)
def test_band_summary(codes: list[str | None], expected: str) -> None:
    assert _band_summary(codes) == expected


@pytest.mark.parametrize(
    ("size", "expected"),
    [(512, "512 B"), (2048, "2.0 KB"), (4_180_159, "4.0 MB"), (3 * 1024**3, "3.0 GB")],
)
def test_format_size(size: int, expected: str) -> None:
    assert format_size(size) == expected
