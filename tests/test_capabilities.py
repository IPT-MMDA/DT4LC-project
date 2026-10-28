"""``GET /v1/capabilities`` reproduces the two server states of the v0.6.0 design.

(a) public demo: ``capabilities-demo-top.png``, ``capabilities-demo-models-data.png`` and
``capabilities-prithvi-gated.png`` (guest, Gemini connected, Delineate installed, Prithvi not).
(b) offline laptop: ``capabilities-offline-top.png`` (member, no assistant, no models, no EE).
"""

from collections.abc import Iterator
from pathlib import Path
import socket
import time
from typing import Any
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
import pytest

from dta.dti.data_sources import gee_common
from dta.dti.models import model_states as real_model_states
from server.app import app
import server.routes.health as health

DELINEATE = "delineate-anything-small"
PRITHVI = "prithvi-eo-v1-100m"


def _provider(name: str, model: str) -> MagicMock:
    provider = MagicMock()
    provider.name = name
    provider.model = model
    provider.is_available.return_value = True
    return provider


@pytest.fixture(autouse=True)
def _defaults(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(health, "_assistant_cache", None)
    for name in ("DT4LC_DEPLOYMENT", "MODEL_DOWNLOAD_ON_DEMAND", "GUEST_ANALYSES_PER_HOUR", "GUEST_SESSION_HOURS"):
        monkeypatch.delenv(name, raising=False)
    yield


def _capabilities(*, providers: list[MagicMock], installed: set[str], ee: bool) -> dict[str, Any]:
    router = MagicMock()
    router.providers = providers
    states = [{**s, "installed": s["id"] in installed} for s in real_model_states()]
    with (
        patch("server.routes.health.get_llm_router", return_value=router),
        patch("server.routes.health.model_states", return_value=states),
        patch("server.routes.health.ee_is_configured", return_value=ee),
    ):
        resp = TestClient(app).get("/v1/capabilities")
    assert resp.status_code == 200
    body: dict[str, Any] = resp.json()
    return body


def _tier(doc: dict[str, Any], tier_id: str) -> dict[str, Any]:
    tier: dict[str, Any] = next(t for t in doc["tiers"] if t["id"] == tier_id)
    return tier


def _item(doc: dict[str, Any], tier_id: str, item_id: str) -> dict[str, Any]:
    item: dict[str, Any] = next(i for i in _tier(doc, tier_id)["items"] if i["id"] == item_id)
    return item


@pytest.fixture
def demo(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """State (a): public demo, guest, Gemini 3.5 Flash, Delineate installed, Prithvi not, EE configured."""
    monkeypatch.setenv("DT4LC_DEPLOYMENT", "demo")
    monkeypatch.setenv("DEV_ROLE", "guest")
    return _capabilities(providers=[_provider("gemini", "gemini-3.5-flash")], installed={DELINEATE}, ee=True)


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """State (b): local server, member, no assistant, no models (members may download), no EE."""
    monkeypatch.setenv("DT4LC_DEPLOYMENT", "local")
    monkeypatch.setenv("DEV_ROLE", "member")
    monkeypatch.setenv("MODEL_DOWNLOAD_ON_DEMAND", "members")
    return _capabilities(providers=[], installed=set(), ee=False)


# --- (a) public demo -----------------------------------------------------------


def test_demo_summary_line(demo: dict[str, Any]) -> None:
    assert demo["summary"] == {
        "deployment": "demo",
        "parts": {
            "deployment": "public demo",
            "role": "guest access",
            "assistant": "Gemini 3.5 Flash connected",
            "models": "Delineate-Anything installed",
            "earth_engine": "Earth Engine configured",
        },
        "line": "public demo · guest access · Gemini 3.5 Flash connected · Delineate-Anything installed"
        " · Earth Engine configured",
    }
    assert demo["assistant"] == {
        "connected": True,
        "provider": "gemini",
        "model": "gemini-3.5-flash",
        "display": "Gemini 3.5 Flash",
    }


def test_demo_tiers_in_order_and_not_greyed(demo: dict[str, Any]) -> None:
    assert [(t["id"], t["number"], t["title"]) for t in demo["tiers"]] == [
        ("ask", 1, "Ask"),
        ("composed", 2, "Composed analyses"),
        ("builtin", 3, "Built-in analyses"),
        ("models", 4, "Models"),
        ("data", 5, "Data sources"),
        ("coming", 6, "Coming"),
    ]
    assert demo["offline"] is False
    assert demo["banner"] is None
    ask = _tier(demo, "ask")
    assert ask["status"] == "connected"
    assert ask["status_label"] == "Assistant · Gemini 3.5 Flash · connected"
    assert ask["subtitle"] == "Questions, clarifications, explanations of methods and results, follow-ups."
    assert [i["prompt"] for i in ask["items"]] == [
        "Why is NDWI better than NDVI for water?",
        "Explain the last result in two sentences",
        "What would you do differently with Landsat?",
    ]
    assert _tier(demo, "composed")["status"] == "available"
    assert [t["needs_assistant"] for t in demo["tiers"]] == [True, True, False, False, False, False]


def test_demo_composed_tier(demo: dict[str, Any]) -> None:
    composed = _tier(demo, "composed")
    assert composed["subtitle"] == "The assistant combines the built-in steps into new pipelines."
    assert composed["badge"] == "Composed"
    assert [(i["title"], i["recipe"], i["prompt"]) for i in composed["items"]] == [
        (
            "Any spectral index",
            "NDMI = (B5 − B6)/(B5 + B6) · NBR · SAVI · NDBI · custom",
            "Map moisture change with NDMI between the two dates",
        ),
        ("Masked classification", "ndwi_change → mask → lulc → stats", "Classify land cover on the flooded fields"),
        ("Multi-date comparison", "index(t₁…tₙ) → Δ per step → table", "Track water weekly through June"),
        ("Parameter changes", "threshold 0.2 → 0.1 · cloud ≤10 → ≤25 %", "Redo the water map with a 0.1 threshold"),
    ]
    first = composed["items"][0]
    assert first["description"] == "Band math for indices not in the catalog, from the bands the image has."
    assert all(i["badge"] == "Composed" and i["needs_label"] == "Needs the assistant" for i in composed["items"])


def test_demo_builtin_spectral_indices(demo: dict[str, Any]) -> None:
    builtin = _tier(demo, "builtin")
    assert builtin["subtitle"] == "Run without the assistant — the catalog every installation has."
    assert [g["title"] for g in builtin["groups"]] == [
        "Spectral indices",
        "Change, classification, statistics",
        "Model-based",
    ]
    assert _item(demo, "builtin", "algorithms/ndvi") == {
        "id": "algorithms/ndvi",
        "group": "spectral",
        "title": "NDVI",
        "formula": "(NIR−Red)/(NIR+Red)",
        "description": "Vegetation density and health.",
        "inputs": ["Red", "NIR"],
        "outputs": ["Index map", "legend", "stats"],
        "time_estimate": "~10 s",
        "runs_without_assistant": True,
        "prompt": "Map vegetation health across the region",
        "model": None,
        "status": "available",
        "status_label": "Runs without the assistant",
    }
    rows = {
        i["id"]: (
            i["title"],
            i["formula"],
            i["description"],
            i["inputs"],
            i["outputs"],
            i["time_estimate"],
            i["prompt"],
        )
        for i in builtin["items"]
        if i["group"] == "spectral"
    }
    assert list(rows) == ["algorithms/ndvi", "algorithms/evi", "algorithms/ndwi", "algorithms/ndsi"]
    assert rows["algorithms/evi"] == (
        "EVI",
        "2.5·(NIR−Red)/(NIR+6·Red−7.5·Blue+1)",
        "Vegetation in dense canopy where NDVI saturates.",
        ["Blue", "Red", "NIR"],
        ["Index map", "legend", "stats"],
        "~10 s",
        "Use EVI instead of NDVI for the forest",
    )
    assert rows["algorithms/ndwi"] == (
        "NDWI",
        "(Green−NIR)/(Green+NIR)",
        "Open water and surface moisture.",
        ["Green", "NIR"],
        ["Index map", "water mask", "stats"],
        "~10 s",
        "Where is open water on 9 June?",
    )
    assert rows["algorithms/ndsi"] == (
        "NDSI",
        "(Green−SWIR)/(Green+SWIR)",
        "Snow and ice; values above 0.42 usually mean snow.",
        ["Green", "SWIR1"],
        ["Index map", "snow mask", "stats"],
        "~10 s",
        "Map snow cover with NDSI",
    )


def test_demo_builtin_change_classification_statistics(demo: dict[str, Any]) -> None:
    rows = [
        (i["title"], i["inputs"], i["outputs"], i["time_estimate"], i["prompt"], i["status_label"])
        for i in _tier(demo, "builtin")["items"]
        if i["group"] == "change"
    ]
    runs = "Runs without the assistant"
    assert rows == [
        (
            "Change detection",
            ["Two dates", "same bands"],
            ["Difference map", "3 classes", "areas"],
            "~50 s",
            "Compare water before and after the dam breach",
            runs,
        ),
        (
            "Land-cover classification",
            ["Blue", "Green", "Red", "NIR (+SWIR)"],
            ["Class map", "legend", "per-class area"],
            "~90 s",
            "Classify land cover on the flooded fields",
            runs,
        ),
        (
            "Snow classifier",
            ["Green", "Red", "NIR", "SWIR1"],
            ["Snow / ice / other map", "areas"],
            "~20 s",
            "Classify snow and ice robustly",
            runs,
        ),
        (
            "Raster statistics",
            ["Any raster"],
            ["Table", "histogram", "CSV"],
            "~5 s",
            "Give me statistics for every band",
            runs,
        ),
    ]
    snow = _item(demo, "builtin", "algorithms/snow-classifier")
    assert snow["description"] == "Multi-criteria: NDSI ≥ 0.4, NDVI below 0.1 and brightness above 0.3."
    assert all(i["runs_without_assistant"] is True for i in _tier(demo, "builtin")["items"])


def test_demo_models_and_their_analyses(demo: dict[str, Any]) -> None:
    """``capabilities-demo-models-data.png`` + ``capabilities-prithvi-gated.png``: Prithvi gated for guests."""
    models = _tier(demo, "models")
    assert models["subtitle"] == "Downloaded weights that some built-in analyses need."
    assert [
        (m["id"], m["title"], m["description"], m["size_label"], m["licence"], m["status"], m["status_label"])
        for m in models["items"]
    ] == [
        (
            DELINEATE,
            "Delineate-Anything",
            "Field-boundary detection for agricultural parcels.",
            "85 MB",
            "AGPL-3.0",
            "installed",
            "Installed on this server",
        ),
        (
            PRITHVI,
            "Prithvi EO v1 (100M)",
            "NASA/IBM foundation model; reconstructs masked or cloudy pixels.",
            "450 MB",
            "Apache-2.0",
            "not_available_for_guests",
            "Not available for guests — ask an operator",
        ),
    ]
    assert models["items"][0]["used_by"] == ["models/delineate-anything"]

    fields = _item(demo, "builtin", "models/delineate-anything")
    assert (
        fields["title"],
        fields["model"]["short"],
        fields["inputs"],
        fields["outputs"],
        fields["time_estimate"],
    ) == (
        "Field boundaries",
        "Delineate-Anything",
        ["Red", "Green", "Blue (10–30 m)"],
        ["GeoPackage", "count", "area histogram"],
        "~2 min",
    )
    assert (fields["status"], fields["status_label"]) == ("installed", "Runs without the assistant")
    assert fields["prompt"] == "Detect field boundaries east of the river"

    prithvi = _item(demo, "builtin", "models/prithvi-reconstruction")
    assert (prithvi["title"], prithvi["model"]["short"], prithvi["inputs"], prithvi["time_estimate"]) == (
        "Prithvi reconstruction",
        "Prithvi EO v1",
        ["6 HLS bands", "cloud mask"],
        "~3 min",
    )
    assert prithvi["description"] == "Fills masked or cloudy pixels from the surrounding dates."
    assert prithvi["outputs"] == ["Reconstructed GeoTIFF", "mask band"]
    assert (prithvi["status"], prithvi["status_label"]) == (
        "not_available_for_guests",
        "Not available for guests — ask an operator",
    )
    assert prithvi["prompt"] == "Fill the cloud gaps in the 9 June image"


def test_demo_data_sources_and_guest_limits(demo: dict[str, Any]) -> None:
    data = _tier(demo, "data")
    assert data["subtitle"] == "Where imagery can come from on this server."
    assert [(i["title"], i["detail"], i["status"], i["status_label"]) for i in data["items"]] == [
        ("Uploads", "GeoTIFF, up to 200 MB per file · any projection", "always", "Always"),
        ("Bundled sample", "kakhovka_hls_l30 · HLS L30 · 2023-06-01 → 06-09", "always", "Always"),
        (
            "Earth Engine · Sentinel-2 / Landsat / MODIS",
            "Fetch by region, dates, bands and cloud filter from the Map",
            "configured",
            "Configured",
        ),
    ]
    assert data["note"] == "Guests: up to 5 analyses per hour, results kept 24 h in this browser, no account needed."
    assert demo["guest_limits"] == {"analyses_per_hour": 5, "kept_hours": 24}
    session = demo["session"]
    assert (session["role"], session["guest"], session["analyses"]["limit"], session["kept_hours"]) == (
        "guest",
        True,
        5,
        24,
    )
    assert session["can_download_models"] is False


def test_demo_coming_tier_is_collapsed_with_whole_descriptions(demo: dict[str, Any]) -> None:
    coming = _tier(demo, "coming")
    assert coming["collapsed"] is True
    assert coming["status"] == "coming"
    assert [i["title"] for i in coming["items"]] == [
        "ConvLSTM NDVI forecast",
        "PINN-LINDA nowcasting",
        "VI forecasting DTI",
        "Runoff ensemble",
    ]
    assert coming["items"][3] == {
        "id": "models/runoff-ensemble",
        "title": "Runoff ensemble",
        "description": "Ensemble runoff estimates from precipitation and land-cover inputs.",
    }
    # The whole registry description, not its first line.
    assert coming["items"][0]["description"].endswith(
        "Trained on Cherkasy, Vinnytsia, and Mykolaiv oblasts (2018-2021)."
    )


def test_demo_example_prompts_are_full_prompts(demo: dict[str, Any]) -> None:
    prompts = {(p["tier"], p["id"]): p["prompt"] for p in demo["example_prompts"]}
    assert prompts[("ask", "ndwi-vs-ndvi")] == "Why is NDWI better than NDVI for water?"
    assert prompts[("composed", "any-index")] == "Map moisture change with NDMI between the two dates"
    assert prompts[("builtin", "algorithms/ndvi")] == "Map vegetation health across the region"
    assert len(prompts) == 3 + 4 + 10


# --- (b) offline laptop ----------------------------------------------------------


def test_offline_summary_line(offline: dict[str, Any]) -> None:
    """``capabilities-offline-top.png`` shows no "local server" in the line."""
    assert offline["summary"]["line"] == (
        "member account · assistant offline · models download on first use · no Earth Engine"
    )
    assert offline["summary"]["deployment"] == "local"
    assert offline["summary"]["parts"]["deployment"] == "local server"
    assert offline["assistant"] == {"connected": False, "provider": None, "model": None, "display": None}


def test_offline_banner_and_greyed_tiers(offline: dict[str, Any]) -> None:
    assert offline["offline"] is True
    assert offline["banner"] == (
        "The assistant is offline on this server, so tiers 1 and 2 are unavailable. "
        "Built-in analyses run as usual; answers are limited to the catalog."
    )
    ask = _tier(offline, "ask")
    assert (ask["status"], ask["status_label"]) == ("offline", "Assistant offline — limited answers")
    assert _tier(offline, "composed")["status"] == "offline"
    # Greyed, never hidden: the example questions and composed recipes are still there.
    assert len(ask["items"]) == 3
    assert len(_tier(offline, "composed")["items"]) == 4
    assert _item(offline, "builtin", "algorithms/ndvi")["status"] == "available"


def test_offline_models_download_on_first_use(offline: dict[str, Any]) -> None:
    prithvi_label = "Downloads on first use · 450 MB · ~75 s · Apache-2.0"
    assert [(m["id"], m["status"], m["status_label"]) for m in _tier(offline, "models")["items"]] == [
        (DELINEATE, "downloads_on_first_use", "Downloads on first use · 85 MB · ~14 s · AGPL-3.0"),
        (PRITHVI, "downloads_on_first_use", prithvi_label),
    ]
    prithvi = _item(offline, "builtin", "models/prithvi-reconstruction")
    assert (prithvi["status"], prithvi["status_label"]) == ("downloads_on_first_use", prithvi_label)


def test_offline_no_earth_engine(offline: dict[str, Any]) -> None:
    ee = _item(offline, "data", "earth-engine")
    assert (ee["status"], ee["status_label"]) == ("not_configured", "Not configured")
    session = offline["session"]
    assert (session["role"], session["guest"], session["analyses"]["limit"]) == ("member", False, None)
    assert session["can_download_models"] is True


def test_member_data_note_is_not_the_guest_line(offline: dict[str, Any]) -> None:
    """The Data tier's note follows the caller's role (prototype ``guestLimitLine``)."""
    assert _tier(offline, "data")["note"] == (
        "Member on a local server: no analysis limit; results are kept until you delete them."
    )


def test_operator_data_note_on_the_demo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DT4LC_DEPLOYMENT", "demo")
    monkeypatch.setenv("DEV_ROLE", "operator")
    doc = _capabilities(providers=[], installed=set(), ee=False)
    assert _tier(doc, "data")["note"] == (
        "Operator on the public demo: no analysis limit; results are kept until you delete them."
    )


# --- model statuses per role and policy --------------------------------------------


def test_guest_without_models_is_gated_on_the_model_and_its_analysis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEV_ROLE", "guest")
    doc = _capabilities(providers=[], installed=set(), ee=False)
    gated = ("not_available_for_guests", "Not available for guests — ask an operator")
    assert [(m["status"], m["status_label"]) for m in _tier(doc, "models")["items"]] == [gated, gated]
    for step in ("models/delineate-anything", "models/prithvi-reconstruction"):
        item = _item(doc, "builtin", step)
        assert (item["status"], item["status_label"]) == gated
    assert doc["summary"]["parts"]["models"] == "no models installed"


@pytest.mark.parametrize(
    ("role", "policy"),
    [("guest", "none"), ("member", "none"), ("member", "operators"), ("guest", "operators")],
)
def test_no_download_policy_is_not_installed(monkeypatch: pytest.MonkeyPatch, role: str, policy: str) -> None:
    monkeypatch.setenv("DEV_ROLE", role)
    monkeypatch.setenv("MODEL_DOWNLOAD_ON_DEMAND", policy)
    doc = _capabilities(providers=[], installed=set(), ee=False)
    expected = ("not_installed", "Not installed on this server — ask an operator")
    assert {(m["status"], m["status_label"]) for m in _tier(doc, "models")["items"]} == {expected}
    assert (_item(doc, "builtin", "models/prithvi-reconstruction")["status"]) == "not_installed"


def test_operator_with_both_models_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEV_ROLE", "operator")
    doc = _capabilities(
        providers=[_provider("groq", "llama-3.3-70b-versatile")], installed={DELINEATE, PRITHVI}, ee=True
    )
    assert doc["summary"]["line"] == (
        "operator · Llama 3.3 70B connected · Delineate-Anything and Prithvi EO v1 (100M) installed"
        " · Earth Engine configured"
    )
    assert _item(doc, "builtin", "models/prithvi-reconstruction")["status_label"] == "Runs without the assistant"


def test_unknown_deployment_and_model_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DT4LC_DEPLOYMENT", "production")
    doc = _capabilities(providers=[_provider("ollama", "qwen3:8b")], installed=set(), ee=False)
    assert doc["summary"]["deployment"] == "local"
    assert doc["assistant"]["display"] == "qwen3:8b"
    assert doc["summary"]["parts"]["assistant"] == "qwen3:8b connected"


# --- cheap: no network, fast when warm --------------------------------------------


def test_capabilities_is_fast_and_offline_when_warm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEV_ROLE", "guest")
    router = MagicMock()
    router.providers = [_provider("gemini", "gemini-3.5-flash")]

    def no_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("GET /v1/capabilities opened a network connection")

    client = TestClient(app)
    with patch("server.routes.health.get_llm_router", return_value=router):
        client.get("/v1/health")  # warms the assistant cache
        monkeypatch.setattr(socket.socket, "connect", no_network)
        timings = []
        for _ in range(5):
            start = time.perf_counter()
            assert client.get("/v1/capabilities").status_code == 200
            timings.append(time.perf_counter() - start)
    assert min(timings) < 0.1
    assert router.providers[0].is_available.call_count == 1


# --- Earth Engine configuration check ------------------------------------------------


def test_ee_is_configured_needs_project_and_key_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gee_common, "_ee_initialized", False)
    key = tmp_path / "key.json"
    monkeypatch.setenv("GEE_PROJECT_ID", "demo-project")
    monkeypatch.setenv("GEE_SERVICE_ACCOUNT_KEY", str(key))
    assert gee_common.is_configured() is False  # key file missing
    key.write_text("{}")
    assert gee_common.is_configured() is True
    monkeypatch.delenv("GEE_PROJECT_ID")
    assert gee_common.is_configured() is False
    monkeypatch.setattr(gee_common, "_ee_initialized", True)
    assert gee_common.is_configured() is True
