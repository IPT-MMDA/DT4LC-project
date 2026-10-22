"""Tests for execution planner."""

from unittest.mock import MagicMock, patch

from dta.dti.coe.planner import _detect_index_type, _is_change_detection_request, plan, plan_template
from dta.dti.schemas import ContextUnderstanding, Registry, RegistryItem, Runner, Triggers


def _make_change_detection_registry() -> Registry:
    """Build a minimal registry with the change-detection item for planner tests."""
    from dta.dti.schemas import RegistryItem, Runner, Triggers

    cd_item = RegistryItem(
        id="algorithms/change-detection",
        kind="algorithm",
        runner=Runner(type="python"),
        inputs=["RasterPathBefore", "RasterPathAfter", "IndexType"],
        outputs=["ChangeMap"],
        triggers=Triggers(
            keywords=[
                "change detection",
                "detect changes",
                "compare images",
                "image comparison",
                "temporal comparison",
                "image difference",
                "temporal difference",
                "ndvi change",
                "ndsi change",
                "ndwi change",
                "vegetation change",
                "vegetation changes",
                "snow change",
                "ice change",
                "ice cover change",
                "ice cover changes",
                "water change",
                "glacier change",
                "glacier melt",
                "snow melt",
                "flood detection",
                "flooding",
                "before",
                "after",
            ]
        ),
        config={
            "default_index_type": "ndvi",
            "index_keyword_map": {
                "ndvi": ["ndvi", "vegetation", "vegetation change"],
                "ndsi": ["ndsi", "snow", "ice", "glacier", "frozen", "melt", "ice cover", "ice cover change"],
                "ndwi": ["ndwi", "water", "flood", "lake", "river", "reservoir", "drought"],
            },
        },
    )
    return Registry(version="1.0", types=[], instances=[cd_item])


def test_is_change_detection_request() -> None:
    reg = _make_change_detection_registry()

    # Test positive cases
    ctx = ContextUnderstanding(goal="detect changes", required_inputs=[], desired_outputs=[])
    assert _is_change_detection_request(ctx, reg) is True

    ctx = ContextUnderstanding(goal="compare images", required_inputs=[], desired_outputs=[])
    assert _is_change_detection_request(ctx, reg) is True

    ctx = ContextUnderstanding(goal="analyze before and after", required_inputs=[], desired_outputs=[])
    assert _is_change_detection_request(ctx, reg) is True

    ctx = ContextUnderstanding(
        goal="analysis", hints={"keywords": ["change detection"]}, required_inputs=[], desired_outputs=[]
    )
    assert _is_change_detection_request(ctx, reg) is True

    ctx = ContextUnderstanding(goal="analysis", desired_outputs=["ChangeMap"], required_inputs=[])
    assert _is_change_detection_request(ctx, reg) is True

    # Test negative cases
    ctx = ContextUnderstanding(goal="calculate ndvi", required_inputs=[], desired_outputs=[])
    assert _is_change_detection_request(ctx, reg) is False

    ctx = ContextUnderstanding(goal="find fields", required_inputs=[], desired_outputs=[])
    assert _is_change_detection_request(ctx, reg) is False


def test_detect_index_type() -> None:
    reg = _make_change_detection_registry()

    # Test explicit hint
    ctx = ContextUnderstanding(goal="analysis", hints={"index_type": "ndwi"}, required_inputs=[], desired_outputs=[])
    assert _detect_index_type(ctx, reg) == "ndwi"

    # Test ndsi indicators
    ctx = ContextUnderstanding(goal="detect snow changes", required_inputs=[], desired_outputs=[])
    assert _detect_index_type(ctx, reg) == "ndsi"

    ctx = ContextUnderstanding(goal="glacier melt", required_inputs=[], desired_outputs=[])
    assert _detect_index_type(ctx, reg) == "ndsi"

    # Test ndwi indicators
    ctx = ContextUnderstanding(goal="flood detection", required_inputs=[], desired_outputs=[])
    assert _detect_index_type(ctx, reg) == "ndwi"

    # Test default
    ctx = ContextUnderstanding(goal="vegetation change", required_inputs=[], desired_outputs=[])
    assert _detect_index_type(ctx, reg) == "ndvi"


def test_plan_template_change_detection() -> None:
    reg = Registry(
        version="1.0",
        types=[],
        instances=[
            RegistryItem(
                id="input/file", kind="input", runner=Runner(type="passthrough"), inputs=[], outputs=["RasterPath"]
            ),
            RegistryItem(
                id="input/file-before",
                kind="input",
                runner=Runner(type="passthrough"),
                inputs=[],
                outputs=["RasterPath1"],
            ),
            RegistryItem(
                id="input/file-after",
                kind="input",
                runner=Runner(type="passthrough"),
                inputs=[],
                outputs=["RasterPath2"],
            ),
            RegistryItem(
                id="algorithms/change-detection",
                kind="algorithm",
                runner=Runner(type="python"),
                inputs=["RasterPath1", "RasterPath2"],
                outputs=["ChangeMap"],
                triggers=Triggers(
                    keywords=[
                        "change detection",
                        "detect changes",
                        "vegetation change",
                        "vegetation changes",
                        "before",
                        "after",
                    ]
                ),
                config={
                    "default_index_type": "ndvi",
                    "index_keyword_map": {
                        "ndvi": ["ndvi", "vegetation", "vegetation change"],
                        "ndsi": ["ndsi", "snow", "ice", "glacier"],
                        "ndwi": ["ndwi", "water", "flood"],
                    },
                },
            ),
        ],
    )

    ctx = ContextUnderstanding(goal="detect vegetation change", required_inputs=[], desired_outputs=[])

    result = plan_template(ctx, reg)
    assert result.flow == "detect vegetation change"
    assert len(result.steps) == 3
    assert result.steps[0].uses == "input/file-before"
    assert result.steps[1].uses == "input/file-after"
    assert result.steps[2].uses == "algorithms/change-detection"
    assert result.steps[2].binds["IndexType"] == "ndvi"


def test_plan_template_standard() -> None:
    reg = Registry(
        version="1.0",
        types=[],
        instances=[
            RegistryItem(
                id="input/file", kind="input", runner=Runner(type="passthrough"), inputs=[], outputs=["RasterPath"]
            ),
            RegistryItem(
                id="algorithms/ndvi",
                kind="algorithm",
                runner=Runner(type="python"),
                inputs=["RasterPath"],
                outputs=["NDVIMap"],
                keywords=["ndvi"],
            ),
        ],
    )

    ctx = ContextUnderstanding(
        goal="calculate ndvi", required_inputs=[], desired_outputs=[], hints={"keywords": ["ndvi"]}
    )

    result = plan_template(ctx, reg)
    assert result.flow == "calculate ndvi"
    # Should include input/file because algorithm needs RasterPath
    assert len(result.steps) == 2
    assert result.steps[0].uses == "input/file"
    assert result.steps[1].uses == "algorithms/ndvi"


def test_plan_template_no_inputs() -> None:
    reg = Registry(
        version="1.0",
        types=[],
        instances=[
            RegistryItem(
                id="algorithms/standalone",
                kind="algorithm",
                runner=Runner(type="python"),
                inputs=[],
                outputs=["Result"],
                keywords=["standalone"],
            ),
        ],
    )

    ctx = ContextUnderstanding(
        goal="run standalone", required_inputs=[], desired_outputs=[], hints={"keywords": ["standalone"]}
    )

    result = plan_template(ctx, reg)
    # Should NOT include input/file because algorithm needs no inputs
    assert len(result.steps) == 1
    assert result.steps[0].uses == "algorithms/standalone"


@patch("dta.dti.coe.planner.plan_template")
def test_plan_routing_to_template(mock_template: MagicMock) -> None:
    ctx = ContextUnderstanding(goal="test", required_inputs=[], desired_outputs=[])
    reg = Registry(version="1.0", types=[], instances=[])

    with patch("dta.dti.coe.llm_planner.estimate_plan_confidence", return_value=0.9):
        plan(ctx, reg, use_llm=True)
        mock_template.assert_called_once_with(ctx, reg)


@patch("dta.dti.coe.llm_planner.plan_with_llm")
def test_plan_routing_to_llm(mock_llm: MagicMock) -> None:
    ctx = ContextUnderstanding(goal="test", required_inputs=[], desired_outputs=[])
    reg = Registry(version="1.0", types=[], instances=[])

    with patch("dta.dti.coe.llm_planner.estimate_plan_confidence", return_value=0.4):
        plan(ctx, reg, use_llm=True)
        mock_llm.assert_called_once_with(ctx, reg)


@patch("dta.dti.coe.planner.plan_template")
def test_plan_routing_llm_error_fallback(mock_template: MagicMock) -> None:
    ctx = ContextUnderstanding(goal="test", required_inputs=[], desired_outputs=[])
    reg = Registry(version="1.0", types=[], instances=[])

    with patch("dta.dti.coe.llm_planner.estimate_plan_confidence", side_effect=Exception("Error")):
        plan(ctx, reg, use_llm=True)
        mock_template.assert_called_once_with(ctx, reg)
