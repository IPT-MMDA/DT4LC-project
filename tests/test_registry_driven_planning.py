"""Extensibility tests for registry-driven context_agent + planner.

Pins the load-bearing property: the change-detection trigger keywords and
per-index keyword map come from ``algorithms/change-detection`` in the
registry, and the context_agent SYS prompt's DOMAIN KNOWLEDGE block is
rendered from each user-runnable item's ``triggers`` + ``outputs``.

Adding a new item or new index keyword to ``registry.yaml`` must be picked up
by the planner and the context-agent prompt **without any code change**.
"""

from __future__ import annotations

from dta.dti.coe.context_agent import _build_sys_prompt, _render_domain_knowledge
from dta.dti.coe.planner import _detect_index_type, _is_change_detection_request
from dta.dti.registry import load_registry
from dta.dti.schemas import (
    ContextUnderstanding,
    Registry,
    RegistryItem,
    Runner,
    Triggers,
    UserGuide,
)


def _ctx(goal: str, keywords: list[str], desired: list[str] | None = None) -> ContextUnderstanding:
    return ContextUnderstanding(
        goal=goal,
        required_inputs=[],
        desired_outputs=desired or [],
        hints={"keywords": keywords},
    )


def _registry_with_extra_change_keyword(extra_keyword: str = "fictional-rot") -> Registry:
    """Real registry, but with one extra trigger keyword on change-detection."""
    real = load_registry()
    instances = []
    for it in real.instances:
        if it.id == "algorithms/change-detection":
            new_triggers = Triggers(
                keywords=[*it.triggers.keywords, extra_keyword] if it.triggers else [extra_keyword],
                action_phrases=it.triggers.action_phrases if it.triggers else [],
            )
            instances.append(it.model_copy(update={"triggers": new_triggers}))
        else:
            instances.append(it)
    return Registry(version=real.version, types=real.types, instances=instances)


def _registry_with_extra_index(name: str, kws: list[str]) -> Registry:
    """Real registry, but with an extra entry in change-detection.config.index_keyword_map."""
    real = load_registry()
    instances = []
    for it in real.instances:
        if it.id == "algorithms/change-detection":
            new_config = dict(it.config)
            new_map = dict(new_config.get("index_keyword_map", {}))
            new_map[name] = kws
            new_config["index_keyword_map"] = new_map
            instances.append(it.model_copy(update={"config": new_config}))
        else:
            instances.append(it)
    return Registry(version=real.version, types=real.types, instances=instances)


class TestPlannerChangeDetectionDrivenByRegistry:
    """`_is_change_detection_request` must read trigger keywords from registry."""

    def test_existing_change_keyword_still_detected(self) -> None:
        # Existing keyword from change-detection.triggers (in registry.yaml).
        ctx = _ctx("Detect changes in this image", ["change detection"])
        assert _is_change_detection_request(ctx, load_registry())

    def test_new_change_keyword_picked_up_without_code_change(self) -> None:
        # Adding a fresh keyword to change-detection.triggers is enough for
        # the planner to recognize it — no edit to planner.py.
        reg = _registry_with_extra_change_keyword("fictional-rot")
        ctx = _ctx("please run a fictional-rot analysis on these images", ["fictional-rot"])
        assert _is_change_detection_request(ctx, reg)

    def test_before_after_structural_signal_preserved(self) -> None:
        # The "before AND after" co-occurrence stays in code (it's generic
        # English, not algorithm-specific).
        ctx = _ctx("compare the before and after images", [])
        assert _is_change_detection_request(ctx, load_registry())

    def test_changemap_desired_output_signal_preserved(self) -> None:
        ctx = _ctx("just give me the change map", [], desired=["ChangeMap"])
        assert _is_change_detection_request(ctx, load_registry())

    def test_unrelated_request_returns_false(self) -> None:
        ctx = _ctx("calculate ndvi for this image", ["ndvi"])
        assert not _is_change_detection_request(ctx, load_registry())


class TestPlannerIndexDetectionDrivenByRegistry:
    """`_detect_index_type` must read from change-detection.config.index_keyword_map."""

    def test_ice_cover_maps_to_ndsi(self) -> None:
        # The "ice cover" → ndsi mapping lives in registry.yaml's
        # change-detection.config.index_keyword_map.
        ctx = _ctx("analyze ice cover changes", ["ice cover"])
        assert _detect_index_type(ctx, load_registry()) == "ndsi"

    def test_water_change_maps_to_ndwi(self) -> None:
        ctx = _ctx("analyze water changes", ["water"])
        assert _detect_index_type(ctx, load_registry()) == "ndwi"

    def test_default_when_nothing_matches(self) -> None:
        ctx = _ctx("compare these images", [])
        # default_index_type from change-detection.config.
        assert _detect_index_type(ctx, load_registry()) == "ndvi"

    def test_explicit_hint_overrides_keyword_match(self) -> None:
        # "vegetation" would match ndvi; explicit hint says ndwi.
        ctx = ContextUnderstanding(
            goal="vegetation comparison",
            required_inputs=[],
            desired_outputs=[],
            hints={"keywords": ["vegetation"], "index_type": "ndwi"},
        )
        assert _detect_index_type(ctx, load_registry()) == "ndwi"

    def test_new_index_picked_up_without_code_change(self) -> None:
        # Add a fictional new index with a unique keyword that doesn't
        # collide with existing ndvi/ndsi/ndwi triggers.
        reg = _registry_with_extra_index("fictiondex", ["fictiondex-marker"])
        ctx = _ctx("analyze fictiondex-marker changes please", ["fictiondex-marker"])
        assert _detect_index_type(ctx, reg) == "fictiondex"


class TestPlannerWithoutChangeDetectionItem:
    """If the registry has no change-detection item, the planner still works.

    Test registries used by tests/test_planner.py omit change-detection;
    the public `plan_template` entrypoint must keep handling them gracefully.
    """

    def test_is_change_detection_returns_false(self) -> None:
        bare = Registry(version="1.0", types=["X"], instances=[])
        ctx = _ctx("change detection please", ["change detection"])
        assert not _is_change_detection_request(ctx, bare)

    def test_detect_index_type_returns_default(self) -> None:
        bare = Registry(version="1.0", types=["X"], instances=[])
        ctx = _ctx("snow change", ["snow"])
        # Default falls through to "ndvi" when no change-detection item.
        assert _detect_index_type(ctx, bare) == "ndvi"


class TestContextAgentDomainKnowledgeRendered:
    """Verify the SYS prompt's DOMAIN KNOWLEDGE block comes from the registry."""

    def test_real_registry_renders_all_algorithms(self) -> None:
        domain = _render_domain_knowledge(load_registry())
        # Every user-runnable item with triggers should appear in the rendered block.
        for label in ("NDVIMap", "EVIMap", "NDWIMap", "NDSIMap", "LULCMap", "ChangeMap"):
            assert label in domain, f"{label} should appear in DOMAIN KNOWLEDGE"

    def test_change_detection_index_map_rendered(self) -> None:
        domain = _render_domain_knowledge(load_registry())
        # Per-index hint mappings come from config.index_keyword_map.
        for hint in ('hints.index_type="ndvi"', 'hints.index_type="ndsi"', 'hints.index_type="ndwi"'):
            assert hint in domain, f"{hint!r} should appear in change-detection bullets"

    def test_evi_appears_in_domain_knowledge(self) -> None:
        # Regression guard: every registered algorithm with triggers should
        # be reachable via the rendered SYS prompt — including EVI, which
        # was historically prone to being forgotten when this list was
        # hand-maintained.
        domain = _render_domain_knowledge(load_registry())
        assert "evi" in domain.lower(), "EVI must appear in DOMAIN KNOWLEDGE rendered from registry"

    def test_new_item_appears_in_domain_knowledge(self) -> None:
        # Synthesise a registry with a fictional new algorithm; assert
        # _render_domain_knowledge picks it up without any code change.
        real = load_registry()
        fictional = RegistryItem(
            id="algorithms/fictional-mood",
            kind="algorithm",
            display_name="Fictional Mood Index",
            keywords=["mood"],
            inputs=["RasterPath"],
            outputs=["MoodMap"],
            runner=Runner(type="python", entrypoint="never/loaded.py"),
            triggers=Triggers(keywords=["mood", "vibes", "fictional-mood-keyword"]),
            user_guide=UserGuide(),
        )
        reg = Registry(version=real.version, types=[*real.types, "MoodMap"], instances=[*real.instances, fictional])
        domain = _render_domain_knowledge(reg)
        assert "fictional-mood-keyword" in domain
        assert "MoodMap" in domain

    def test_build_sys_prompt_includes_registry_types_marker(self) -> None:
        prompt = _build_sys_prompt(load_registry(), ["TypeA", "TypeB"])
        assert "[REGISTRY_TYPES]=" in prompt
        assert "TypeA" in prompt and "TypeB" in prompt
