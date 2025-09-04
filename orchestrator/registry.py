from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


@dataclass
class Capability:
    id: str
    kind: str
    inputs: list[str]
    outputs: list[str]
    tags: list[str]
    params: dict[str, Any]


class CapabilitiesRegistry:
    def __init__(self, path: Path):
        self._path = path
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        self._tools: dict[str, Capability] = {}
        for item in raw.get("tools", []):
            cap = Capability(
                id=item["id"],
                kind=item.get("kind", "tool"),
                inputs=list(item.get("inputs", [])),
                outputs=list(item.get("outputs", [])),
                tags=list(item.get("tags", [])),
                params=dict(item.get("params", {})),
            )
            self._tools[cap.id] = cap

    def find_by_tag(self, *tags: str) -> list[Capability]:
        tagset = set(tags)
        return [c for c in self._tools.values() if tagset.issubset(set(c.tags))]

    def get(self, cap_id: str) -> Capability:
        return self._tools[cap_id]

    def exists(self, cap_id: str) -> bool:
        return cap_id in self._tools
