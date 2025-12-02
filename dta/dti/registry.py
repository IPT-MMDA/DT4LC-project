"""Registry loading and querying utilities.

Functions for loading the component registry from YAML and searching
for items by keywords or output types.
"""

from pathlib import Path

import yaml

from dta.config import REGISTRY_PATH

from .schemas import Registry, RegistryItem


def load_registry(path: Path = REGISTRY_PATH) -> Registry:
    """Load component registry from YAML file."""
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Registry(**data)


def find_items_by_keywords(reg: Registry, keywords: list[str]) -> list[RegistryItem]:
    """Find registry items matching keywords, sorted by relevance."""
    ks = {k.lower() for k in keywords}

    def score(item: RegistryItem) -> int:
        return len(ks.intersection({w.lower() for w in item.keywords}))

    return sorted(reg.instances, key=score, reverse=True)


def find_items_producing(reg: Registry, out_type: str) -> list[RegistryItem]:
    """Find registry items that produce the specified output type."""
    return [i for i in reg.instances if out_type in i.outputs]


def get_item(reg: Registry, item_id: str) -> RegistryItem:
    """Get registry item by ID.

    Args:
        reg: Registry
        item_id: Item identifier

    Returns:
        Registry item

    Raises:
        KeyError: If item not found
    """
    try:
        return next(i for i in reg.instances if i.id == item_id)
    except StopIteration:
        raise KeyError(f"Registry item not found: {item_id}") from None
