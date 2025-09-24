from pathlib import Path

import yaml

from dta.config import REGISTRY_PATH

from .schemas import Registry, RegistryItem


def load_registry(path: Path = REGISTRY_PATH) -> Registry:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Registry(**data)


def find_items_by_keywords(reg: Registry, keywords: list[str]) -> list[RegistryItem]:
    ks = {k.lower() for k in keywords}

    def score(item: RegistryItem) -> int:
        return len(ks.intersection({w.lower() for w in item.keywords}))

    return sorted(reg.instances, key=score, reverse=True)


def find_items_producing(reg: Registry, out_type: str) -> list[RegistryItem]:
    return [i for i in reg.instances if out_type in i.outputs]


def get_item(reg: Registry, item_id: str) -> RegistryItem:
    return next(i for i in reg.instances if i.id == item_id)
