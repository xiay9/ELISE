from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .utils import update_config


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config at {path} must be a top-level mapping.")
    base_ref = payload.pop("_base_", None)
    if base_ref is None:
        return payload
    base_path = (path.parent / base_ref).resolve()
    base_payload = _load_yaml(base_path)
    return _deep_merge_dicts(base_payload, payload)


def load_paceext_config(config_path: str | Path, overrides: dict[str, Any] | None = None) -> AttrDict:
    config_path = Path(config_path).resolve()
    payload = AttrDict(_load_yaml(config_path))
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                payload[key] = value
    return update_config(payload)
