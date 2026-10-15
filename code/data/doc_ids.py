from __future__ import annotations

from functools import lru_cache
import pickle
import json
import re
from pathlib import Path
from typing import Any

from data.schema import LoadedSplit


MEC4_DIALOGUE_ID_PATTERN = re.compile(r"^S(?P<season>\d+)E(?P<episode>\d+)_(?P<start>\d+)_(?P<end>\d+)$")


def attach_doc_ids(loaded_split: LoadedSplit) -> LoadedSplit:
    mapping = load_raw_to_doc_id_map(loaded_split.root, loaded_split.dataset)
    for dialogue in loaded_split.dialogues:
        dialogue.metadata.setdefault("raw_doc_id", dialogue.dialogue_id)
        if dialogue.metadata.get("doc_id") is not None:
            continue
        doc_id = resolve_dialogue_doc_id(
            dialogue_id=dialogue.dialogue_id,
            dataset_name=loaded_split.dataset,
            mapping=mapping,
        )
        if doc_id is not None:
            dialogue.metadata["doc_id"] = int(doc_id)
    return loaded_split


def resolve_dialogue_doc_id(
    *,
    dialogue_id: str,
    dataset_name: str,
    mapping: dict[str, int] | None = None,
) -> int | None:
    raw_id = str(dialogue_id)
    if mapping and raw_id in mapping:
        return int(mapping[raw_id])

    numeric = _try_parse_int(raw_id)
    if numeric is not None:
        return numeric

    if dataset_name.lower() == "mec4":
        match = MEC4_DIALOGUE_ID_PATTERN.fullmatch(raw_id)
        if match:
            return int(
                f"{int(match.group('season'))}"
                f"{int(match.group('episode')):03d}"
                f"{int(match.group('start')):03d}"
                f"{int(match.group('end')):03d}"
            )
    return None


@lru_cache(maxsize=16)
def load_raw_to_doc_id_map(root: str | Path, dataset_name: str) -> dict[str, int]:
    root = Path(root)
    json_path = root / f"{dataset_name.lower()}_raw_to_doc_id.json"
    if json_path.exists():
        return {str(k): int(v) for k, v in json.loads(json_path.read_text()).items()}
    mapping_path = root / f"{dataset_name.lower()}_raw_to_doc_id.pkl"
    if mapping_path.exists():
        payload = _load_pickle_file(mapping_path)
        if isinstance(payload, dict):
            return {str(key): int(value) for key, value in payload.items()}

    inferred = _infer_mapping_from_preprocessed(root)
    return inferred


def _infer_mapping_from_preprocessed(root: Path) -> dict[str, int]:
    preprocessed_dir = root / "preprocessed"
    if not preprocessed_dir.exists():
        return {}

    for path in sorted(preprocessed_dir.glob("*.pkl")):
        payload = _load_pickle_file(path)
        if not isinstance(payload, dict):
            continue
        mapping: dict[str, int] = {}
        for split_name in ("train", "valid", "test", "dev"):
            rows = payload.get(split_name)
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                raw_doc_id = row.get("raw_doc_id")
                doc_id = row.get("doc_id")
                if raw_doc_id is None or doc_id is None:
                    continue
                mapping[str(raw_doc_id)] = int(doc_id)
        if mapping:
            return mapping
    return {}


def _load_pickle_file(path: Path) -> Any:
    with open(path, "rb") as handle:
        try:
            return pickle.load(handle, encoding="latin1")
        except TypeError:
            handle.seek(0)
            return pickle.load(handle)


def _try_parse_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
