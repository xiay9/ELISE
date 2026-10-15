from __future__ import annotations

from pathlib import Path

from configs.paths import DATA_ROOT
from data.adapters import ECFAdapter, MECADAdapter
from data.doc_ids import attach_doc_ids
from data.schema import LoadedSplit


DEFAULT_DATA_ROOT = DATA_ROOT

ADAPTER_REGISTRY = {
    "ecf": ECFAdapter,
    "mecad": MECADAdapter,
}


def available_datasets() -> tuple[str, ...]:
    return tuple(dict.fromkeys(adapter.dataset_name for adapter in ADAPTER_REGISTRY.values()))


def get_adapter(dataset: str, data_root: str | Path = DEFAULT_DATA_ROOT):
    key = dataset.lower()
    if key not in ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset}'. "
            f"Available datasets: {available_datasets()}"
        )

    adapter_cls = ADAPTER_REGISTRY[key]
    dataset_root = Path(data_root) / adapter_cls.dataset_name
    return adapter_cls(dataset_root)


def filter_future_pairs(loaded_split: LoadedSplit) -> LoadedSplit:
    removed_total = 0
    kept_total = 0
    for dialogue in loaded_split.dialogues:
        original_pairs = list(dialogue.emotion_cause_pairs)
        kept_pairs = [
            (int(emotion_turn), int(cause_turn))
            for emotion_turn, cause_turn in original_pairs
            if int(cause_turn) <= int(emotion_turn)
        ]
        removed_pairs = [
            (int(emotion_turn), int(cause_turn))
            for emotion_turn, cause_turn in original_pairs
            if int(cause_turn) > int(emotion_turn)
        ]
        dialogue.emotion_cause_pairs = kept_pairs
        if removed_pairs:
            dialogue.metadata["future_cause_pairs_removed"] = removed_pairs
        removed_total += len(removed_pairs)
        kept_total += len(kept_pairs)

    loaded_split.metadata["future_cause_filter"] = {
        "rule": "keep cause_turn <= emotion_turn",
        "removed_pairs": removed_total,
        "kept_pairs": kept_total,
    }
    return loaded_split


def normalize_candidate_scope(value: str | None) -> str:
    key = str(value or "lower_triangle").strip().lower().replace("-", "_")
    aliases = {
        "lower": "lower_triangle",
        "lower_triangle": "lower_triangle",
        "past": "lower_triangle",
        "full": "full_matrix",
        "full_matrix": "full_matrix",
        "all_turn_pairs": "full_matrix",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported candidate_scope={value!r}")
    return aliases[key]


def apply_candidate_scope(loaded_split: LoadedSplit, candidate_scope: str) -> LoadedSplit:
    scope = normalize_candidate_scope(candidate_scope)
    if scope == "lower_triangle":
        return filter_future_pairs(loaded_split)

    future_pairs = 0
    total_pairs = 0
    for dialogue in loaded_split.dialogues:
        pairs = [
            (int(emotion_turn), int(cause_turn))
            for emotion_turn, cause_turn in dialogue.emotion_cause_pairs
        ]
        dialogue.emotion_cause_pairs = pairs
        future_pairs += sum(cause_turn > emotion_turn for emotion_turn, cause_turn in pairs)
        total_pairs += len(pairs)
    loaded_split.metadata["future_cause_filter"] = {
        "rule": "keep all ordered source pairs",
        "removed_pairs": 0,
        "kept_pairs": total_pairs,
        "future_pairs_kept": future_pairs,
    }
    return loaded_split


def load_split(
    dataset: str,
    split: str,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    candidate_scope: str = "lower_triangle",
) -> LoadedSplit:
    adapter = get_adapter(dataset=dataset, data_root=data_root)
    return apply_candidate_scope(
        attach_doc_ids(adapter.load_split(split)),
        candidate_scope,
    )


def load_dataset(
    dataset: str,
    splits: tuple[str, ...] | list[str] | None = None,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    candidate_scope: str = "lower_triangle",
) -> dict[str, LoadedSplit]:
    adapter = get_adapter(dataset=dataset, data_root=data_root)
    target_splits = tuple(splits) if splits is not None else adapter.available_splits
    return {
        split: load_split(
            dataset=dataset,
            split=split,
            data_root=data_root,
            candidate_scope=candidate_scope,
        )
        for split in target_splits
    }
