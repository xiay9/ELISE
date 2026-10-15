from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PAIR_ROLE_IGNORE_INDEX = -100

PAIR_ROLE_CLASS_NAMES = ["emo_cause", "emo_context", "non_pair"]
PAIR_ROLE_SOURCE_BINARY_CLASS_NAMES = ["pair", "non_pair"]
PAIR_ROLE_CAUSE_CLASS_ID = 0
PAIR_ROLE_CONTEXT_CLASS_ID = 1
PAIR_ROLE_NON_PAIR_CLASS_ID = 2

CONTEXT_SUBTYPE_CLASS_NAMES = [
    "background",
    "discourse_bridge",
    "implicit_or_extra_context",
    "alternative_plausible_cause",
]

_TASK_ALIASES = {
    "none": "none",
    "off": "none",
    "false": "none",
    "0": "none",
    "true": "3class",
    "1": "3class",
    "3": "3class",
    "3class": "3class",
    "three": "3class",
    "three_class": "3class",
    "pair_role": "3class",
    "pair_roles": "3class",
    "cause_context_none": "3class",
    "cause-context-none": "3class",
    "context": "3class",
    "source_binary": "source_binary",
    "source-binary": "source_binary",
    "orig_binary": "source_binary",
    "orig-binary": "source_binary",
    "original_binary": "source_binary",
    "original-binary": "source_binary",
    "binary_source": "source_binary",
    "binary-source": "source_binary",
}

_SUBTYPE_ALIASES = {
    "none": "none",
    "off": "none",
    "false": "none",
    "0": "none",
    "true": "4tag",
    "1": "4tag",
    "4": "4tag",
    "4tag": "4tag",
    "4tags": "4tag",
    "context": "4tag",
    "context_subtype": "4tag",
    "context_subtypes": "4tag",
    "subtype": "4tag",
    "subtypes": "4tag",
    # Backward-compatible aliases from the old temporary implementation.
    "cause_role": "4tag",
    "cause4": "4tag",
    "cause5": "4tag",
    "5": "4tag",
    "5tag": "4tag",
    "5tags": "4tag",
    "5class": "4tag",
    "taxonomy": "4tag",
}

_VERIFIED = {"verified", "verified_cause", "accept", "accepted", "positive", "cause", "emo_cause"}
_REFUTED = {"refuted", "refuted_cause", "clearly_refuted", "clearly_non_causal", "reject", "rejected", "non_pair"}
_BACKGROUND = {"background_context", "background", "context", "contextual", "emo_context"}
_UNDER = {
    "under_evidenced",
    "under-evidenced",
    "under_evidenced_cause",
    "unresolved",
    "uncertain",
    "insufficient_evidence",
}

_DISCOURSE_BRIDGE = {"discourse_bridge", "bridge", "transition"}
_ALTERNATIVE = {"alternative_plausible_cause", "alternative_cause", "alternative"}
_IMPLICIT_OR_EXTRA = {
    "implicit_commonsense",
    "implicit",
    "commonsense",
    "multimodal_needed",
    "missing_context",
    "missing_action",
    "lexical_anchor",
    "under_evidenced",
    "under-evidenced",
    "speaker_state",
    "long_range_dependency",
}
_BACKGROUND_SUBTYPES = {
    "background",
    "background_context",
    "speaker_state",
    "long_range_dependency",
    "topic_framing",
}
_CONTEXT_SUBTYPE_NAME_TO_ID = {name: index for index, name in enumerate(CONTEXT_SUBTYPE_CLASS_NAMES)}


def normalize_pair_role_task(task: Any) -> str:
    key = str(task or "none").strip().lower().replace("_", "-")
    key = key.replace("-", "_")
    if key not in _TASK_ALIASES:
        raise ValueError(f"Unsupported pair_role_task={task!r}. Use none, 3class, or source_binary.")
    return _TASK_ALIASES[key]


def pair_role_class_names(task: Any) -> list[str]:
    task = normalize_pair_role_task(task)
    if task == "3class":
        return list(PAIR_ROLE_CLASS_NAMES)
    if task == "source_binary":
        return list(PAIR_ROLE_SOURCE_BINARY_CLASS_NAMES)
    return []


def pair_role_confusion_rows(golds, preds, class_names: list[str] | None = None) -> list[dict[str, Any]]:
    names = list(class_names or PAIR_ROLE_CLASS_NAMES)
    counts = Counter((int(gold), int(pred)) for gold, pred in zip(golds, preds))
    rows = []
    for gold_idx, gold_name in enumerate(names):
        row: dict[str, Any] = {"gold": gold_name}
        for pred_idx, pred_name in enumerate(names):
            row[f"pred_{pred_name}"] = int(counts[(gold_idx, pred_idx)])
        rows.append(row)
    return rows


def write_pair_role_predictions_csv(
    path: str | Path,
    records,
    class_names: list[str] | None = None,
    *,
    run_name: str = "",
    split: str = "test",
) -> None:
    names = list(class_names or PAIR_ROLE_CLASS_NAMES)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_name",
                "split",
                "dialogue_id",
                "target_turn",
                "candidate_turn",
                "gold_id",
                "pred_id",
                "gold",
                "pred",
            ],
        )
        writer.writeheader()
        for dialogue_id, emotion_turn, cause_turn, gold_label, pred_label in records:
            gold_id = int(gold_label)
            pred_id = int(pred_label)
            writer.writerow(
                {
                    "run_name": str(run_name),
                    "split": str(split),
                    "dialogue_id": str(dialogue_id),
                    "target_turn": int(emotion_turn),
                    "candidate_turn": int(cause_turn),
                    "gold_id": gold_id,
                    "pred_id": pred_id,
                    "gold": names[gold_id] if 0 <= gold_id < len(names) else str(gold_id),
                    "pred": names[pred_id] if 0 <= pred_id < len(names) else str(pred_id),
                }
            )


def write_pair_role_predictions_with_scores_csv(
    path: str | Path,
    records,
    class_names: list[str] | None = None,
    *,
    run_name: str = "",
    split: str = "test",
) -> None:
    names = list(class_names or PAIR_ROLE_CLASS_NAMES)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    score_fields = []
    for prefix in ("logit", "prob"):
        score_fields.extend(f"{prefix}_{name}" for name in names)
    fieldnames = [
        "run_name",
        "split",
        "dialogue_id",
        "target_turn",
        "candidate_turn",
        "gold_id",
        "pred_id",
        "gold",
        "pred",
        *score_fields,
        "max_prob",
        "gold_prob",
        "context_prob",
        "margin_top2",
        "scored_by_model",
    ]
    context_idx = names.index("emo_context") if "emo_context" in names else None
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            if isinstance(record, dict):
                dialogue_id = record["dialogue_id"]
                emotion_turn = record["emotion_turn"]
                cause_turn = record["cause_turn"]
                gold_label = record["gold_label"]
                pred_label = record["pred_label"]
                logits = record.get("logits", [])
                probs = record.get("probs", [])
                scored_by_model = record.get("scored_by_model", 1)
            else:
                dialogue_id, emotion_turn, cause_turn, gold_label, pred_label, logits, probs, scored_by_model = record
            gold_id = int(gold_label)
            pred_id = int(pred_label)
            probs = [float(x) for x in probs]
            logits = [float(x) for x in logits] if logits is not None else [float("nan")] * len(names)
            row = {
                "run_name": str(run_name),
                "split": str(split),
                "dialogue_id": str(dialogue_id),
                "target_turn": int(emotion_turn),
                "candidate_turn": int(cause_turn),
                "gold_id": gold_id,
                "pred_id": pred_id,
                "gold": names[gold_id] if 0 <= gold_id < len(names) else str(gold_id),
                "pred": names[pred_id] if 0 <= pred_id < len(names) else str(pred_id),
                "max_prob": max(probs) if probs else "",
                "gold_prob": probs[gold_id] if 0 <= gold_id < len(probs) else "",
                "context_prob": probs[context_idx] if context_idx is not None and context_idx < len(probs) else "",
                "margin_top2": "",
                "scored_by_model": int(scored_by_model),
            }
            if len(probs) >= 2:
                top2 = sorted(probs, reverse=True)[:2]
                row["margin_top2"] = top2[0] - top2[1]
            for idx, name in enumerate(names):
                row[f"logit_{name}"] = logits[idx] if idx < len(logits) else ""
                row[f"prob_{name}"] = probs[idx] if idx < len(probs) else ""
            writer.writerow(row)


def map_pair_role_label_for_task(label_id: int, task: Any) -> int:
    task = normalize_pair_role_task(task)
    label_id = int(label_id)
    if task == "source_binary":
        return PAIR_ROLE_CAUSE_CLASS_ID if label_id == PAIR_ROLE_CAUSE_CLASS_ID else 1
    return label_id


def normalize_context_subtype_task(task: Any) -> str:
    key = str(task or "none").strip().lower().replace("_", "-")
    key = key.replace("-", "_")
    if key not in _SUBTYPE_ALIASES:
        raise ValueError(f"Unsupported context_subtype_task={task!r}. Use none or 4tag.")
    return _SUBTYPE_ALIASES[key]


def context_subtype_class_names(task: Any) -> list[str]:
    task = normalize_context_subtype_task(task)
    if task == "4tag":
        return list(CONTEXT_SUBTYPE_CLASS_NAMES)
    return []


def map_taxonomy_label_to_pair_role(label: Any) -> int | None:
    normalized = str(label or "").strip().lower().replace("-", "_")
    if not normalized:
        return None
    if normalized in _VERIFIED:
        return PAIR_ROLE_CAUSE_CLASS_ID
    if normalized in _BACKGROUND or normalized in _UNDER:
        return PAIR_ROLE_CONTEXT_CLASS_ID
    if normalized in _REFUTED:
        return PAIR_ROLE_NON_PAIR_CLASS_ID
    return None


def _split_subtypes(raw: Any) -> set[str]:
    text = str(raw or "").strip().lower().replace("-", "_")
    if not text:
        return set()
    parts = []
    for chunk in text.replace(",", ";").split(";"):
        chunk = chunk.strip()
        if chunk:
            parts.append(chunk)
    return set(parts)


def map_context_subtypes(row: dict[str, str], pair_role: int) -> list[int]:
    if pair_role != PAIR_ROLE_CONTEXT_CLASS_ID:
        return []

    label = str(row.get("taxonomy_label", "")).strip().lower().replace("-", "_")
    subtypes = _split_subtypes(row.get("taxonomy_subtypes", ""))
    result = set()

    if label in _BACKGROUND or subtypes & _BACKGROUND_SUBTYPES:
        result.add(0)
    if subtypes & _DISCOURSE_BRIDGE:
        result.add(1)
    if label in _UNDER or subtypes & _IMPLICIT_OR_EXTRA:
        result.add(2)
    if subtypes & _ALTERNATIVE:
        result.add(3)

    if not result:
        result.add(0 if label in _BACKGROUND else 2)
    return sorted(result)


def normalize_context_subtype_ids(raw: Any) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, str):
        values = [part.strip() for part in raw.replace(",", ";").split(";") if part.strip()]
    elif isinstance(raw, (list, tuple, set)):
        values = list(raw)
    else:
        values = [raw]

    result = set()
    for value in values:
        if isinstance(value, int):
            subtype_id = value
        else:
            text = str(value).strip().lower().replace("-", "_")
            if text == "":
                continue
            if text.isdigit():
                subtype_id = int(text)
            else:
                subtype_id = _CONTEXT_SUBTYPE_NAME_TO_ID.get(text, -1)
        if 0 <= int(subtype_id) < len(CONTEXT_SUBTYPE_CLASS_NAMES):
            result.add(int(subtype_id))
    return sorted(result)


def _first_present(row: dict[str, str], names: tuple[str, ...]) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return ""


def _split_paths(raw_paths: Any) -> list[str]:
    if raw_paths is None:
        return []
    if isinstance(raw_paths, (list, tuple)):
        return [str(path) for path in raw_paths if str(path).strip()]
    text = str(raw_paths).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(os.pathsep) if part.strip()]


def resolve_pair_role_label_paths(config) -> list[Path]:
    raw_paths = (
        config.get("pair_role_labels_path")
        or config.get("pair_role_label_path")
        or config.get("verifiability_labels_path")
        or config.get("verifiability_label_path")
    )
    paths = []
    for raw_path in _split_paths(raw_paths):
        rendered = raw_path.format(
            data_root=str(config.get("data_root", "")),
            dataset_name=str(config.get("dataset_name", "")),
            dataset_slug=str(config.get("dataset_name", "")).lower(),
        )
        paths.append(Path(rendered).expanduser().resolve())
    return paths


def pair_role_cache_tag(config) -> str:
    task = normalize_pair_role_task(config.get("pair_role_task", config.get("verifiability_task", "none")))
    if task == "none":
        return "none"
    records = []
    for path in resolve_pair_role_label_paths(config):
        records.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "mtime_ns": int(path.stat().st_mtime_ns) if path.exists() else None,
                "size": int(path.stat().st_size) if path.exists() else None,
            }
        )
    payload = repr({"task": task, "paths": records}).encode("utf-8")
    return hashlib.md5(payload).hexdigest()[:8]


def context_subtype_cache_tag(config) -> str:
    task = normalize_context_subtype_task(
        config.get("context_subtype_task", config.get("cause_role_task", "none"))
    )
    if task == "none":
        return "none"
    label_config = dict(config)
    label_config["pair_role_task"] = "3class"
    return pair_role_cache_tag(label_config)


def _row_dataset_matches(row: dict[str, str], dataset_name: str) -> bool:
    row_dataset = _first_present(row, ("dataset", "dataset_name", "corpus"))
    if not row_dataset:
        return True
    return row_dataset.lower() == str(dataset_name).lower()


def _insert_label(
    index: dict[str, dict[tuple[str, int, int], dict[str, Any]]],
    seen_sources: dict[tuple[str, str, int, int], str],
    *,
    split: str,
    dialogue_id: str,
    target_turn: int,
    cause_turn: int,
    label_id: int,
    label_text: str,
    context_subtypes: list[int] | None,
    source: str,
) -> None:
    key = (str(dialogue_id), int(target_turn), int(cause_turn))
    source_key = (str(split), str(dialogue_id), int(target_turn), int(cause_turn))
    previous = index[split].get(key)
    if previous is not None and int(previous["label"]) != int(label_id):
        raise ValueError(
            "Conflicting pair-role labels for "
            f"{source_key}: {previous['taxonomy_label']!r} from {seen_sources[source_key]} "
            f"vs {label_text!r} from {source}"
        )
    index[split][key] = {
        "label": int(label_id),
        "label_name": PAIR_ROLE_CLASS_NAMES[int(label_id)],
        "taxonomy_label": str(label_text),
        "context_subtypes": list(context_subtypes or []),
        "source_path": str(source),
    }
    seen_sources[source_key] = str(source)


def _load_pair_role_json(
    path: Path,
    *,
    dataset_name: str,
    index: dict[str, dict[tuple[str, int, int], dict[str, Any]]],
    seen_sources: dict[tuple[str, str, int, int], str],
) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Pair-role JSON must be a mapping: {path}")

    payload_dataset = str(payload.get("dataset", "")).strip()
    if payload_dataset and payload_dataset.lower() != str(dataset_name).lower():
        return

    splits = payload.get("splits", {})
    if not isinstance(splits, dict):
        raise ValueError(f"Pair-role JSON field 'splits' must be a mapping: {path}")

    for split, dialogues in splits.items():
        split = str(split).lower()
        if split == "dev":
            split = "valid"
        if not isinstance(dialogues, list):
            raise ValueError(f"Pair-role JSON split must be a list: {path}:{split}")
        for dialogue in dialogues:
            if not isinstance(dialogue, dict):
                continue
            dialogue_id = _first_present(dialogue, ("dialogue_id", "raw_doc_id", "doc_id"))
            if not dialogue_id:
                continue
            pairs = dialogue.get("pairs", [])
            if not isinstance(pairs, list):
                raise ValueError(f"Pair-role JSON dialogue pairs must be a list: {path}:{split}:{dialogue_id}")
            for pair in pairs:
                if not isinstance(pair, dict):
                    continue
                target_text = _first_present(pair, ("target_emotion_turn", "emotion_turn", "target_turn"))
                cause_text = _first_present(pair, ("cause_turn", "candidate_cause_turn"))
                label_text = _first_present(pair, ("pair_role_label", "label_name", "label", "taxonomy_label"))
                if not target_text or not cause_text:
                    continue
                try:
                    target_turn = int(float(target_text))
                    cause_turn = int(float(cause_text))
                except ValueError as exc:
                    raise ValueError(f"Invalid turn id in {path}:{split}:{dialogue_id}") from exc

                label_id = pair.get("label_id")
                if label_id is None:
                    label_id = map_taxonomy_label_to_pair_role(label_text)
                else:
                    label_id = int(label_id)
                if label_id is None or not (0 <= int(label_id) < len(PAIR_ROLE_CLASS_NAMES)):
                    continue

                subtype_ids = normalize_context_subtype_ids(
                    pair.get("context_subtype_ids", pair.get("context_subtypes"))
                )
                _insert_label(
                    index,
                    seen_sources,
                    split=split,
                    dialogue_id=str(dialogue_id),
                    target_turn=target_turn,
                    cause_turn=cause_turn,
                    label_id=int(label_id),
                    label_text=str(label_text or PAIR_ROLE_CLASS_NAMES[int(label_id)]),
                    context_subtypes=subtype_ids,
                    source=str(path),
                )


def _load_pair_role_csv(
    path: Path,
    *,
    dataset_name: str,
    index: dict[str, dict[tuple[str, int, int], dict[str, Any]]],
    seen_sources: dict[tuple[str, str, int, int], str],
) -> None:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for line_number, row in enumerate(reader, start=2):
            if not _row_dataset_matches(row, dataset_name):
                continue
            split = _first_present(row, ("split", "data_split")).lower()
            if split == "dev":
                split = "valid"
            dialogue_id = _first_present(row, ("dialogue_id", "raw_doc_id", "doc_id"))
            target_text = _first_present(row, ("target_emotion_turn", "emotion_turn", "target_turn"))
            cause_text = _first_present(row, ("cause_turn", "candidate_cause_turn"))
            label_text = _first_present(
                row,
                (
                    "pair_role_label",
                    "taxonomy_label",
                    "llm_verifiability_label",
                    "human_verifiability_label",
                    "verifiability_label",
                    "label",
                ),
            )
            if not split or not dialogue_id or not target_text or not cause_text:
                continue
            label_id = map_taxonomy_label_to_pair_role(label_text)
            if label_id is None:
                continue
            try:
                target_turn = int(float(target_text))
                cause_turn = int(float(cause_text))
            except ValueError as exc:
                raise ValueError(f"Invalid turn id in {path}:{line_number}") from exc

            _insert_label(
                index,
                seen_sources,
                split=split,
                dialogue_id=str(dialogue_id),
                target_turn=target_turn,
                cause_turn=cause_turn,
                label_id=int(label_id),
                label_text=str(label_text),
                context_subtypes=map_context_subtypes(row, int(label_id)),
                source=f"{path}:{line_number}",
            )


def load_pair_role_label_index(config) -> dict[str, dict[tuple[str, int, int], dict[str, Any]]]:
    task = normalize_pair_role_task(config.get("pair_role_task", config.get("verifiability_task", "none")))
    if task == "none":
        return {}

    paths = resolve_pair_role_label_paths(config)
    if not paths:
        raise ValueError("pair_role_task is enabled, but pair_role_labels_path is empty.")

    dataset_name = str(config.get("dataset_name", ""))
    index: dict[str, dict[tuple[str, int, int], dict[str, Any]]] = defaultdict(dict)
    seen_sources: dict[tuple[str, str, int, int], str] = {}

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Pair-role label file does not exist: {path}")
        if path.suffix.lower() == ".json":
            _load_pair_role_json(path, dataset_name=dataset_name, index=index, seen_sources=seen_sources)
        else:
            _load_pair_role_csv(path, dataset_name=dataset_name, index=index, seen_sources=seen_sources)
    return {split: dict(values) for split, values in index.items()}


def dialogue_key_candidates(dialogue) -> list[str]:
    candidates = [
        str(dialogue.dialogue_id),
        str(dialogue.metadata.get("raw_doc_id", "")),
        str(dialogue.metadata.get("doc_id", "")),
    ]
    result = []
    seen = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            result.append(candidate)
    return result


def labels_for_dialogue(
    label_index: dict[str, dict[tuple[str, int, int], dict[str, Any]]],
    *,
    split: str,
    dialogue,
) -> list[dict[str, Any]]:
    split = split.lower()
    split_index = label_index.get(split, {})
    if not split_index:
        return []

    turns = {int(utterance.turn) for utterance in dialogue.utterances}
    labels = []
    for dialogue_key in dialogue_key_candidates(dialogue):
        for target_turn in turns:
            for cause_turn in turns:
                payload = split_index.get((dialogue_key, target_turn, cause_turn))
                if payload is None:
                    continue
                labels.append(
                    {
                        "emotion_turn": int(target_turn),
                        "cause_turn": int(cause_turn),
                        "label": int(payload["label"]),
                        "label_name": str(payload["label_name"]),
                        "taxonomy_label": str(payload["taxonomy_label"]),
                        "context_subtypes": list(payload.get("context_subtypes", [])),
                    }
                )
    dedup = {}
    for item in labels:
        key = (item["emotion_turn"], item["cause_turn"])
        if key in dedup and dedup[key]["label"] != item["label"]:
            raise ValueError(f"Conflicting labels for dialogue={dialogue.dialogue_id} pair={key}")
        dedup[key] = item
    return [dedup[key] for key in sorted(dedup)]


def count_pair_role_labels(rows: list[dict[str, Any]]) -> Counter:
    counter = Counter()
    for row in rows:
        for item in row.get("pair_role_labels", []):
            counter[int(item["label"])] += 1
    return counter


def count_pair_role_labels_for_task(
    rows: list[dict[str, Any]],
    task: Any,
    *,
    non_pair_max_distance: int | None = None,
) -> Counter:
    task = normalize_pair_role_task(task)
    counter = Counter()
    for row in rows:
        for item in row.get("pair_role_labels", []):
            label_id = int(item["label"])
            mapped_label = map_pair_role_label_for_task(label_id, task)
            if non_pair_max_distance is not None and label_id == PAIR_ROLE_NON_PAIR_CLASS_ID:
                distance = int(item["emotion_turn"]) - int(item["cause_turn"])
                if distance > int(non_pair_max_distance):
                    continue
            counter[int(mapped_label)] += 1
    return counter


def count_context_subtype_labels(rows: list[dict[str, Any]], num_classes: int) -> tuple[Counter, int]:
    counter = Counter()
    item_count = 0
    for row in rows:
        for item in row.get("pair_role_labels", []):
            if int(item.get("label", -1)) != PAIR_ROLE_CONTEXT_CLASS_ID:
                continue
            subtype_ids = [int(x) for x in item.get("context_subtypes", []) if 0 <= int(x) < int(num_classes)]
            if not subtype_ids:
                continue
            item_count += 1
            for subtype_id in set(subtype_ids):
                counter[int(subtype_id)] += 1
    return counter, item_count


def make_class_weights(counts: Counter, num_classes: int, *, max_weight: float = 10.0) -> list[float]:
    if num_classes <= 0:
        return []
    values = [float(counts.get(i, 0)) for i in range(num_classes)]
    positive = [value for value in values if value > 0]
    if not positive:
        return [1.0] * num_classes

    mean = sum(positive) / len(positive)
    weights = []
    for value in values:
        if value <= 0:
            weights.append(1.0)
            continue
        ratio = mean / value
        weight = ratio ** 0.5
        weights.append(float(max(0.0, min(float(max_weight), weight))))
    return weights


def binary_set_metric(gold_items, pred_items, prefix: str) -> dict[str, Any]:
    gold_set = set(gold_items)
    pred_set = set(pred_items)
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        f"{prefix}_precision": float(precision),
        f"{prefix}_recall": float(recall),
        f"{prefix}_f1": float(f1),
        f"{prefix}_support": int(len(gold_set)),
        f"{prefix}_predicted": int(len(pred_set)),
        f"{prefix}_tp": int(tp),
        f"{prefix}_fp": int(fp),
        f"{prefix}_fn": int(fn),
    }


def pair_role_turn_metric_dict(records, *, gold_emotion_turns=None) -> dict[str, Any]:
    gold_emotions = set(gold_emotion_turns or [])
    pred_emotions = set()
    gold_causes = set()
    pred_causes = set()
    gold_contexts = set()
    pred_contexts = set()

    for record in records:
        dialogue_id, emotion_turn, cause_turn, gold_label, pred_label = record
        target_key = (str(dialogue_id), int(emotion_turn))
        candidate_key = (str(dialogue_id), int(cause_turn))
        gold_label = int(gold_label)
        pred_label = int(pred_label)
        if gold_label in {PAIR_ROLE_CAUSE_CLASS_ID, PAIR_ROLE_CONTEXT_CLASS_ID} and gold_emotion_turns is None:
            gold_emotions.add(target_key)
        if pred_label in {PAIR_ROLE_CAUSE_CLASS_ID, PAIR_ROLE_CONTEXT_CLASS_ID}:
            pred_emotions.add(target_key)
        if gold_label == PAIR_ROLE_CAUSE_CLASS_ID:
            gold_causes.add(candidate_key)
        if pred_label == PAIR_ROLE_CAUSE_CLASS_ID:
            pred_causes.add(candidate_key)
        if gold_label == PAIR_ROLE_CONTEXT_CLASS_ID:
            gold_contexts.add(candidate_key)
        if pred_label == PAIR_ROLE_CONTEXT_CLASS_ID:
            pred_contexts.add(candidate_key)

    metrics = {"turn_role_eval_scope": "dialogue_turn"}
    metrics.update(binary_set_metric(gold_emotions, pred_emotions, "turn_emotion"))
    metrics.update(binary_set_metric(gold_causes, pred_causes, "turn_cause"))
    metrics.update(binary_set_metric(gold_contexts, pred_contexts, "turn_context"))
    return metrics


def make_multilabel_pos_weights(
    counts: Counter,
    num_classes: int,
    item_count: int,
    *,
    max_weight: float = 10.0,
) -> list[float]:
    if num_classes <= 0 or item_count <= 0:
        return [1.0] * max(0, int(num_classes))
    weights = []
    for label_id in range(int(num_classes)):
        pos = float(counts.get(label_id, 0))
        neg = max(0.0, float(item_count) - pos)
        if pos <= 0:
            weights.append(1.0)
            continue
        weight = (neg / pos) ** 0.5
        weights.append(float(max(0.0, min(float(max_weight), weight))))
    return weights
