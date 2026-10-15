"""Unified data access for PaceVer datasets."""

from .doc_ids import attach_doc_ids, load_raw_to_doc_id_map, resolve_dialogue_doc_id
from .registry import available_datasets, filter_future_pairs, get_adapter, load_dataset, load_split
from .schema import Dialogue, FeaturePaths, LoadedSplit, Utterance

__all__ = [
    "attach_doc_ids",
    "Dialogue",
    "FeaturePaths",
    "LoadedSplit",
    "Utterance",
    "available_datasets",
    "filter_future_pairs",
    "get_adapter",
    "load_raw_to_doc_id_map",
    "load_dataset",
    "load_split",
    "resolve_dialogue_doc_id",
]
