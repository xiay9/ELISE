from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from data.schema import FeaturePaths, LoadedSplit


class DatasetAdapter(ABC):
    dataset_name: str = ""
    available_splits: tuple[str, ...] = ()
    split_aliases: dict[str, tuple[str, ...]] = {}

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def canonical_split(self, split: str) -> str:
        split = split.lower()
        if split == "dev":
            split = "valid"
        if split not in self.available_splits:
            raise ValueError(
                f"Unsupported split '{split}' for {self.dataset_name}. "
                f"Available splits: {self.available_splits}"
            )
        return split

    def split_candidates(self, split: str) -> tuple[str, ...]:
        split = self.canonical_split(split)
        return self.split_aliases.get(split, (split,))

    def resolve_feature_paths(self, split: str) -> FeaturePaths:
        cache_dir = self.root / "cache"
        if not cache_dir.exists():
            return FeaturePaths()

        split = self.canonical_split(split)
        audio_path = None
        video_path = None
        for candidate in self.split_candidates(split):
            if audio_path is None:
                path = cache_dir / f"audio_features_{candidate}.pt"
                if path.exists():
                    audio_path = path
            if video_path is None:
                path = cache_dir / f"video_features_{candidate}.pt"
                if path.exists():
                    video_path = path
        return FeaturePaths(audio_pt=audio_path, video_pt=video_path)

    @abstractmethod
    def load_split(self, split: str) -> LoadedSplit:
        raise NotImplementedError
