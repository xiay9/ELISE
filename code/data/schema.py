from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Utterance:
    turn: int
    speaker: str
    text: str
    emotion: str | None = None
    timecode: str | None = None
    utterance_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Dialogue:
    dataset: str
    split: str
    dialogue_id: str
    utterances: list[Utterance]
    emotion_cause_pairs: list[tuple[int, int]] = field(default_factory=list)
    raw_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_utterances(self) -> int:
        return len(self.utterances)

    @property
    def num_pairs(self) -> int:
        return len(self.emotion_cause_pairs)


@dataclass(slots=True)
class FeaturePaths:
    audio_pt: Path | None = None
    video_pt: Path | None = None
    extras: dict[str, Path] = field(default_factory=dict)


@dataclass(slots=True)
class LoadedSplit:
    dataset: str
    split: str
    root: Path
    dialogues: list[Dialogue]
    feature_paths: FeaturePaths = field(default_factory=FeaturePaths)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_dialogues(self) -> int:
        return len(self.dialogues)

    @property
    def num_utterances(self) -> int:
        return sum(dialogue.num_utterances for dialogue in self.dialogues)

    @property
    def num_pairs(self) -> int:
        return sum(dialogue.num_pairs for dialogue in self.dialogues)
