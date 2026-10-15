from __future__ import annotations

import re
from pathlib import Path

from data.adapters.base import DatasetAdapter
from data.schema import Dialogue, LoadedSplit, Utterance


PAIR_PATTERN = re.compile(r"\((\d+),(\d+)\)")
MELD_KEY_PATTERN = re.compile(r"^(train|dev|test)_dia(\d+)_utt(\d+)$")
MELD_SPLIT_DIRS = {
    "train": "train_splits_complete",
    "dev": "dev_splits_complete",
    "test": "test_splits_complete",
}


class ECFAdapter(DatasetAdapter):
    dataset_name = "ECF"
    available_splits = ("train", "valid", "test")
    split_aliases = {"valid": ("dev", "valid")}

    def split_file(self, split: str) -> Path:
        split = self.canonical_split(split)
        file_map = {"train": "train.txt", "valid": "valid.txt", "test": "test.txt"}
        return self.root / "dataset" / file_map[split]

    def parse_pairs(self, raw_pairs: str) -> list[tuple[int, int]]:
        return [(int(src), int(dst)) for src, dst in PAIR_PATTERN.findall(raw_pairs)]

    def ecf_meld_mapping_file(self) -> Path:
        return self.root / "dataset" / "all_data_pair_ECFvsMELD.txt"

    def load_ecf_meld_mapping(self) -> dict[tuple[str, int], dict[str, object]]:
        path = self.ecf_meld_mapping_file()
        if not path.exists():
            return {}

        lines = path.read_text(encoding="utf-8").splitlines()
        mapping: dict[tuple[str, int], dict[str, object]] = {}
        idx = 0
        while idx < len(lines):
            header = lines[idx].strip()
            if not header:
                idx += 1
                continue

            parts = header.split()
            if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
                idx += 1
                continue

            dialogue_id, num_turns_text = parts
            num_turns = int(num_turns_text)
            idx += 2

            for _ in range(num_turns):
                if idx >= len(lines):
                    break
                row = lines[idx]
                idx += 1
                pieces = row.split(" | ", 4)
                if len(pieces) != 5:
                    continue
                turn_text, speaker, emotion, text, meld_key = pieces
                if not turn_text.isdigit():
                    continue
                match = MELD_KEY_PATTERN.match(meld_key.strip())
                payload: dict[str, object] = {
                    "speaker": speaker,
                    "emotion": emotion,
                    "text": text,
                    "meld_key": meld_key.strip(),
                }
                if match:
                    meld_split, meld_dialogue_id, meld_utterance_id = match.groups()
                    payload.update(
                        {
                            "meld_split": meld_split,
                            "meld_dialogue_id": int(meld_dialogue_id),
                            "meld_utterance_id": int(meld_utterance_id),
                        }
                    )
                mapping[(dialogue_id, int(turn_text))] = payload
        return mapping

    def load_split(self, split: str) -> LoadedSplit:
        split = self.canonical_split(split)
        path = self.split_file(split)
        lines = path.read_text(encoding="utf-8").splitlines()
        ecf_meld_mapping = self.load_ecf_meld_mapping()

        dialogues: list[Dialogue] = []
        idx = 0
        while idx < len(lines):
            header = lines[idx].strip()
            if not header:
                idx += 1
                continue

            dialogue_id, num_turns = header.split()
            num_turns = int(num_turns)
            pairs = self.parse_pairs(lines[idx + 1].strip())
            utterances: list[Utterance] = []

            for offset in range(num_turns):
                row = lines[idx + 2 + offset]
                turn, speaker, emotion, text, timecode = row.split(" | ", 4)
                metadata = {}
                mapped = ecf_meld_mapping.get((dialogue_id, int(turn)))
                if mapped:
                    metadata["ecf_meld_mapping"] = mapped
                    if "meld_split" in mapped:
                        meld_split = str(mapped["meld_split"])
                        meld_dialogue_id = int(mapped["meld_dialogue_id"])
                        meld_utterance_id = int(mapped["meld_utterance_id"])
                        video_path = (
                            self.root
                            / "dataset"
                            / "MELD.Raw"
                            / MELD_SPLIT_DIRS[meld_split]
                            / f"dia{meld_dialogue_id}_utt{meld_utterance_id}.mp4"
                        )
                        metadata["meld_video_path"] = str(video_path.resolve())
                        metadata["video_path"] = str(video_path.resolve()) if video_path.exists() else None
                        metadata["video_available"] = video_path.exists()
                    else:
                        metadata["video_available"] = False
                utterances.append(
                    Utterance(
                        turn=int(turn),
                        speaker=speaker,
                        text=text,
                        emotion=emotion,
                        timecode=timecode,
                        utterance_name=f"dia{dialogue_id}utt{int(turn)}",
                        metadata={key: value for key, value in metadata.items() if value is not None},
                    )
                )

            dialogues.append(
                Dialogue(
                    dataset=self.dataset_name,
                    split=split,
                    dialogue_id=str(dialogue_id),
                    utterances=utterances,
                    emotion_cause_pairs=pairs,
                    metadata={"source_file": str(path)},
                )
            )
            idx += 2 + num_turns

        return LoadedSplit(
            dataset=self.dataset_name,
            split=split,
            root=self.root,
            dialogues=dialogues,
            feature_paths=self.resolve_feature_paths(split),
            metadata={"source_file": str(path)},
        )
