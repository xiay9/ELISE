from __future__ import annotations

import json
from pathlib import Path

from data.adapters.base import DatasetAdapter
from data.schema import Dialogue, LoadedSplit, Utterance


class MECADAdapter(DatasetAdapter):
    dataset_name = "MECAD"
    available_splits = ("train", "valid", "test")
    split_aliases = {"valid": ("valid", "dev")}

    def split_file(self, split: str) -> Path:
        split = self.canonical_split(split)
        file_map = {
            "train": "train_data_pair.json",
            "valid": "valid_data_pair.json",
            "test": "test_data_pair.json",
        }
        return self.root / file_map[split]

    def load_split(self, split: str) -> LoadedSplit:
        split = self.canonical_split(split)
        path = self.split_file(split)
        rows = json.loads(path.read_text(encoding="utf-8"))
        utterance_audio_dir = self.root / "utterance_audio"
        utterance_video_dir = self.root / "utterance_video"

        dialogues: list[Dialogue] = []
        for dialogue_id, dialogue_rows in rows.items():
            utterance_rows = dialogue_rows[0]
            utterances: list[Utterance] = []
            pairs: list[tuple[int, int]] = []

            for row in utterance_rows:
                turn = int(row["turn"])
                causes = row.get("expanded emotion cause evidence", [])
                pairs.extend((turn, int(cause_turn)) for cause_turn in causes)

                metadata = {
                    key: value
                    for key, value in row.items()
                    if key
                    not in {
                        "turn",
                        "speaker",
                        "utterance",
                        "emotion",
                        "utterance_name",
                        "video",
                    }
                }
                utterance_name = row.get("utterance_name")
                if utterance_name:
                    audio_path = utterance_audio_dir / f"{utterance_name}.wav"
                    video_path = utterance_video_dir / f"{utterance_name}.mp4"
                    if audio_path.exists():
                        metadata["audio_path"] = str(audio_path.resolve())
                    if video_path.exists():
                        metadata["video_path"] = str(video_path.resolve())
                utterances.append(
                    Utterance(
                        turn=turn,
                        speaker=row["speaker"],
                        text=row["utterance"],
                        emotion=row.get("emotion"),
                        utterance_name=utterance_name,
                        metadata={
                            **metadata,
                            "video_file": row.get("video"),
                        },
                    )
                )

            dialogues.append(
                Dialogue(
                    dataset=self.dataset_name,
                    split=split,
                    dialogue_id=dialogue_id,
                    utterances=utterances,
                    emotion_cause_pairs=pairs,
                    metadata={"source_file": str(path)},
                )
            )

        feature_paths = self.resolve_feature_paths(split)
        extras = {}
        if utterance_audio_dir.exists():
            extras["utterance_audio_dir"] = utterance_audio_dir
        if utterance_video_dir.exists():
            extras["utterance_video_dir"] = utterance_video_dir
        feature_paths.extras.update(extras)

        return LoadedSplit(
            dataset=self.dataset_name,
            split=split,
            root=self.root,
            dialogues=dialogues,
            feature_paths=feature_paths,
            metadata={"source_file": str(path)},
        )
