#!/usr/bin/env python

"""
Name: loader.py
"""

import os
import math
import fcntl
import json
import hashlib
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import transformers
import logging
import pickle as pkl
from pathlib import Path
from typing import Dict
from dataclasses import dataclass
from collections import Counter
import numpy as np
import random

from data import load_split

from .pair_roles import (
    PAIR_ROLE_CAUSE_CLASS_ID,
    PAIR_ROLE_CONTEXT_CLASS_ID,
    PAIR_ROLE_IGNORE_INDEX,
    PAIR_ROLE_NON_PAIR_CLASS_ID,
    context_subtype_cache_tag,
    context_subtype_class_names,
    count_context_subtype_labels,
    count_pair_role_labels,
    count_pair_role_labels_for_task,
    labels_for_dialogue,
    load_pair_role_label_index,
    make_class_weights,
    make_multilabel_pos_weights,
    map_pair_role_label_for_task,
    normalize_context_subtype_task,
    normalize_pair_role_task,
    pair_role_cache_tag,
    pair_role_class_names,
)


class PaceExtDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data, mode):
        super(PaceExtDataset, self).__init__()
        logging.warning("Loading data...")
        self.data = data[mode]
        self.label_dict = data['label_dict']
        self.speaker_dict = data['speaker_dict']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        sample = self.data[i]
        return {
            'doc_id': int(sample['doc_id']),
            'raw_doc_id': str(sample.get('raw_doc_id', sample['doc_id'])),
            'emotion_cause_pairs': list(sample.get('emotion_cause_pairs', [])),
            'pair_role_labels': list(sample.get('pair_role_labels', [])),
            'lines': list(sample.get('lines', [])),
            'speakers': [self.speaker_dict[w] for w in sample.get('speakers', [])],
            'emotions': [self.label_dict[w] for w in sample.get('emotions', [])],
            'timecodes': list(sample.get('timecodes', [])),
        }

@dataclass
class PaceExtCollator:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    config: Dict
    video_map: Dict
    audio_map: Dict
    teacher_annotations: Dict | None = None
    mode: str = "train"
    def __call__(self, instances) -> Dict[str, torch.Tensor]:
        doc_ids = [line['doc_id'] for line in instances]
        pairs = [[(w - 1, z - 1) for w, z in line['emotion_cause_pairs']] for line in instances]
        pair_role_items = [line.get('pair_role_labels', []) for line in instances]
        utterances = [line['lines'] for line in instances]
        emotions = [line['emotions'] for line in instances]
        speakers = [line['speakers'] for line in instances]
        label_dict = self.config['label_dict']

        if self.config.emo_cat != 'yes':
            neutral_id = None
            for candidate in ('neutral', 'neu', 'Neutral'):
                if candidate in label_dict:
                    neutral_id = label_dict[candidate]
                    break
            if neutral_id is None:
                raise KeyError("Could not find a neutral emotion label in label_dict.")
            emotions = [[0 if w == neutral_id else 1 for w in line] for line in emotions]

        utterance_nums = [len(line) for line in utterances]
        IGNORE_INDEX = -100
        max_utterance = max(utterance_nums)

        emotions = [w + [IGNORE_INDEX] * (max_utterance - len(w)) for w in emotions]
        res = []
        max_length = self.config['max_length']
        total_length = self.config['total_length']
        input_tokens, indices = pack(utterances, max_length, total_length, self.tokenizer, self.config)

        seq_lens = [len(w) for w in input_tokens]
        max_seq_len = max(seq_lens)
        pad_token = self.tokenizer.pad_token
        input_tokens = [w + [pad_token] * (max_seq_len - l) for w, l in zip(input_tokens, seq_lens)]

        input_ids = [self.tokenizer.convert_tokens_to_ids(w) for w in input_tokens]
        attention_mask = [[1] * l + [0] * (max_seq_len - l) for l in seq_lens]

        # padding pairs
        pair_nums = [len(line) for line in pairs]
        max_pair = max(pair_nums)
        pairs = [w + [(IGNORE_INDEX, IGNORE_INDEX)] * (max_pair - len(w)) for w in pairs]
        pairs_tensor = torch.tensor(pairs, dtype=torch.long)

        z = pairs_tensor[:, :, 1]
        cause_labels = torch.zeros(len(pairs), max_utterance, dtype=torch.long)
        valid = (z != IGNORE_INDEX) & (z >= 0) & (z < max_utterance)
        if valid.any():
            b_idx = torch.arange(len(pairs)).unsqueeze(1).expand_as(z)[valid]
            z_idx = z[valid]
            cause_labels[b_idx, z_idx] = 1

        teacher_pair_positive_mask = torch.zeros(len(pairs), max_utterance, max_utterance, dtype=torch.float32)
        teacher_pair_negative_mask = torch.zeros(len(pairs), max_utterance, max_utterance, dtype=torch.float32)
        if self.teacher_annotations:
            for batch_index, (doc_id, utterance_num) in enumerate(zip(doc_ids, utterance_nums)):
                doc_annotations = self.teacher_annotations.get(int(doc_id), {})
                if not doc_annotations:
                    continue
                for emotion_turn, annotation in doc_annotations.items():
                    row_index = int(emotion_turn) - 1
                    if row_index < 0 or row_index >= int(utterance_num):
                        continue
                    positive_turns = [
                        int(turn)
                        for turn in annotation.get('distill_positive_turns', [])
                    ]
                    positive_weights = [
                        float(weight)
                        for weight in annotation.get('distill_positive_weights', [1.0] * len(positive_turns))
                    ]
                    negative_turns = [
                        int(turn)
                        for turn in annotation.get('distill_negative_turns', [])
                        if not (
                            bool(getattr(self.config, 'teacher_exclude_self_from_negatives', False))
                            and int(turn) == (row_index + 1)
                        )
                    ]
                    negative_weights = [
                        float(weight)
                        for weight in annotation.get('distill_negative_weights', [1.0] * len(negative_turns))
                    ]
                    if len(positive_turns) != len(positive_weights) or len(negative_turns) != len(negative_weights):
                        continue
                    if not positive_turns or not negative_turns:
                        continue
                    for cause_turn, cause_weight in zip(positive_turns, positive_weights):
                        col_index = int(cause_turn) - 1
                        if col_index < 0 or col_index >= int(utterance_num) or col_index > row_index:
                            continue
                        teacher_pair_positive_mask[batch_index, row_index, col_index] = float(cause_weight)
                    for cause_turn, cause_weight in zip(negative_turns, negative_weights):
                        col_index = int(cause_turn) - 1
                        if col_index < 0 or col_index >= int(utterance_num) or col_index > row_index:
                            continue
                        if int(cause_turn) == (row_index + 1):
                            cause_weight = float(cause_weight) * float(
                                getattr(self.config, 'teacher_self_negative_weight', 1.0)
                            )
                            if cause_weight <= 0.0:
                                continue
                        teacher_pair_negative_mask[batch_index, row_index, col_index] = float(cause_weight)

        pair_role_labels = torch.full(
            (len(pairs), max_utterance, max_utterance),
            PAIR_ROLE_IGNORE_INDEX,
            dtype=torch.long,
        )
        context_subtype_num_classes = int(getattr(self.config, 'context_subtype_num_classes', 0))
        context_subtype_labels = torch.zeros(
            (len(pairs), max_utterance, max_utterance, max(0, context_subtype_num_classes)),
            dtype=torch.float32,
        )
        context_subtype_mask = torch.zeros(len(pairs), max_utterance, max_utterance, dtype=torch.float32)
        if bool(getattr(self.config, 'pair_role_enable', False)):
            for batch_index, (items, utterance_num) in enumerate(zip(pair_role_items, utterance_nums)):
                for item in items:
                    try:
                        row_index = int(item['emotion_turn']) - 1
                        col_index = int(item['cause_turn']) - 1
                        label_id = int(item['label'])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if row_index < 0 or col_index < 0:
                        continue
                    if row_index >= int(utterance_num) or col_index >= int(utterance_num):
                        continue
                    if col_index > row_index:
                        continue
                    pair_role_labels[batch_index, row_index, col_index] = map_pair_role_label_for_task(
                        label_id,
                        getattr(self.config, 'pair_role_task', '3class'),
                    )
            if str(self.mode).lower() == 'train':
                max_non_pair_distance = getattr(self.config, 'pair_role_train_non_pair_max_distance', None)
                if max_non_pair_distance is not None:
                    max_non_pair_distance = int(max_non_pair_distance)
                    row_ids = torch.arange(max_utterance).view(1, max_utterance, 1)
                    col_ids = torch.arange(max_utterance).view(1, 1, max_utterance)
                    too_far = (row_ids - col_ids) > max_non_pair_distance
                    non_pair_id = int(getattr(self.config, 'pair_role_non_pair_class_id', PAIR_ROLE_NON_PAIR_CLASS_ID))
                    pair_role_labels[(pair_role_labels == non_pair_id) & too_far] = PAIR_ROLE_IGNORE_INDEX
            if bool(getattr(self.config, 'pair_role_use_emo_cause_for_cause_labels', True)):
                cause_class_id = int(getattr(self.config, 'pair_role_cause_class_id', PAIR_ROLE_CAUSE_CLASS_ID))
                cause_labels.zero_()
                cause_pairs = pair_role_labels == cause_class_id
                if cause_pairs.any():
                    batch_indices, _, cause_indices = torch.where(cause_pairs)
                    cause_labels[batch_indices, cause_indices] = 1
        if bool(getattr(self.config, 'context_subtype_enable', False)):
            for batch_index, (items, utterance_num) in enumerate(zip(pair_role_items, utterance_nums)):
                for item in items:
                    try:
                        row_index = int(item['emotion_turn']) - 1
                        col_index = int(item['cause_turn']) - 1
                    except (KeyError, TypeError, ValueError):
                        continue
                    if row_index < 0 or col_index < 0:
                        continue
                    if row_index >= int(utterance_num) or col_index >= int(utterance_num):
                        continue
                    if col_index > row_index:
                        continue
                    subtype_ids = item.get('context_subtypes', [])
                    if not subtype_ids:
                        continue
                    context_subtype_mask[batch_index, row_index, col_index] = 1.0
                    for subtype_id in subtype_ids:
                        subtype_id = int(subtype_id)
                        if 0 <= subtype_id < context_subtype_num_classes:
                            context_subtype_labels[batch_index, row_index, col_index, subtype_id] = 1.0

        speakers = [w + [-1] * (max_utterance - len(w)) for w in speakers]
        speaker_ids = torch.tensor(speakers, dtype=torch.long)
        use_video = bool(getattr(self.config, 'use_video', True))
        use_audio = bool(getattr(self.config, 'use_audio', True))

        video_features = []
        for i in range(len(doc_ids)):
            dialogue_video = []
            for j in range(1, utterance_nums[i] + 1):
                key = (doc_ids[i], j)
                if use_video and key in self.video_map:
                    dialogue_video.append(np.asarray(self.video_map[key], dtype=np.float32))
                else:
                    dialogue_video.append(np.zeros(self.config.video_dim, dtype=np.float32))
            video_features.append(np.stack(dialogue_video))
        video_features = np.stack([np.concatenate([w, np.zeros((max_utterance - w.shape[0], self.config.video_dim), dtype=np.float32)], axis=0) for w in video_features])

        audio_features = []
        for i in range(len(doc_ids)):
            dialogue_audio = []
            for j in range(1, utterance_nums[i] + 1):
                key = (doc_ids[i], j)
                if use_audio and key in self.audio_map:
                    dialogue_audio.append(np.asarray(self.audio_map[key], dtype=np.float32))
                else:
                    dialogue_audio.append(np.zeros(self.config.audio_dim, dtype=np.float32))
            audio_features.append(np.stack(dialogue_audio))
        audio_features = np.stack([np.concatenate([w, np.zeros((max_utterance - w.shape[0], self.config.audio_dim), dtype=np.float32)], axis=0) for w in audio_features])

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'input_masks': torch.tensor(attention_mask, dtype=torch.long),
            'indices': indices, # 'indices': [(global_id, start, end), ...]
            'utterance_nums': torch.tensor(utterance_nums, dtype=torch.long),
            'pairs': pairs_tensor,
            'pair_nums': torch.tensor(pair_nums, dtype=torch.long),
            'labels': torch.tensor(emotions, dtype=torch.long),
            'cause_labels': cause_labels,
            'doc_ids': doc_ids,
            'speaker_ids': speaker_ids,
            'video_features': torch.tensor(video_features, dtype=torch.float),
            'audio_features': torch.tensor(audio_features, dtype=torch.float),
            'teacher_pair_positive_mask': teacher_pair_positive_mask,
            'teacher_pair_negative_mask': teacher_pair_negative_mask,
            'pair_role_labels': pair_role_labels,
            'context_subtype_labels': context_subtype_labels,
            'context_subtype_mask': context_subtype_mask,
        }

def _resolve_effective_total_len(tokenizer, requested_total_len):
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 1_000_000:
        return min(int(requested_total_len), tokenizer_limit)
    return int(requested_total_len)


def pack(dialogues, max_len, total_len, tokenizer, config):
    del config
    res = []
    indices = []
    total_len = _resolve_effective_total_len(tokenizer, total_len)
    if total_len < 3:
        raise ValueError(f"total_len must be at least 3, but got {total_len}")

    for i in range(len(dialogues)):
        cur_res = [tokenizer.cls_token]
        cur_indices = []
        for line in dialogues[i]:
            tokens = tokenizer.tokenize(line)
            max_line_len = max(1, min(int(max_len), total_len - 2))
            if len(tokens) > max_line_len:
                tokens = tokens[:max_line_len]

            required_len = len(tokens) + 1  # reserve space for sep token
            if len(cur_res) + required_len > total_len and len(cur_res) > 1:
                res.append(cur_res)
                cur_res = [tokenizer.cls_token]
            global_id = len(res)
            start = len(cur_res)
            cur_res += tokens + [tokenizer.sep_token]
            end = len(cur_res) - 1
            cur_indices.append((global_id, start, end))
        res.append(cur_res)
        indices.append(cur_indices)
    return res, indices

def build_distance_prior(train_data, max_utt=35, smooth=1.0):
    """
    Build relative distance prior Prior(Δ) from training emotion-cause pairs only.

    Δ = (w - z) where w/z are 0-based indices and w >= z.
    """
    max_utt = int(max_utt)
    hist = np.zeros(max_utt, dtype=np.float64)
    for conv in train_data:
        for (w, z) in conv.get('emotion_cause_pairs', []):
            dw = (w - 1) - (z - 1)
            if 0 <= dw < max_utt:
                hist[dw] += 1.0
    hist = hist + float(smooth)
    prior = hist / hist.sum()
    return prior.astype(np.float32).tolist()

def _load_feature_map(path) -> Dict:
    if path is None or not os.path.exists(str(path)):
        return {}
    import torch
    raw = torch.load(str(path), weights_only=False)
    feature_map = {}
    for key, value in raw.items():
        if not isinstance(key, tuple) or len(key) != 2:
            continue
        feature_map[(int(key[0]), int(key[1]))] = np.asarray(value, dtype=np.float32)
    return feature_map


_AUDIO_SUMMARY_KEY_ALIAS = {
    'energy_level': 'en',
    'pitch_level': 'pi',
    'pitch_variability': 'pv',
    'speech_rate': 'sr',
    'audio_reliability': 'ar',
    'pause_before': 'pb',
    'pause_after': 'pa',
    'overlap_or_interruption': 'oi',
}

_VIDEO_SUMMARY_KEY_ALIAS = {
    'facial_affect': 'fa',
    'affect_intensity': 'ai',
    'gesture': 'ge',
    'mouth_activity': 'ma',
}

_SUMMARY_VALUE_ALIAS = {
    'neutral': 'neu',
    'positive_display': 'pos',
    'negative_display': 'neg',
    'mixed': 'mix',
    'unknown': 'unk',
    'low': 'lo',
    'medium': 'med',
    'high': 'hi',
    'none': 'none',
    'small': 'sm',
    'emphatic': 'emph',
    'self_touch': 'self',
    'pointing': 'point',
    'speaking_like': 'spk',
    'not_speaking_like': 'nspk',
    'unclear': 'unc',
    'normal': 'norm',
    'slow': 'slow',
    'fast': 'fast',
    'short': 'sh',
    'long': 'lg',
    'interrupts_other': 'intr',
    'interrupted_by_other': 'intrd',
}

_SUMMARY_CACHE_VERSION = 2


def _format_summary_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        parts = [_format_summary_value(item) for item in value]
        return ",".join(part for part in parts if part)
    text = str(value).strip()
    if not text:
        return ""
    return _SUMMARY_VALUE_ALIAS.get(text, text)


def _format_summary_payload(kind: str, payload) -> str:
    key_alias = _AUDIO_SUMMARY_KEY_ALIAS if kind == 'audio' else _VIDEO_SUMMARY_KEY_ALIAS
    if isinstance(payload, dict):
        segments = []
        for key, value in payload.items():
            if value is None:
                continue
            rendered = _format_summary_value(value)
            if rendered:
                segments.append(f"{key_alias.get(str(key), str(key))}={rendered}")
        return " ".join(segments)
    if isinstance(payload, str):
        return _format_summary_value(payload)
    return ""


def _resolve_summary_path(config, kind: str, split: str) -> Path | None:
    explicit_key = f"{kind}_summary_{split}_path"
    explicit_value = config.get(explicit_key)
    if explicit_value:
        return Path(str(explicit_value)).expanduser().resolve()

    template_key = f"{kind}_summary_path_template"
    template_value = config.get(template_key)
    if not template_value:
        return None

    rendered = str(template_value).format(
        data_root=str(config['data_root']),
        dataset_name=str(config['dataset_name']),
        dataset_slug=str(config['dataset_name']).lower(),
        split=str(split),
    )
    return Path(rendered).expanduser().resolve()


def _load_summary_map(config, kind: str, split: str) -> Dict[tuple[str, int], str]:
    path = _resolve_summary_path(config, kind, split)
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"{kind} summary file does not exist for split={split}: {path}")

    summary_field = f"{kind}_summary"
    summary_map: Dict[tuple[str, int], str] = {}
    with open(path, 'r', encoding='utf-8') as handle:
        for line_index, raw_line in enumerate(handle, start=1):
            text = raw_line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_index}") from exc
            dialogue_id = payload.get('dialogue_id')
            turn_id = payload.get('turn_id')
            if dialogue_id is None or turn_id is None:
                continue
            rendered = _format_summary_payload(kind, payload.get(summary_field))
            if rendered:
                summary_map[(str(dialogue_id), int(turn_id))] = rendered
    return summary_map


def _load_teacher_annotations(annotation_dir, split: str, *, keep_only: bool = False) -> Dict[int, Dict[int, dict]]:
    if annotation_dir is None:
        return {}

    path = Path(str(annotation_dir)).expanduser().resolve() / f"{split}.jsonl"
    if not path.exists():
        if split == 'test':
            return {}
        raise FileNotFoundError(f"Teacher annotation file does not exist: {path}")

    annotations: Dict[int, Dict[int, dict]] = {}
    with open(path, 'r', encoding='utf-8') as handle:
        for line_index, raw_line in enumerate(handle, start=1):
            text = raw_line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if keep_only and not bool(payload.get('distill_keep', False)):
                continue
            doc_id = int(payload['doc_id'])
            emotion_turn = int(payload['emotion_turn'])
            doc_bucket = annotations.setdefault(doc_id, {})
            if emotion_turn in doc_bucket:
                raise ValueError(
                    f"Duplicate teacher annotation for doc_id={doc_id}, emotion_turn={emotion_turn} "
                    f"in {path} at line {line_index}."
                )
            doc_bucket[emotion_turn] = {
                'candidate_turns': [int(turn) for turn in payload.get('candidate_turns', [])],
                'teacher_supported': bool(payload.get('teacher_supported', False)),
                'teacher_cause_turns': [int(turn) for turn in payload.get('teacher_cause_turns', [])],
                'teacher_ranked_turns': [int(turn) for turn in payload.get('teacher_ranked_turns', payload.get('teacher_cause_turns', []))],
                'gold_nonself_turns': [int(turn) for turn in payload.get('gold_nonself_turns', [])],
                'distill_positive_turns': [int(turn) for turn in payload.get('distill_positive_turns', [])],
                'distill_positive_weights': [
                    float(weight)
                    for weight in payload.get(
                        'distill_positive_weights',
                        [1.0] * len(payload.get('distill_positive_turns', [])),
                    )
                ],
                'distill_negative_turns': [int(turn) for turn in payload.get('distill_negative_turns', [])],
                'distill_negative_turns_all': [int(turn) for turn in payload.get('distill_negative_turns_all', payload.get('distill_negative_turns', []))],
                'distill_negative_weights': [
                    float(weight)
                    for weight in payload.get(
                        'distill_negative_weights',
                        [1.0] * len(payload.get('distill_negative_turns', [])),
                    )
                ],
                'distill_keep': bool(payload.get('distill_keep', False)),
                'distill_skip_reasons': [str(reason) for reason in payload.get('distill_skip_reasons', [])],
                'row_type': str(payload.get('row_type', 'unknown')),
                'base_row_f1': float(payload.get('base_row_f1', 0.0)),
                'teacher_row_f1': float(payload.get('teacher_row_f1', 0.0)),
            }
    return annotations


def _build_structured_rows_from_loaded_split(
    loaded_split,
    *,
    config,
    pair_role_label_index: dict | None = None,
    audio_summary_map: Dict[tuple[str, int], str] | None = None,
    video_summary_map: Dict[tuple[str, int], str] | None = None,
) -> list[dict]:
    audio_summary_map = audio_summary_map or {}
    video_summary_map = video_summary_map or {}
    use_audio_summary = bool(config.get('use_audio_summary', False))
    use_video_summary = bool(config.get('use_video_summary', False))
    audio_prefix = str(config.get('audio_summary_prefix', 'Audio summary:')).strip()
    video_prefix = str(config.get('video_summary_prefix', 'Video summary:')).strip()

    rows = []
    for dialogue in loaded_split.dialogues:
        doc_id = dialogue.metadata.get('doc_id')
        if doc_id is None:
            raise ValueError(
                f"Unified loader did not provide doc_id for dataset={loaded_split.dataset} "
                f"dialogue_id={dialogue.dialogue_id}"
            )
        rows.append(
            {
                'doc_id': int(doc_id),
                'emotion_cause_pairs': [(int(w), int(z)) for w, z in dialogue.emotion_cause_pairs],
                'lines': [
                    _augment_utterance_text(
                        utterance.text,
                        dialogue_id=str(dialogue.dialogue_id),
                        turn_id=int(utterance.turn),
                        use_audio_summary=use_audio_summary,
                        use_video_summary=use_video_summary,
                        audio_summary_map=audio_summary_map,
                        video_summary_map=video_summary_map,
                        audio_prefix=audio_prefix,
                        video_prefix=video_prefix,
                    )
                    for utterance in dialogue.utterances
                ],
                'speakers': [utterance.speaker for utterance in dialogue.utterances],
                'emotions': [utterance.emotion or 'neutral' for utterance in dialogue.utterances],
                'timecodes': [utterance.timecode or '' for utterance in dialogue.utterances],
                'raw_doc_id': str(dialogue.metadata.get('raw_doc_id', dialogue.dialogue_id)),
                'pair_role_labels': labels_for_dialogue(
                    pair_role_label_index or {},
                    split=str(loaded_split.split),
                    dialogue=dialogue,
                ),
            }
        )
    return rows


def _validate_mec4_emotion_categories(config, rows: list[dict]) -> None:
    if str(config.get('dataset_name', '')).lower() != 'mec4':
        return

    counts = Counter()
    for row in rows:
        counts.update(str(emotion).lower() for emotion in row.get('emotions', []))

    labels = set(counts)
    category_labels = labels - {'neutral', 'neu', 'emotion'}
    if bool(config.get('mec4_require_emotion_categories', False)) and not category_labels:
        if labels <= {'neutral', 'neu'}:
            reason = 'only neutral labels were loaded'
        elif labels <= {'neutral', 'neu', 'emotion'}:
            reason = 'only binary neutral/emotion labels were loaded'
        else:
            reason = f'unsupported MEC4 emotion labels were loaded: {sorted(labels)}'
        raise ValueError(
            f"MEC4 requires utterance-level emotion category labels, but {reason}. "
            "The public MEC4_text files expose emotion-cause pair turns, but not Anger/Surprise/etc. "
            "Re-export MEC4_text from the original annotations with per-turn category labels, "
            "or set mec4_require_emotion_categories=false to use pair-derived binary labels. "
            f"Loaded label counts: {dict(counts)}"
        )

    if bool(config.get('mec4_require_non_neutral_emotions', True)) and labels <= {'neutral', 'neu'}:
        raise ValueError(
            "MEC4 requires at least pair-derived binary emotion labels, but only neutral labels were loaded. "
            "This usually means a stale preprocessed cache was built before pair-derived labels were enabled. "
            "Remove stale files under MEC4/preprocessed or use a cache path generated by the current code. "
            f"Loaded label counts: {dict(counts)}"
        )


def _augment_utterance_text(
    base_text: str,
    *,
    dialogue_id: str,
    turn_id: int,
    use_audio_summary: bool,
    use_video_summary: bool,
    audio_summary_map: Dict[tuple[str, int], str],
    video_summary_map: Dict[tuple[str, int], str],
    audio_prefix: str,
    video_prefix: str,
) -> str:
    segments = [str(base_text).strip()]
    summary_key = (str(dialogue_id), int(turn_id))
    if use_audio_summary:
        audio_summary = audio_summary_map.get(summary_key)
        if audio_summary:
            segments.append(f"{audio_prefix} {audio_summary}")
    if use_video_summary:
        video_summary = video_summary_map.get(summary_key)
        if video_summary:
            segments.append(f"{video_prefix} {video_summary}")
    return " ".join(segment for segment in segments if segment)


def _resolve_dialogue_limit(config, split: str, total_dialogues: int) -> int | None:
    split = split.lower()
    explicit_key = f'max_{split}_dialogues'
    explicit_value = config.get(explicit_key)
    if explicit_value is not None:
        return max(1, min(int(explicit_value), total_dialogues))

    fraction = config.get(f'{split}_dialogue_fraction')
    if fraction is None:
        fraction = config.get('dialogue_fraction')
    if fraction is None:
        return None
    fraction = float(fraction)
    if not (0 < fraction <= 1):
        raise ValueError(f"dialogue_fraction must be in (0, 1], got {fraction}")
    return max(1, min(int(math.ceil(total_dialogues * fraction)), total_dialogues))


def _summary_cache_tag(config, kind: str) -> str:
    split_records = []
    for split in ('train', 'valid', 'test'):
        path = _resolve_summary_path(config, kind, split)
        split_records.append(
            {
                'split': split,
                'path': str(path) if path is not None else None,
                'exists': bool(path is not None and path.exists()),
                'mtime_ns': int(path.stat().st_mtime_ns) if path is not None and path.exists() else None,
            }
        )
    payload = {
        'version': _SUMMARY_CACHE_VERSION,
        'enabled': bool(config.get(f'use_{kind}_summary', False)),
        'template': config.get(f'{kind}_summary_path_template'),
        'train': config.get(f'{kind}_summary_train_path'),
        'valid': config.get(f'{kind}_summary_valid_path'),
        'test': config.get(f'{kind}_summary_test_path'),
        'prefix': config.get(f'{kind}_summary_prefix'),
        'splits': split_records,
    }
    digest = hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode('utf-8')).hexdigest()
    return digest[:8]


def _build_unified_data(config) -> Dict:
    data = {}
    all_rows_for_vocab: list[dict] = []
    pair_role_task = normalize_pair_role_task(config.get('pair_role_task', config.get('verifiability_task', 'none')))
    context_subtype_task = normalize_context_subtype_task(
        config.get('context_subtype_task', config.get('cause_role_task', 'none'))
    )
    label_task = pair_role_task if pair_role_task != 'none' else ('3class' if context_subtype_task != 'none' else 'none')
    label_config = dict(config)
    label_config['pair_role_task'] = label_task
    pair_role_label_index = (
        load_pair_role_label_index(label_config)
        if label_task != 'none'
        else {}
    )
    loaded_splits = {
        mode: load_split(
            config['dataset_name'],
            mode,
            data_root=config['data_root'],
            candidate_scope=config.get('candidate_scope', 'lower_triangle'),
        )
        for mode in ['train', 'valid', 'test']
    }
    use_audio_summary = bool(config.get('use_audio_summary', False))
    use_video_summary = bool(config.get('use_video_summary', False))
    for mode, loaded_split in loaded_splits.items():
        audio_summary_map = _load_summary_map(config, 'audio', mode) if use_audio_summary else {}
        video_summary_map = _load_summary_map(config, 'video', mode) if use_video_summary else {}
        full_rows = _build_structured_rows_from_loaded_split(
            loaded_split,
            config=config,
            pair_role_label_index=pair_role_label_index,
            audio_summary_map=audio_summary_map,
            video_summary_map=video_summary_map,
        )
        all_rows_for_vocab.extend(full_rows)
        limit = _resolve_dialogue_limit(config, mode, len(loaded_split.dialogues))
        data[mode] = full_rows[:limit] if limit is not None else full_rows
        data[f'audio_{mode}'] = _load_feature_map(loaded_split.feature_paths.audio_pt)
        data[f'video_{mode}'] = _load_feature_map(loaded_split.feature_paths.video_pt)
    _validate_mec4_emotion_categories(config, all_rows_for_vocab)
    data['label_dict'] = build_dict(all_rows_for_vocab)
    data['speaker_dict'] = build_speaker_dict(all_rows_for_vocab)
    return data


def _preprocessed_cache_path(config) -> str:
    backbone_name = str(config['backbone']).replace('/', '__')
    dataset_name = str(config.get('dataset_name', 'legacy')).lower()
    suffix_parts = [backbone_name, dataset_name]
    suffix_parts.append(f"scope_{str(config.get('candidate_scope', 'lower_triangle'))}")
    if dataset_name == 'mec4':
        suffix_parts.append('label_v2')
    fraction = config.get('dialogue_fraction')
    if fraction is not None:
        suffix_parts.append(f"frac_{str(fraction).replace('.', 'p')}")
    for split in ('train', 'valid', 'test'):
        split_fraction = config.get(f'{split}_dialogue_fraction')
        if split_fraction is not None:
            suffix_parts.append(f"{split}_frac_{str(split_fraction).replace('.', 'p')}")
    for split in ('train', 'valid', 'test'):
        value = config.get(f'max_{split}_dialogues')
        if value is not None:
            suffix_parts.append(f"{split}_{int(value)}")
    if bool(config.get('use_audio_summary', False)):
        suffix_parts.append(f"asum_{_summary_cache_tag(config, 'audio')}")
    if bool(config.get('use_video_summary', False)):
        suffix_parts.append(f"vsum_{_summary_cache_tag(config, 'video')}")
    pair_role_task = normalize_pair_role_task(config.get('pair_role_task', config.get('verifiability_task', 'none')))
    if pair_role_task != 'none':
        suffix_parts.append(f"pair_role_{pair_role_task}_{pair_role_cache_tag(config)}")
    context_subtype_task = normalize_context_subtype_task(
        config.get('context_subtype_task', config.get('cause_role_task', 'none'))
    )
    if context_subtype_task != 'none':
        suffix_parts.append(f"context_subtype_{context_subtype_task}_{context_subtype_cache_tag(config)}")
    return os.path.join(config['preprocessed_dir'], f'{"__".join(suffix_parts)}.pkl')


def _load_or_build_cached_data(config) -> Dict:
    path = _preprocessed_cache_path(config)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pkl.load(f)

    fallback_dir = config.get('preprocessed_fallback_dir')
    if fallback_dir:
        fallback_path = os.path.join(fallback_dir, os.path.basename(path))
        if os.path.isfile(fallback_path):
            with open(fallback_path, 'rb') as f:
                return pkl.load(f)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    lock_path = f"{path}.lock"
    with open(lock_path, 'w', encoding='utf-8') as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            if not os.path.exists(path):
                data = _build_unified_data(config)
                with open(path, 'wb') as f:
                    pkl.dump(data, f)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    with open(path, 'rb') as f:
        return pkl.load(f)

def build_dict(data):
    labels = []
    seen = set()
    for line in data:
        for emotion in line['emotions']:
            if emotion in seen:
                continue
            seen.add(emotion)
            labels.append(emotion)

    def sort_key(label):
        label_text = str(label)
        lowered = label_text.lower()
        is_neutral = lowered in {'neutral', 'neu'}
        return (0 if is_neutral else 1, lowered, label_text)

    labels = sorted(labels, key=sort_key)
    return {label: idx for idx, label in enumerate(labels)}


def resolve_neutral_label_id(label_dict) -> int | None:
    exact_candidates = ('neutral', 'neu', 'Neutral')
    for candidate in exact_candidates:
        if candidate in label_dict:
            return int(label_dict[candidate])
    for label, index in label_dict.items():
        if str(label).lower() in {'neutral', 'neu'}:
            return int(index)
    return None

def build_speaker_dict(data):
    wordlist = []
    for line in data:
        wordlist.extend(line['speakers'])
    wordcount = Counter(wordlist)
    word2dict = {w: i for i, w in enumerate(wordcount.keys())}
    return word2dict

def make_paceext_datamodule(config, tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    pair_role_task = normalize_pair_role_task(config.get('pair_role_task', config.get('verifiability_task', 'none')))
    context_subtype_task = normalize_context_subtype_task(
        config.get('context_subtype_task', config.get('cause_role_task', 'none'))
    )
    config['pair_role_task'] = pair_role_task
    config['pair_role_enable'] = pair_role_task != 'none'
    config['pair_role_ignore_index'] = PAIR_ROLE_IGNORE_INDEX
    config['pair_role_class_names'] = pair_role_class_names(pair_role_task)
    config['pair_role_cause_class_id'] = PAIR_ROLE_CAUSE_CLASS_ID
    config['pair_role_context_class_id'] = PAIR_ROLE_CONTEXT_CLASS_ID
    config['pair_role_non_pair_class_id'] = PAIR_ROLE_NON_PAIR_CLASS_ID
    config['pair_num_classes'] = len(config['pair_role_class_names']) if config['pair_role_enable'] else 2
    config['context_subtype_task'] = context_subtype_task
    config['context_subtype_enable'] = context_subtype_task != 'none'
    config['context_subtype_class_names'] = context_subtype_class_names(context_subtype_task)
    config['context_subtype_num_classes'] = len(config['context_subtype_class_names'])

    data = _load_or_build_cached_data(config)
    if 'label_dict' not in data:
        data['label_dict'] = build_dict(data['train'] + data['valid'] + data['test'])
    if 'speaker_dict' not in data:
        data['speaker_dict'] = build_speaker_dict(data['train'] + data['valid'] + data['test'])
    _validate_mec4_emotion_categories(config, data['train'] + data['valid'] + data['test'])
    config['label_dict'] = data['label_dict']
    config['speaker_dict'] = data['speaker_dict']
    neutral_label_id = resolve_neutral_label_id(data['label_dict'])
    config['neutral_label_id'] = 0 if neutral_label_id is None else neutral_label_id

    # Distance prior from training set only (no leakage).
    max_utt = int(config.get('max_length', 35))
    smooth = float(config.get('dist_prior_smooth', 1.0))
    config['dist_prior'] = build_distance_prior(data['train'], max_utt=max_utt, smooth=smooth)

    if config['pair_role_enable']:
        train_non_pair_max_distance = config.get('pair_role_train_non_pair_max_distance')
        if train_non_pair_max_distance is not None:
            train_non_pair_max_distance = int(train_non_pair_max_distance)
        train_counts = count_pair_role_labels_for_task(
            data['train'],
            pair_role_task,
            non_pair_max_distance=train_non_pair_max_distance,
        )
        all_counts = Counter()
        for split_name in ('train', 'valid', 'test'):
            all_counts.update(count_pair_role_labels_for_task(data[split_name], pair_role_task))
        if bool(config.get('pair_role_require_labels', True)) and sum(train_counts.values()) == 0:
            raise ValueError(
                f"pair_role_task={pair_role_task} is enabled, but no train labels matched "
                f"dataset={config.get('dataset_name')}."
            )
        config['pair_role_train_label_counts'] = {
            config['pair_role_class_names'][int(label_id)]: int(count)
            for label_id, count in sorted(train_counts.items())
        }
        config['pair_role_all_label_counts'] = {
            config['pair_role_class_names'][int(label_id)]: int(count)
            for label_id, count in sorted(all_counts.items())
        }
        config['pair_role_class_weights'] = make_class_weights(
            train_counts,
            int(config['pair_num_classes']),
            max_weight=float(config.get('pair_role_max_class_weight', 10.0)),
        )
        context_weight_multiplier = float(config.get('pair_role_context_class_weight_multiplier', 1.0))
        if context_weight_multiplier != 1.0:
            context_id = int(config['pair_role_context_class_id'])
            config['pair_role_class_weights'][context_id] = float(
                config['pair_role_class_weights'][context_id] * context_weight_multiplier
            )
        related_counts = Counter({
            0: int(train_counts.get(config['pair_role_cause_class_id'], 0))
            + int(train_counts.get(config['pair_role_context_class_id'], 0)),
            1: int(train_counts.get(config['pair_role_non_pair_class_id'], 0)),
        })
        config['pair_role_related_class_weights'] = make_class_weights(
            related_counts,
            2,
            max_weight=float(config.get('pair_role_max_class_weight', 10.0)),
        )
    if config['context_subtype_enable']:
        num_classes = int(config['context_subtype_num_classes'])
        train_counts, train_items = count_context_subtype_labels(data['train'], num_classes)
        all_counts = Counter()
        all_items = 0
        for split_name in ('train', 'valid', 'test'):
            split_counts, split_items = count_context_subtype_labels(data[split_name], num_classes)
            all_counts.update(split_counts)
            all_items += int(split_items)
        if bool(config.get('pair_role_require_labels', True)) and train_items == 0:
            raise ValueError(
                f"context_subtype_task={context_subtype_task} is enabled, but no train labels matched "
                f"dataset={config.get('dataset_name')}."
            )
        config['context_subtype_train_label_counts'] = {
            config['context_subtype_class_names'][int(label_id)]: int(count)
            for label_id, count in sorted(train_counts.items())
        }
        config['context_subtype_all_label_counts'] = {
            config['context_subtype_class_names'][int(label_id)]: int(count)
            for label_id, count in sorted(all_counts.items())
        }
        config['context_subtype_train_item_count'] = int(train_items)
        config['context_subtype_all_item_count'] = int(all_items)
        config['context_subtype_pos_weights'] = make_multilabel_pos_weights(
            train_counts,
            num_classes,
            train_items,
            max_weight=float(config.get('context_subtype_max_pos_weight', 10.0)),
        )

    train_dataset = PaceExtDataset(data, 'train')
    valid_dataset = PaceExtDataset(data, 'valid')
    test_dataset = PaceExtDataset(data, 'test')
    teacher_annotation_dir = config.get('teacher_annotation_dir')
    teacher_enable = bool(config.get('teacher_distill_enable', False)) and teacher_annotation_dir
    train_teacher_annotations = _load_teacher_annotations(teacher_annotation_dir, 'train', keep_only=True) if teacher_enable else {}
    valid_teacher_annotations = {}
    test_teacher_annotations = {}
    # 为不同split创建对应的collator
    train_collator = PaceExtCollator(
        tokenizer=tokenizer,
        config=config,
        video_map=data['video_train'],
        audio_map=data['audio_train'],
        teacher_annotations=train_teacher_annotations,
        mode="train",
    )
    valid_collator = PaceExtCollator(
        tokenizer=tokenizer,
        config=config,
        video_map=data['video_valid'],
        audio_map=data['audio_valid'],
        teacher_annotations=valid_teacher_annotations,
        mode="valid",
    )
    test_collator = PaceExtCollator(
        tokenizer=tokenizer,
        config=config,
        video_map=data['video_test'],
        audio_map=data['audio_test'],
        teacher_annotations=test_teacher_annotations,
        mode="test",
    )

    base_seed = int(config.get('seed', 42))

    def seed_worker(worker_id):
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    generator = torch.Generator()
    generator.manual_seed(base_seed)

    num_workers = int(config.get('num_workers', min(8, (os.cpu_count() or 4) // 2)))
    shuffle = bool(config.get('shuffle', True))

    sampler = None
    context_oversample_factor = float(config.get('pair_role_context_oversample_factor', 1.0))
    if config['pair_role_enable'] and context_oversample_factor > 1.0:
        weights = []
        for row in data['train']:
            has_context = any(int(item.get('label', -1)) == PAIR_ROLE_CONTEXT_CLASS_ID for item in row.get('pair_role_labels', []))
            weights.append(float(context_oversample_factor if has_context else 1.0))
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)
        shuffle = False

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=shuffle if sampler is None else False, sampler=sampler, collate_fn=train_collator,
                              num_workers=num_workers, pin_memory=True,
                              worker_init_fn=seed_worker, generator=generator)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=valid_collator,
                              num_workers=num_workers, pin_memory=True,
                              worker_init_fn=seed_worker, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=test_collator,
                             num_workers=num_workers, pin_memory=True,
                             worker_init_fn=seed_worker, generator=generator)

    return train_loader, valid_loader, test_loader, config


SupervisedDataset = PaceExtDataset
CollateFN = PaceExtCollator
make_supervised_data_module = make_paceext_datamodule
