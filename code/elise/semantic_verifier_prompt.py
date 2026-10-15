#!/usr/bin/env python
"""Frozen action-blind prompt contract."""

from __future__ import annotations

import hashlib
import json
from typing import Any


PROMPT_VERSION = "semantic-verifier-matched-text-v2"
CONDITIONS = ("blind",)

SYSTEM_PROMPT = """You are a strict semantic verifier for emotion-cause pair extraction.
Judge the candidate from the supplied dialogue evidence. Any explicitly supplied corpus label or model metadata is context, never ground truth.
Return only valid JSON without markdown."""

USER_PROMPT_TEMPLATE = """Task: decide whether the candidate utterance genuinely helps explain the target speaker's expressed emotional state at the target turn.

<MODEL_METADATA>
{metadata_block}
</MODEL_METADATA>

<EMOTION_LABEL_METADATA>
{emotion_label_block}
</EMOTION_LABEL_METADATA>

Target utterance:
- turn={emotion_turn}
- speaker={emotion_speaker}
- text={emotion_text}

Candidate cause utterance:
- turn={cause_turn}
- speaker={cause_speaker}
- text={cause_text}

Dialogue context:
{dialogue_text}

Decision criteria:
- Return support_pair only when the candidate provides direct or clearly necessary explanatory evidence for the target state.
- Reject mere proximity, shared topic, emotional wording, or same-speaker coincidence.
- Return uncertain when the displayed text does not support a reliable binary judgment.

Return exactly this JSON object (no reason or free-form explanation):
{{
  "verdict": "support_pair" | "reject_pair" | "uncertain",
  "confidence": 0.0,
  "evidence_turns": [integer]
}}"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["support_pair", "reject_pair", "uncertain"],
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "evidence_turns": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["verdict", "confidence", "evidence_turns"],
    "additionalProperties": False,
}


def selected_utterances(
    record: dict[str, Any],
    emotion_turn: int,
    cause_turn: int,
    *,
    window: int,
    full_max: int,
) -> list[dict[str, Any]]:
    utterances = record["utterances"]
    if len(utterances) <= full_max:
        return utterances
    low = max(1, min(emotion_turn, cause_turn) - window)
    high = max(emotion_turn, cause_turn) + window
    return [
        utterance
        for utterance in utterances
        if low <= int(utterance["turn"]) <= high
    ]


def utterance_by_turn(record: dict[str, Any], turn: int) -> dict[str, Any]:
    for utterance in record["utterances"]:
        if int(utterance["turn"]) == int(turn):
            return utterance
    raise KeyError(f"Missing turn {turn} in dialogue {record['raw_doc_id']}")


def format_dialogue(
    record: dict[str, Any],
    emotion_turn: int,
    cause_turn: int,
    *,
    window: int,
    full_max: int,
) -> str:
    shown = selected_utterances(
        record,
        emotion_turn,
        cause_turn,
        window=window,
        full_max=full_max,
    )
    lines: list[str] = []
    if len(shown) != len(record["utterances"]):
        lines.append("[Only a local context window is shown because the dialogue is long.]")
    for utterance in shown:
        turn = int(utterance["turn"])
        markers = []
        if turn == emotion_turn:
            markers.append("TARGET")
        if turn == cause_turn:
            markers.append("CANDIDATE_CAUSE")
        marker_text = f" [{' | '.join(markers)}]" if markers else ""
        lines.append(
            f"{turn}. speaker={utterance['speaker']}{marker_text}: {utterance['text']}"
        )
    return "\n".join(lines)


def metadata_block(condition: str, occurrence: dict[str, Any]) -> str:
    if condition != "blind":
        raise ValueError("This release exposes only action-blind verification")
    return "No model action or score is supplied."


def emotion_label_block(condition: str, record: dict[str, Any], emotion_turn: int, cause_turn: int, *, window: int, full_max: int) -> str:
    if condition != "blind":
        raise ValueError("This release exposes only action-blind verification")
    return "No emotion category labels are supplied."


def build_prompt(
    occurrence: dict[str, Any],
    record: dict[str, Any],
    condition: str,
    *,
    window: int,
    full_max: int,
) -> str:
    emotion_turn = int(occurrence["emotion_turn"])
    cause_turn = int(occurrence["cause_turn"])
    emotion_utterance = utterance_by_turn(record, emotion_turn)
    cause_utterance = utterance_by_turn(record, cause_turn)
    return USER_PROMPT_TEMPLATE.format(
        metadata_block=metadata_block(condition, occurrence),
        emotion_label_block=emotion_label_block(
            condition,
            record,
            emotion_turn,
            cause_turn,
            window=window,
            full_max=full_max,
        ),
        emotion_turn=emotion_turn,
        emotion_speaker=emotion_utterance["speaker"],
        emotion_text=emotion_utterance["text"],
        cause_turn=cause_turn,
        cause_speaker=cause_utterance["speaker"],
        cause_text=cause_utterance["text"],
        dialogue_text=format_dialogue(
            record,
            emotion_turn,
            cause_turn,
            window=window,
            full_max=full_max,
        ),
    )


def template_sha256() -> str:
    canonical = SYSTEM_PROMPT + "\n" + USER_PROMPT_TEMPLATE
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def messages_sha256(messages: list[dict[str, str]]) -> str:
    canonical = json.dumps(
        messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def semantic_request_id(
    *,
    condition: str,
    candidate_scope: str,
    cache_key: str,
    messages: list[dict[str, str]],
) -> str:
    message_hash = messages_sha256(messages)
    text = "||".join(
        (PROMPT_VERSION, condition, candidate_scope, cache_key, message_hash)
    )
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    return f"elise-{digest}"
