#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from tqdm import tqdm


RUNNER_CONTRACT_VERSION = "elise-vllm-runner-p0-v1"

VERIFICATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["support_pair", "reject_pair", "uncertain"],
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "evidence_turns": {
            "type": "array",
            "items": {"type": "integer"},
        },
        "reason": {
            "type": "string",
        },
    },
    "required": ["verdict", "confidence", "evidence_turns", "reason"],
    "additionalProperties": False,
}

COMPACT_VERIFICATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["support_pair", "reject_pair", "uncertain"],
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "evidence_turns": {
            "type": "array",
            "items": {"type": "integer"},
        },
    },
    "required": ["verdict", "confidence", "evidence_turns"],
    "additionalProperties": False,
}

PAIR_GENERATION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "emotion_turn": {"type": "integer"},
                    "cause_turn": {"type": "integer"},
                },
                "required": ["emotion_turn", "cause_turn"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["pairs"],
    "additionalProperties": False,
}

SCHEMAS = {
    "verification": VERIFICATION_JSON_SCHEMA,
    "compact_verification": COMPACT_VERIFICATION_JSON_SCHEMA,
    "pair_generation": PAIR_GENERATION_JSON_SCHEMA,
}


class ContractError(ValueError):
    """Raised before model loading when a frozen artifact contract is violated."""


class IncompleteScoringError(RuntimeError):
    """Raised after inference when one or more requests remain schema-invalid."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def messages_sha256(messages: list[dict[str, Any]]) -> str:
    return sha256_json(messages)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ELISE LLM verification with offline vLLM."
    )
    parser.add_argument("--input", required=True, help="Request JSONL file or directory.")
    parser.add_argument("--output", required=True, help="Output JSONL file or directory.")
    parser.add_argument(
        "--model",
        required=True,
        help="Local Qwen3.6-27B snapshot; see configs/verifier.json for the fixed revision.",
    )
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--include-glob", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument(
        "--trust-remote-code", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--language-model-only", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-structured-outputs", action="store_true")
    parser.add_argument(
        "--structured-output-schema",
        choices=tuple(SCHEMAS),
        default="compact_verification",
        help="JSON grammar used for constrained decoding (final pair cache default: compact).",
    )
    parser.add_argument(
        "--strict-request-contract",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require every request to carry the exact response schema, messages hash, "
            "prompt mode, scope, and verification cache metadata."
        ),
    )
    parser.add_argument(
        "--max-schema-retries",
        type=int,
        default=2,
        help="Retries after the first model attempt for schema-invalid responses.",
    )
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--gpu-memory-monitor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Read-only nvidia-smi sampling for the runtime sidecar.",
    )
    parser.add_argument("--gpu-memory-sample-interval", type=float, default=2.0)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate requests/resume contracts and write sidecars without importing vLLM.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0:
        raise ContractError("--batch-size must be positive")
    if args.tensor_parallel_size <= 0:
        raise ContractError("--tensor-parallel-size must be positive")
    if args.max_schema_retries < 0:
        raise ContractError("--max-schema-retries must be non-negative")
    if args.limit is not None and args.limit < 0:
        raise ContractError("--limit must be non-negative")
    if args.gpu_memory_sample_interval <= 0:
        raise ContractError("--gpu-memory-sample-interval must be positive")
    if args.strict_request_contract and args.disable_structured_outputs:
        raise ContractError(
            "Strict final-cache mode requires structured outputs; remove "
            "--disable-structured-outputs or explicitly use --no-strict-request-contract."
        )


def iter_request_files(path: Path, include_globs: list[str] | None) -> list[Path]:
    if path.is_file():
        return [path]
    patterns = include_globs or ["*_requests.jsonl"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(path.glob(f"**/{pattern}"))
    return sorted(set(files))


def output_path_for(input_file: Path, input_root: Path, output_root: Path) -> Path:
    if input_root.is_file():
        return output_root
    rel = input_file.relative_to(input_root)
    return output_root / rel.parent / rel.name.replace(
        "_requests.jsonl", "_results.jsonl"
    )


def output_contract_path(output_file: Path) -> Path:
    return Path(str(output_file) + ".contract.json")


def runtime_sidecar_path(output_file: Path) -> Path:
    return Path(str(output_file) + ".runtime.json")


def failure_path(output_file: Path) -> Path:
    if output_file.name.endswith("_results.jsonl"):
        return output_file.with_name(
            output_file.name[: -len("_results.jsonl")] + "_failures.jsonl"
        )
    return Path(str(output_file) + ".failures.jsonl")


def strict_response_error(content: str, schema_name: str) -> tuple[dict[str, Any], str]:
    text = str(content or "").strip()
    if not text:
        return {}, "empty_response"
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return {}, f"json_decode_error:{exc}"
    if not isinstance(payload, dict):
        return {}, "response_not_object"

    expected_keys = set(SCHEMAS[schema_name]["required"])
    actual_keys = set(payload)
    if actual_keys != expected_keys:
        return payload, (
            "schema_keys_mismatch:"
            f"missing={sorted(expected_keys - actual_keys)}:"
            f"extra={sorted(actual_keys - expected_keys)}"
        )

    if schema_name in {"verification", "compact_verification"}:
        if payload["verdict"] not in {"support_pair", "reject_pair", "uncertain"}:
            return payload, "invalid_verdict"
        confidence = payload["confidence"]
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            return payload, "invalid_confidence"
        evidence_turns = payload["evidence_turns"]
        if not isinstance(evidence_turns, list) or any(
            isinstance(turn, bool) or not isinstance(turn, int)
            for turn in evidence_turns
        ):
            return payload, "invalid_evidence_turns"
        if schema_name == "verification" and not isinstance(payload["reason"], str):
            return payload, "invalid_reason"
        return payload, ""

    pairs = payload["pairs"]
    if not isinstance(pairs, list):
        return payload, "pairs_not_array"
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            return payload, f"pair_{index}_not_object"
        if set(pair) != {"emotion_turn", "cause_turn"}:
            return payload, f"pair_{index}_keys_mismatch"
        for field in ("emotion_turn", "cause_turn"):
            value = pair[field]
            if isinstance(value, bool) or not isinstance(value, int):
                return payload, f"pair_{index}_{field}_not_integer"
    return payload, ""


def request_prompt_version(request: dict[str, Any]) -> str:
    prompt_version = str(request.get("prompt_version") or "")
    prompt_variant = str(request.get("prompt_variant") or "")
    if prompt_version and prompt_variant and prompt_version != prompt_variant:
        raise ContractError(
            f"{request.get('request_id')}: prompt_version={prompt_version!r} "
            f"does not match prompt_variant={prompt_variant!r}"
        )
    return prompt_version or prompt_variant


def validate_request(
    request: dict[str, Any],
    *,
    path: Path,
    line_number: int,
    schema_name: str,
    strict: bool,
) -> dict[str, Any]:
    location = f"{path}:{line_number}"
    request_id = str(request.get("request_id") or "")
    if not request_id:
        raise ContractError(f"{location}: missing request_id")
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ContractError(f"{location}: {request_id}: messages must be a non-empty list")
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ContractError(
                f"{location}: {request_id}: message {message_index} is not an object"
            )
        if not isinstance(message.get("role"), str) or not isinstance(
            message.get("content"), str
        ):
            raise ContractError(
                f"{location}: {request_id}: message {message_index} needs string role/content"
            )

    computed_messages_hash = messages_sha256(messages)
    supplied_messages_hash = str(request.get("messages_sha256") or "")
    if strict and not supplied_messages_hash:
        raise ContractError(f"{location}: {request_id}: missing messages_sha256")
    if supplied_messages_hash and supplied_messages_hash != computed_messages_hash:
        raise ContractError(
            f"{location}: {request_id}: messages_sha256 mismatch "
            f"{supplied_messages_hash}!={computed_messages_hash}"
        )

    expected_schema = SCHEMAS[schema_name]
    supplied_schema = request.get("response_schema")
    if strict and supplied_schema is None:
        raise ContractError(f"{location}: {request_id}: missing response_schema")
    if supplied_schema is not None and supplied_schema != expected_schema:
        raise ContractError(
            f"{location}: {request_id}: response_schema does not match {schema_name}"
        )

    prompt_version = request_prompt_version(request)
    if strict and not prompt_version:
        raise ContractError(
            f"{location}: {request_id}: missing prompt_version/prompt_variant"
        )

    condition = str(
        request.get("condition") or request.get("anchoring_condition") or ""
    )
    if strict and schema_name in {"verification", "compact_verification"}:
        required_fields = (
            "pair_key",
            "cache_key",
            "candidate_scope",
            "emotion_label_mode",
        )
        missing = [field for field in required_fields if request.get(field) in (None, "")]
        if not condition:
            missing.append("condition")
        if missing:
            raise ContractError(
                f"{location}: {request_id}: missing verification contract fields {missing}"
            )
    elif strict and request.get("candidate_scope") in (None, ""):
        raise ContractError(f"{location}: {request_id}: missing candidate_scope")

    normalized = dict(request)
    normalized["_runner_messages_sha256"] = computed_messages_hash
    normalized["_runner_prompt_version"] = prompt_version
    normalized["_runner_condition"] = condition
    normalized["_runner_source_line"] = line_number
    return normalized


def request_contract_entry(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_id": str(request["request_id"]),
        "messages_sha256": request["_runner_messages_sha256"],
        "prompt_version": request["_runner_prompt_version"],
        "condition": request["_runner_condition"],
        "cache_key": request.get("cache_key"),
        "candidate_scope": request.get("candidate_scope"),
        "emotion_label_mode": request.get("emotion_label_mode"),
        "pair_key": request.get("pair_key"),
        "response_schema_sha256": sha256_json(request.get("response_schema")),
    }


def prepare_request_file(
    path: Path,
    *,
    schema_name: str,
    strict: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(raw, dict):
                raise ContractError(f"{path}:{line_number}: request must be an object")
            row = validate_request(
                raw,
                path=path,
                line_number=line_number,
                schema_name=schema_name,
                strict=strict,
            )
            request_id = str(row["request_id"])
            if request_id in ids:
                raise ContractError(f"{path}:{line_number}: duplicate request_id {request_id}")
            ids.add(request_id)
            rows.append(row)
    if not rows:
        raise ContractError(f"{path}: no requests")
    entries = [request_contract_entry(row) for row in rows]
    return {
        "input_file": path,
        "input_file_sha256": sha256_file(path),
        "request_contract_sha256": sha256_json(entries),
        "requests": rows,
        "request_by_id": {str(row["request_id"]): row for row in rows},
        "total_requests": len(rows),
    }


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def inference_config(args: argparse.Namespace) -> dict[str, Any]:
    schema = SCHEMAS[args.structured_output_schema]
    return {
        "runner_contract_version": RUNNER_CONTRACT_VERSION,
        "runner_source_sha256": sha256_file(Path(__file__).resolve()),
        "vllm_version": package_version("vllm"),
        "transformers_version": package_version("transformers"),
        "model": str(args.model),
        "tokenizer": str(args.tokenizer or args.model),
        "trust_remote_code": bool(args.trust_remote_code),
        "dtype": str(args.dtype),
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "max_model_len": int(args.max_model_len),
        "max_num_seqs": int(args.max_num_seqs),
        "language_model_only": bool(args.language_model_only),
        "enforce_eager": bool(args.enforce_eager),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "max_tokens": int(args.max_tokens),
        "seed": int(args.seed),
        "enable_thinking": bool(args.enable_thinking),
        "structured_outputs_enabled": not bool(args.disable_structured_outputs),
        "structured_output_schema": str(args.structured_output_schema),
        "structured_output_schema_sha256": sha256_json(schema),
        "strict_request_contract": bool(args.strict_request_contract),
        "batch_size": int(args.batch_size),
        "max_schema_retries": int(args.max_schema_retries),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


def ensure_output_contract(
    output_file: Path,
    prepared: dict[str, Any],
    *,
    config: dict[str, Any],
    config_hash: str,
    schema_name: str,
) -> dict[str, Any]:
    contract_path = output_contract_path(output_file)
    expected = {
        "runner_contract_version": RUNNER_CONTRACT_VERSION,
        "input_file": str(Path(prepared["input_file"]).resolve()),
        "input_file_sha256": prepared["input_file_sha256"],
        "request_contract_sha256": prepared["request_contract_sha256"],
        "total_requests": prepared["total_requests"],
        "output_file": str(output_file.resolve()),
        "inference_config_sha256": config_hash,
        "inference_config": config,
        "response_schema_name": schema_name,
        "response_schema_sha256": sha256_json(SCHEMAS[schema_name]),
    }
    if contract_path.exists():
        try:
            existing = json.loads(contract_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ContractError(f"Invalid output contract {contract_path}: {exc}") from exc
        for field, value in expected.items():
            if existing.get(field) != value:
                raise ContractError(
                    f"Resume guard failed for {output_file}: {field} changed; "
                    "use a new output path rather than mixing configurations."
                )
        return existing
    if output_file.exists() and output_file.stat().st_size:
        raise ContractError(
            f"{output_file} exists without {contract_path}; refusing unguarded resume"
        )
    payload = {**expected, "created_at": utc_now()}
    atomic_write_json(contract_path, payload)
    return payload


def result_metadata_mismatch(
    row: dict[str, Any],
    request: dict[str, Any],
    *,
    config_hash: str,
    prepared: dict[str, Any],
    schema_name: str,
) -> str:
    hard_expected = {
        "inference_config_sha256": config_hash,
        "request_contract_sha256": prepared["request_contract_sha256"],
        "response_schema_name": schema_name,
        "response_schema_sha256": sha256_json(SCHEMAS[schema_name]),
        "messages_sha256": request["_runner_messages_sha256"],
        "prompt_version": request["_runner_prompt_version"],
        "condition": request["_runner_condition"],
        "cache_key": request.get("cache_key"),
        "candidate_scope": request.get("candidate_scope"),
        "pair_key": request.get("pair_key"),
    }
    for field, expected in hard_expected.items():
        if row.get(field) != expected:
            return f"{field}:{row.get(field)!r}!={expected!r}"
    if row.get("response_schema") != request.get("response_schema"):
        return "response_schema:mismatch"
    return ""


def sanitize_existing_results(
    output_file: Path,
    prepared: dict[str, Any],
    *,
    config_hash: str,
    schema_name: str,
    job_id: str,
) -> tuple[set[str], dict[str, Any]]:
    if not output_file.exists() or not output_file.stat().st_size:
        return set(), {"existing_rows": 0, "valid_resume_rows": 0, "quarantined_rows": 0}
    request_by_id = prepared["request_by_id"]
    done: set[str] = set()
    valid_lines: list[str] = []
    rejected: list[dict[str, Any]] = []
    with output_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                rejected.append(
                    {
                        "line_number": line_number,
                        "reason": f"invalid_json:{exc}",
                        "original_line": line.rstrip("\n"),
                    }
                )
                continue
            request_id = str(row.get("request_id") or "")
            if request_id not in request_by_id:
                raise ContractError(
                    f"{output_file}:{line_number}: unexpected request_id {request_id!r}"
                )
            mismatch = result_metadata_mismatch(
                row,
                request_by_id[request_id],
                config_hash=config_hash,
                prepared=prepared,
                schema_name=schema_name,
            )
            if mismatch:
                raise ContractError(
                    f"{output_file}:{line_number}: resume metadata mismatch for "
                    f"{request_id}: {mismatch}"
                )
            _, response_error = strict_response_error(
                row.get("response_content", ""), schema_name
            )
            invalid_reasons = []
            if row.get("scored_by_model") != 1:
                invalid_reasons.append("scored_by_model_not_1")
            if row.get("schema_valid") != 1:
                invalid_reasons.append("schema_valid_not_1")
            if str(row.get("error") or ""):
                invalid_reasons.append("nonempty_error")
            if response_error:
                invalid_reasons.append(response_error)
            if request_id in done:
                invalid_reasons.append("duplicate_valid_request_id")
            if invalid_reasons:
                rejected.append(
                    {
                        "line_number": line_number,
                        "request_id": request_id,
                        "reason": ";".join(invalid_reasons),
                        "row": row,
                    }
                )
                continue
            done.add(request_id)
            valid_lines.append(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")

    if rejected:
        quarantine = Path(str(output_file) + f".resume_rejected.{job_id}.jsonl")
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        with quarantine.open("w", encoding="utf-8") as handle:
            for row in rejected:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        temporary = output_file.with_name(f".{output_file.name}.{job_id}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.writelines(valid_lines)
        os.replace(temporary, output_file)
    return done, {
        "existing_rows": len(valid_lines) + len(rejected),
        "valid_resume_rows": len(done),
        "quarantined_rows": len(rejected),
    }


def chunks(rows: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def output_content_and_tokens(output: Any) -> tuple[str, int, int]:
    sequence = output.outputs[0] if getattr(output, "outputs", None) else None
    content = sequence.text if sequence is not None else ""
    prompt_token_ids = getattr(output, "prompt_token_ids", None) or []
    completion_token_ids = getattr(sequence, "token_ids", None) or []
    return str(content), len(prompt_token_ids), len(completion_token_ids)


def common_result_fields(
    request: dict[str, Any],
    args: argparse.Namespace,
    *,
    prepared: dict[str, Any],
    config_hash: str,
    schema_name: str,
    job_id: str,
    batch_id: str,
    batch_attempt: int,
) -> dict[str, Any]:
    return {
        "runner_contract_version": RUNNER_CONTRACT_VERSION,
        "request_id": request["request_id"],
        "dataset": request.get("dataset"),
        "dataset_key": request.get("dataset_key"),
        "split": request.get("split"),
        "run_name": request.get("run_name"),
        "model": request.get("model"),
        "modality": request.get("modality"),
        "seed": request.get("seed"),
        "dialogue_id": request.get("dialogue_id"),
        "emotion_turn": request.get("emotion_turn"),
        "cause_turn": request.get("cause_turn"),
        "pair_key": request.get("pair_key"),
        "selector_action": request.get("selector_action"),
        "gold_pair": request.get("gold_pair"),
        "pred_pair": request.get("pred_pair"),
        "prompt_version": request["_runner_prompt_version"],
        "prompt_variant": request.get("prompt_variant")
        or request["_runner_prompt_version"],
        "prompt_template_sha256": request.get("prompt_template_sha256"),
        "messages_sha256": request["_runner_messages_sha256"],
        "emotion_label_mode": request.get("emotion_label_mode"),
        "candidate_source": request.get("candidate_source"),
        "candidate_scope": request.get("candidate_scope"),
        "condition": request["_runner_condition"],
        "cache_key": request.get("cache_key"),
        "candidate_occurrence_id": request.get("candidate_occurrence_id"),
        "response_schema": request.get("response_schema"),
        "response_schema_name": schema_name,
        "response_schema_sha256": sha256_json(SCHEMAS[schema_name]),
        "input_file_sha256": prepared["input_file_sha256"],
        "request_contract_sha256": prepared["request_contract_sha256"],
        "inference_config_sha256": config_hash,
        "llm_model": str(args.model),
        "llm_tokenizer": str(args.tokenizer or args.model),
        "job_id": job_id,
        "batch_id": batch_id,
        "batch_attempt": batch_attempt,
        "created_at": utc_now(),
    }


def result_record(
    request: dict[str, Any],
    output: Any,
    args: argparse.Namespace,
    *,
    prepared: dict[str, Any],
    config_hash: str,
    schema_name: str,
    job_id: str,
    batch_id: str,
    batch_attempt: int,
    batch_wall_time_seconds: float,
    batch_size: int,
) -> dict[str, Any]:
    content, prompt_tokens, completion_tokens = output_content_and_tokens(output)
    response_payload, response_error = strict_response_error(content, schema_name)
    if response_error:
        raise ContractError(
            f"result_record called for schema-invalid response: {response_error}"
        )
    return {
        **common_result_fields(
            request,
            args,
            prepared=prepared,
            config_hash=config_hash,
            schema_name=schema_name,
            job_id=job_id,
            batch_id=batch_id,
            batch_attempt=batch_attempt,
        ),
        "response_content": content,
        "response_payload": response_payload,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "batch_wall_time_seconds": batch_wall_time_seconds,
        "batch_size": batch_size,
        "estimated_request_wall_time_seconds": (
            batch_wall_time_seconds / batch_size if batch_size else 0.0
        ),
        "scored_by_model": 1,
        "schema_valid": 1,
        "error": "",
    }


def failure_record(
    request: dict[str, Any],
    output: Any,
    args: argparse.Namespace,
    *,
    prepared: dict[str, Any],
    config_hash: str,
    schema_name: str,
    job_id: str,
    batch_id: str,
    batch_attempt: int,
    batch_wall_time_seconds: float,
    batch_size: int,
    error: str,
    final_attempt: bool,
) -> dict[str, Any]:
    content, prompt_tokens, completion_tokens = output_content_and_tokens(output)
    return {
        **common_result_fields(
            request,
            args,
            prepared=prepared,
            config_hash=config_hash,
            schema_name=schema_name,
            job_id=job_id,
            batch_id=batch_id,
            batch_attempt=batch_attempt,
        ),
        "response_content": content,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "batch_wall_time_seconds": batch_wall_time_seconds,
        "batch_size": batch_size,
        "estimated_request_wall_time_seconds": (
            batch_wall_time_seconds / batch_size if batch_size else 0.0
        ),
        "scored_by_model": 0,
        "schema_valid": 0,
        "final_attempt": int(final_attempt),
        "error": error,
    }


def build_llm(args: argparse.Namespace) -> Any:
    from vllm import LLM

    kwargs: dict[str, Any] = {
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "trust_remote_code": args.trust_remote_code,
        "dtype": args.dtype,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "enforce_eager": args.enforce_eager,
        "seed": args.seed,
    }
    if args.language_model_only:
        kwargs["language_model_only"] = True
    return LLM(**kwargs)


def build_sampling_params(args: argparse.Namespace) -> Any:
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    kwargs: dict[str, Any] = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    if not args.disable_structured_outputs:
        kwargs["structured_outputs"] = StructuredOutputsParams(
            json=SCHEMAS[args.structured_output_schema],
            disable_additional_properties=True,
        )
    return SamplingParams(**kwargs)


def process_is_descendant_or_self(pid: int, root_pid: int) -> bool:
    current = pid
    seen: set[int] = set()
    while current > 1 and current not in seen:
        if current == root_pid:
            return True
        seen.add(current)
        try:
            status = Path(f"/proc/{current}/status").read_text(encoding="utf-8")
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            return False
        parent = None
        for line in status.splitlines():
            if line.startswith("PPid:"):
                parent = int(line.split()[1])
                break
        if parent is None or parent == current:
            return False
        current = parent
    return current == root_pid


def parse_memory_mib(value: str) -> float | None:
    stripped = value.strip()
    if stripped in {"", "N/A", "[N/A]", "Not Supported"}:
        return None
    try:
        return float(stripped.split()[0])
    except (TypeError, ValueError):
        return None


class GpuMemoryMonitor:
    """Read-only GPU-memory sampler; it never signals or modifies any process."""

    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = interval_seconds
        self.root_pid = os.getpid()
        self.executable = shutil.which("nvidia-smi")
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()
        self.samples = 0
        self.query_errors: list[str] = []
        self.devices: dict[str, dict[str, Any]] = {}
        self.peak_process_tree_used_mib = 0.0

    def _run_query(self, query: str) -> list[str]:
        assert self.executable is not None
        completed = subprocess.run(
            [
                self.executable,
                f"--query-{query.split(':', 1)[0]}={query.split(':', 1)[1]}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return [line for line in completed.stdout.splitlines() if line.strip()]

    def _sample(self) -> None:
        if self.executable is None:
            return
        try:
            gpu_lines = self._run_query("gpu:index,uuid,memory.used")
            process_lines = self._run_query(
                "compute-apps:gpu_uuid,pid,used_gpu_memory"
            )
            current_devices: dict[str, tuple[int, float]] = {}
            for line in gpu_lines:
                parts = [part.strip() for part in line.split(",")]
                if len(parts) != 3:
                    continue
                memory = parse_memory_mib(parts[2])
                if memory is None:
                    continue
                current_devices[parts[1]] = (int(parts[0]), memory)
            process_tree_total = 0.0
            for line in process_lines:
                parts = [part.strip() for part in line.split(",")]
                if len(parts) != 3:
                    continue
                try:
                    pid = int(parts[1])
                except ValueError:
                    continue
                memory = parse_memory_mib(parts[2])
                if memory is None:
                    continue
                if process_is_descendant_or_self(pid, self.root_pid):
                    process_tree_total += memory
            with self.lock:
                for gpu_uuid, (index, current_mib) in current_devices.items():
                    device = self.devices.setdefault(
                        gpu_uuid,
                        {
                            "index": index,
                            "baseline_total_used_mib": current_mib,
                            "peak_total_used_mib": current_mib,
                        },
                    )
                    device["peak_total_used_mib"] = max(
                        float(device["peak_total_used_mib"]), current_mib
                    )
                self.peak_process_tree_used_mib = max(
                    self.peak_process_tree_used_mib, process_tree_total
                )
                self.samples += 1
        except Exception as exc:  # instrumentation must not abort inference
            with self.lock:
                if len(self.query_errors) < 20:
                    self.query_errors.append(f"{type(exc).__name__}:{exc}")

    def _loop(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        if self.executable is None:
            return
        self._sample()
        self.thread = threading.Thread(
            target=self._loop,
            name="elise-readonly-gpu-memory-monitor",
            daemon=True,
        )
        self.thread.start()

    def stop(self) -> None:
        if self.thread is None:
            return
        self._sample()
        self.stop_event.set()
        self.thread.join(timeout=max(5.0, self.interval_seconds + 1.0))

    def summary(self) -> dict[str, Any]:
        with self.lock:
            devices = {}
            for gpu_uuid, values in self.devices.items():
                baseline = float(values["baseline_total_used_mib"])
                peak = float(values["peak_total_used_mib"])
                devices[gpu_uuid] = {
                    **values,
                    "peak_delta_from_baseline_mib": max(0.0, peak - baseline),
                }
            return {
                "available": self.executable is not None,
                "measurement": (
                    "read-only nvidia-smi sampling; process-tree peak is preferred, "
                    "total-device delta is a fallback estimate"
                ),
                "sample_interval_seconds": self.interval_seconds,
                "sample_count": self.samples,
                "peak_process_tree_used_mib": self.peak_process_tree_used_mib,
                "devices": devices,
                "query_errors": list(self.query_errors),
            }


def write_failure(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()


def run_file(
    llm: Any,
    sampling_params: Any,
    prepared: dict[str, Any],
    output_file: Path,
    args: argparse.Namespace,
    *,
    done: set[str],
    resume_diagnostics: dict[str, Any],
    config_hash: str,
    schema_name: str,
    job_id: str,
    file_index: int,
) -> dict[str, Any]:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    all_pending = [
        request
        for request in prepared["requests"]
        if str(request["request_id"]) not in done
    ]
    requests = all_pending
    if args.limit is not None:
        requests = requests[: args.limit]
    runtime: dict[str, Any] = {
        "runner_contract_version": RUNNER_CONTRACT_VERSION,
        "job_id": job_id,
        "input_file": str(prepared["input_file"]),
        "output_file": str(output_file),
        "input_file_sha256": prepared["input_file_sha256"],
        "request_contract_sha256": prepared["request_contract_sha256"],
        "inference_config_sha256": config_hash,
        "response_schema_name": schema_name,
        "started_at": utc_now(),
        "total_requests": prepared["total_requests"],
        "valid_resume_rows": len(done),
        "resume_diagnostics": resume_diagnostics,
        "scheduled_requests": len(requests),
        "deferred_by_limit": len(all_pending) - len(requests),
        "valid_results_written": 0,
        "unresolved_schema_failures": 0,
        "model_attempted_requests": 0,
        "schema_retry_requests": 0,
        "batches": 0,
        "inference_wall_time_seconds": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "status": "running",
    }
    if not requests:
        runtime["status"] = (
            "complete" if not all_pending else "deferred_by_limit"
        )
        runtime["finished_at"] = utc_now()
        atomic_write_json(runtime_sidecar_path(output_file), runtime)
        return runtime

    failure_file = failure_path(output_file)
    failure_file.parent.mkdir(parents=True, exist_ok=True)
    progress = tqdm(total=len(requests), desc=Path(prepared["input_file"]).name)
    failure_handle = None
    try:
        with output_file.open("a", encoding="utf-8") as result_handle:
            for batch_index, base_batch in enumerate(
                chunks(requests, args.batch_size), start=1
            ):
                retry_batch = base_batch
                for attempt in range(args.max_schema_retries + 1):
                    if not retry_batch:
                        break
                    batch_id = (
                        f"{job_id}:f{file_index:04d}:b{batch_index:06d}:a{attempt}"
                    )
                    batch_started = time.perf_counter()
                    outputs = llm.chat(
                        [request["messages"] for request in retry_batch],
                        sampling_params,
                        use_tqdm=False,
                        chat_template_content_format="string",
                        chat_template_kwargs={
                            "enable_thinking": bool(args.enable_thinking)
                        },
                    )
                    batch_wall = time.perf_counter() - batch_started
                    if len(outputs) != len(retry_batch):
                        raise RuntimeError(
                            f"{batch_id}: vLLM returned {len(outputs)} outputs for "
                            f"{len(retry_batch)} requests"
                        )
                    runtime["batches"] += 1
                    runtime["inference_wall_time_seconds"] += batch_wall
                    runtime["model_attempted_requests"] += len(retry_batch)
                    next_retry: list[dict[str, Any]] = []
                    for request, output in zip(retry_batch, outputs, strict=True):
                        content, prompt_tokens, completion_tokens = output_content_and_tokens(
                            output
                        )
                        runtime["prompt_tokens"] += prompt_tokens
                        runtime["completion_tokens"] += completion_tokens
                        _, response_error = strict_response_error(content, schema_name)
                        if response_error:
                            final_attempt = attempt == args.max_schema_retries
                            if failure_handle is None:
                                failure_handle = failure_file.open("a", encoding="utf-8")
                            write_failure(
                                failure_handle,
                                failure_record(
                                    request,
                                    output,
                                    args,
                                    prepared=prepared,
                                    config_hash=config_hash,
                                    schema_name=schema_name,
                                    job_id=job_id,
                                    batch_id=batch_id,
                                    batch_attempt=attempt,
                                    batch_wall_time_seconds=batch_wall,
                                    batch_size=len(retry_batch),
                                    error=response_error,
                                    final_attempt=final_attempt,
                                ),
                            )
                            if final_attempt:
                                runtime["unresolved_schema_failures"] += 1
                            else:
                                runtime["schema_retry_requests"] += 1
                                next_retry.append(request)
                            continue
                        record = result_record(
                            request,
                            output,
                            args,
                            prepared=prepared,
                            config_hash=config_hash,
                            schema_name=schema_name,
                            job_id=job_id,
                            batch_id=batch_id,
                            batch_attempt=attempt,
                            batch_wall_time_seconds=batch_wall,
                            batch_size=len(retry_batch),
                        )
                        result_handle.write(
                            json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                        )
                        runtime["valid_results_written"] += 1
                    result_handle.flush()
                    retry_batch = next_retry
                progress.update(len(base_batch))
    except Exception as exc:
        runtime["status"] = "failed"
        runtime["error"] = f"{type(exc).__name__}:{exc}"
        raise
    finally:
        progress.close()
        if failure_handle is not None:
            failure_handle.close()
        runtime["finished_at"] = utc_now()
        runtime["total_tokens"] = (
            runtime["prompt_tokens"] + runtime["completion_tokens"]
        )
        if runtime["status"] == "running":
            runtime["status"] = (
                "incomplete_schema_failures"
                if runtime["unresolved_schema_failures"]
                else ("partial_limit" if runtime["deferred_by_limit"] else "complete")
            )
        atomic_write_json(runtime_sidecar_path(output_file), runtime)

    print(
        f"WROTE {output_file} valid={runtime['valid_results_written']} "
        f"unresolved={runtime['unresolved_schema_failures']}"
    )
    if runtime["unresolved_schema_failures"]:
        raise IncompleteScoringError(
            f"{output_file}: {runtime['unresolved_schema_failures']} requests remain "
            f"schema-invalid after {args.max_schema_retries + 1} attempts"
        )
    return runtime


def job_sidecar_path(input_root: Path, output_root: Path, job_id: str) -> Path:
    base = output_root if input_root.is_dir() else output_root.parent
    return base / "_run_metadata" / f"{job_id}.json"


def main() -> None:
    args = parse_args()
    validate_args(args)
    started_perf = time.perf_counter()
    started_at = utc_now()
    input_root = Path(args.input)
    output_root = Path(args.output)
    files = iter_request_files(input_root, args.include_glob)
    if not files:
        raise SystemExit(f"No request files found under {input_root}")

    schema_name = args.structured_output_schema
    config = inference_config(args)
    config_hash = sha256_json(config)
    job_id = (
        "vllm-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:12]
    )
    job_path = job_sidecar_path(input_root, output_root, job_id)
    job: dict[str, Any] = {
        "runner_contract_version": RUNNER_CONTRACT_VERSION,
        "job_id": job_id,
        "status": "preflight",
        "started_at": started_at,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "request_files": [str(path) for path in files],
        "inference_config": config,
        "inference_config_sha256": config_hash,
        "response_schema_name": schema_name,
        "response_schema_sha256": sha256_json(SCHEMAS[schema_name]),
        "model_load_wall_time_seconds": 0.0,
        "vllm_sampling_setup_wall_time_seconds": 0.0,
        "inference_wall_time_seconds": 0.0,
        "files": [],
        "gpu_memory": {
            "available": False,
            "measurement": "not started during preflight",
        },
    }

    prepared_files: list[dict[str, Any]] = []
    global_ids: dict[str, str] = {}
    try:
        for file_index, input_file in enumerate(files, start=1):
            prepared = prepare_request_file(
                input_file,
                schema_name=schema_name,
                strict=bool(args.strict_request_contract),
            )
            for request_id in prepared["request_by_id"]:
                if request_id in global_ids:
                    raise ContractError(
                        f"request_id {request_id} appears in both "
                        f"{global_ids[request_id]} and {input_file}"
                    )
                global_ids[request_id] = str(input_file)
            output_file = output_path_for(input_file, input_root, output_root)
            ensure_output_contract(
                output_file,
                prepared,
                config=config,
                config_hash=config_hash,
                schema_name=schema_name,
            )
            done, resume_diagnostics = sanitize_existing_results(
                output_file,
                prepared,
                config_hash=config_hash,
                schema_name=schema_name,
                job_id=job_id,
            )
            prepared.update(
                {
                    "file_index": file_index,
                    "output_file": output_file,
                    "done": done,
                    "resume_diagnostics": resume_diagnostics,
                }
            )
            prepared_files.append(prepared)
        job["status"] = "preflight_complete"
        job["files"] = [
            {
                "input_file": str(item["input_file"]),
                "output_file": str(item["output_file"]),
                "total_requests": item["total_requests"],
                "valid_resume_rows": len(item["done"]),
                "pending_requests": item["total_requests"] - len(item["done"]),
                "input_file_sha256": item["input_file_sha256"],
                "request_contract_sha256": item["request_contract_sha256"],
                "resume_diagnostics": item["resume_diagnostics"],
            }
            for item in prepared_files
        ]
        atomic_write_json(job_path, job)
    except Exception as exc:
        job["status"] = "preflight_failed"
        job["error"] = f"{type(exc).__name__}:{exc}"
        job["finished_at"] = utc_now()
        job["job_wall_time_seconds"] = time.perf_counter() - started_perf
        atomic_write_json(job_path, job)
        raise

    print(
        f"PREFLIGHT files={len(prepared_files)} requests={len(global_ids)} "
        f"config={config_hash}"
    )
    if args.preflight_only:
        job["status"] = "preflight_only_complete"
        job["finished_at"] = utc_now()
        job["job_wall_time_seconds"] = time.perf_counter() - started_perf
        atomic_write_json(job_path, job)
        print(f"PREFLIGHT_ONLY {job_path}")
        return

    pending_total = sum(
        item["total_requests"] - len(item["done"]) for item in prepared_files
    )
    if pending_total == 0:
        job["status"] = "complete_noop"
        job["finished_at"] = utc_now()
        job["job_wall_time_seconds"] = time.perf_counter() - started_perf
        atomic_write_json(job_path, job)
        print(f"SKIP all requests already complete: {job_path}")
        return

    monitor = GpuMemoryMonitor(args.gpu_memory_sample_interval)
    incomplete_errors: list[str] = []
    try:
        if args.gpu_memory_monitor:
            monitor.start()
        setup_started = time.perf_counter()
        sampling_params = build_sampling_params(args)
        job["vllm_sampling_setup_wall_time_seconds"] = (
            time.perf_counter() - setup_started
        )
        model_started = time.perf_counter()
        llm = build_llm(args)
        job["model_load_wall_time_seconds"] = time.perf_counter() - model_started
        job["status"] = "inference"
        atomic_write_json(job_path, job)
        runtime_rows = []
        for item in prepared_files:
            try:
                runtime = run_file(
                    llm,
                    sampling_params,
                    item,
                    item["output_file"],
                    args,
                    done=item["done"],
                    resume_diagnostics=item["resume_diagnostics"],
                    config_hash=config_hash,
                    schema_name=schema_name,
                    job_id=job_id,
                    file_index=item["file_index"],
                )
                runtime_rows.append(runtime)
            except IncompleteScoringError as exc:
                incomplete_errors.append(str(exc))
                runtime_rows.append(
                    json.loads(
                        runtime_sidecar_path(item["output_file"]).read_text(
                            encoding="utf-8"
                        )
                    )
                )
            job["files"] = runtime_rows + job["files"][len(runtime_rows) :]
            job["inference_wall_time_seconds"] = sum(
                float(row.get("inference_wall_time_seconds", 0.0))
                for row in runtime_rows
            )
            atomic_write_json(job_path, job)
        job["status"] = "incomplete" if incomplete_errors else "complete"
        if incomplete_errors:
            job["incomplete_errors"] = incomplete_errors
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = f"{type(exc).__name__}:{exc}"
        raise
    finally:
        if args.gpu_memory_monitor:
            monitor.stop()
        job["gpu_memory"] = monitor.summary()
        job["finished_at"] = utc_now()
        job["job_wall_time_seconds"] = time.perf_counter() - started_perf
        atomic_write_json(job_path, job)

    print(
        f"JOB {job['status']} {job_path} "
        f"load_s={job['model_load_wall_time_seconds']:.3f} "
        f"inference_s={job['inference_wall_time_seconds']:.3f}"
    )
    if incomplete_errors:
        raise SystemExit("Incomplete schema scoring; inspect failure/runtime sidecars")


if __name__ == "__main__":
    main()
