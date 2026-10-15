#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import run_llm_verifier_vllm as runner


def synthetic_request() -> dict:
    messages = [
        {"role": "system", "content": "Return strict JSON."},
        {"role": "user", "content": "Does turn 1 explain turn 2?"},
    ]
    return {
        "request_id": "elise-synthetic-1",
        "dataset": "SYNTHETIC",
        "dataset_key": "synthetic",
        "split": "valid",
        "run_name": "synthetic_valid_blind_cache",
        "model": "blind_semantic_verifier",
        "modality": "text",
        "seed": 42,
        "dialogue_id": "d1",
        "emotion_turn": 2,
        "cause_turn": 1,
        "pair_key": "SYNTHETIC::valid::d1::t2::c1",
        "prompt_version": "synthetic-no-emotion-v1",
        "prompt_variant": "synthetic-no-emotion-v1",
        "emotion_label_mode": "hidden",
        "candidate_source": "all_pairs",
        "candidate_scope": "full_matrix",
        "condition": "blind",
        "cache_key": "SYNTHETIC::valid::d1::t2::c1",
        "response_schema": runner.COMPACT_VERIFICATION_JSON_SCHEMA,
        "messages_sha256": runner.messages_sha256(messages),
        "messages": messages,
    }


def synthetic_args() -> argparse.Namespace:
    return argparse.Namespace(
        model="synthetic-model",
        tokenizer=None,
        trust_remote_code=True,
        dtype="auto",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
        max_model_len=8192,
        max_num_seqs=64,
        language_model_only=True,
        enforce_eager=False,
        temperature=0.0,
        top_p=1.0,
        max_tokens=64,
        seed=42,
        enable_thinking=False,
        disable_structured_outputs=False,
        structured_output_schema="compact_verification",
        strict_request_contract=True,
        batch_size=8,
        max_schema_retries=2,
    )


class RunnerContractTest(unittest.TestCase):
    def test_default_schema_is_compact_without_importing_vllm(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            ["runner", "--input", "in.jsonl", "--output", "out.jsonl", "--model", "local-snapshot"],
        ):
            args = runner.parse_args()
        self.assertEqual(args.structured_output_schema, "compact_verification")
        self.assertNotIn("vllm", sys.modules)

    def test_strict_request_and_response_contract(self) -> None:
        request = synthetic_request()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "valid_blind_requests.jsonl"
            path.write_text(json.dumps(request) + "\n", encoding="utf-8")
            prepared = runner.prepare_request_file(
                path,
                schema_name="compact_verification",
                strict=True,
            )
        self.assertEqual(prepared["total_requests"], 1)
        payload, error = runner.strict_response_error(
            json.dumps(
                {
                    "verdict": "support_pair",
                    "confidence": 0.9,
                    "evidence_turns": [1],
                }
            ),
            "compact_verification",
        )
        self.assertFalse(error)
        self.assertEqual(payload["verdict"], "support_pair")
        _, error = runner.strict_response_error(
            json.dumps({"verdict": "support_pair", "confidence": 0.9}),
            "compact_verification",
        )
        self.assertIn("schema_keys_mismatch", error)

    def test_cli_preflight_only_stays_cpu_and_writes_guards(self) -> None:
        request = synthetic_request()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "valid_blind_requests.jsonl"
            output_path = root / "valid_blind_results.jsonl"
            input_path.write_text(json.dumps(request) + "\n", encoding="utf-8")
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = ""
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(runner.__file__).resolve()),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--model",
                    "synthetic-model",
                    "--preflight-only",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=environment,
            )
            self.assertIn("PREFLIGHT_ONLY", completed.stdout)
            self.assertTrue(runner.output_contract_path(output_path).exists())
            job_sidecars = list((root / "_run_metadata").glob("*.json"))
            self.assertEqual(len(job_sidecars), 1)
            job = json.loads(job_sidecars[0].read_text(encoding="utf-8"))
            self.assertEqual(job["status"], "preflight_only_complete")
            self.assertEqual(job["response_schema_name"], "compact_verification")
            self.assertFalse(output_path.exists())

    def test_result_resume_guard_and_invalid_quarantine(self) -> None:
        request = synthetic_request()
        args = synthetic_args()
        config = runner.inference_config(args)
        config_hash = runner.sha256_json(config)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "valid_blind_requests.jsonl"
            output_path = root / "valid_blind_results.jsonl"
            input_path.write_text(json.dumps(request) + "\n", encoding="utf-8")
            prepared = runner.prepare_request_file(
                input_path,
                schema_name="compact_verification",
                strict=True,
            )
            runner.ensure_output_contract(
                output_path,
                prepared,
                config=config,
                config_hash=config_hash,
                schema_name="compact_verification",
            )
            normalized_request = prepared["requests"][0]
            fake_output = SimpleNamespace(
                prompt_token_ids=[1, 2, 3],
                outputs=[
                    SimpleNamespace(
                        text=json.dumps(
                            {
                                "verdict": "support_pair",
                                "confidence": 0.9,
                                "evidence_turns": [1],
                            }
                        ),
                        token_ids=[4, 5],
                    )
                ],
            )
            record = runner.result_record(
                normalized_request,
                fake_output,
                args,
                prepared=prepared,
                config_hash=config_hash,
                schema_name="compact_verification",
                job_id="job-synthetic",
                batch_id="job-synthetic:f1:b1:a0",
                batch_attempt=0,
                batch_wall_time_seconds=0.1,
                batch_size=1,
            )
            self.assertEqual(record["scored_by_model"], 1)
            self.assertEqual(record["schema_valid"], 1)
            self.assertEqual(record["messages_sha256"], request["messages_sha256"])
            output_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            done, diagnostics = runner.sanitize_existing_results(
                output_path,
                prepared,
                config_hash=config_hash,
                schema_name="compact_verification",
                job_id="resume-valid",
            )
            self.assertEqual(done, {request["request_id"]})
            self.assertEqual(diagnostics["valid_resume_rows"], 1)

            invalid = dict(record)
            invalid["scored_by_model"] = 0
            output_path.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
            done, diagnostics = runner.sanitize_existing_results(
                output_path,
                prepared,
                config_hash=config_hash,
                schema_name="compact_verification",
                job_id="resume-invalid",
            )
            self.assertEqual(done, set())
            self.assertEqual(diagnostics["quarantined_rows"], 1)
            self.assertEqual(output_path.read_text(encoding="utf-8"), "")
            quarantine = Path(str(output_path) + ".resume_rejected.resume-invalid.jsonl")
            self.assertTrue(quarantine.exists())

            with self.assertRaises(runner.ContractError):
                runner.ensure_output_contract(
                    output_path,
                    prepared,
                    config={**config, "seed": 123},
                    config_hash="different-config-hash",
                    schema_name="compact_verification",
                )


if __name__ == "__main__":
    unittest.main()
