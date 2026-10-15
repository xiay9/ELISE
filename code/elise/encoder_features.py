"""Primary ELISE learning primitives extracted from the experimental implementation."""
from __future__ import annotations
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import torch
from paceext import PaceExtModel
from paceext.core.validation import PaceExtMetricsCalculator


def pair_metrics(gold: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    gold = np.asarray(gold, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    tp = int(np.sum((gold == 1) & (pred == 1)))
    fp = int(np.sum((gold == 0) & (pred == 1)))
    fn = int(np.sum((gold == 1) & (pred == 0)))
    tn = int(np.sum((gold == 0) & (pred == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    pair_f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "pair_f1": pair_f1,
        "pair_precision": precision,
        "pair_recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "gold_pairs": int(np.sum(gold == 1)),
        "pred_pairs": int(np.sum(pred == 1)),
        "support": int(gold.size),
    }


def normalize_decode(payload: dict[str, object]) -> dict[str, object]:
    return {
        "pair_threshold": float(payload["pair_threshold"]),
        "gate_enable": bool(payload["gate_enable"]),
        "emo_threshold": float(payload["emo_threshold"]),
        "cause_threshold": float(payload["cause_threshold"]),
        "dist_enable": bool(payload["dist_enable"]),
        "dist_beta": float(payload["dist_beta"]),
    }


def enrich_features(
    frame: pd.DataFrame,
    *,
    decode: dict[str, object],
    model_name: str,
    seed: int,
) -> pd.DataFrame:
    frame = frame.copy()
    pair_score = frame["pair_score"].to_numpy(dtype=np.float64)
    eps = 1e-12
    clipped = np.clip(pair_score, eps, 1.0 - eps)

    frame["non_pair_score"] = 1.0 - frame["pair_score"]
    frame["pair_margin"] = frame["pair_score"] - frame["non_pair_score"]
    frame["max_prob"] = np.maximum(frame["pair_score"], frame["non_pair_score"])
    frame["margin_top2"] = np.abs(frame["pair_margin"])
    frame["cell_entropy"] = -(
        clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped)
    )
    frame["pair_margin_to_threshold"] = (
        frame["pair_score"] - float(decode["pair_threshold"])
    )
    frame["emotion_margin_to_gate"] = (
        frame["emotion_score"] - float(decode["emo_threshold"])
    )
    frame["cause_margin_to_gate"] = (
        frame["cause_score"] - float(decode["cause_threshold"])
    )
    frame["min_gate_margin"] = frame[
        ["emotion_margin_to_gate", "cause_margin_to_gate"]
    ].min(axis=1)
    frame["gate_score_product"] = (
        frame["emotion_score"] * frame["cause_score"]
    )
    frame["raw_non_pair_score"] = 1.0 - frame["raw_pair_score"]
    frame["raw_pair_margin"] = (
        frame["raw_pair_score"] - frame["raw_non_pair_score"]
    )
    frame["raw_max_prob"] = np.maximum(
        frame["raw_pair_score"], frame["raw_non_pair_score"]
    )
    frame["raw_margin_top2"] = np.abs(frame["raw_pair_margin"])

    row_keys = ["dialogue_id", "emotion_turn"]
    row_group = frame.groupby(row_keys, sort=False)
    frame["row_size"] = row_group["pair_score"].transform("size").astype(int)
    frame["row_pred_count"] = (
        row_group["pred_pair"].transform("sum").astype(int)
    )
    frame["row_rank"] = (
        row_group["pair_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    frame["row_rank_fraction"] = frame["row_rank"] / frame["row_size"]
    frame["row_top_score"] = row_group["pair_score"].transform("max")
    frame["row_mean_score"] = row_group["pair_score"].transform("mean")
    frame["row_std_score"] = (
        row_group["pair_score"].transform("std").fillna(0.0)
    )
    frame["row_score_gap_to_top"] = (
        frame["row_top_score"] - frame["pair_score"]
    )
    frame["row_mean_entropy"] = row_group["cell_entropy"].transform("mean")

    dialogue_group = frame.groupby("dialogue_id", sort=False)
    frame["dialogue_candidate_count"] = (
        dialogue_group["pair_score"].transform("size").astype(int)
    )
    frame["dialogue_pred_count"] = (
        dialogue_group["pred_pair"].transform("sum").astype(int)
    )
    frame["dialogue_mean_score"] = dialogue_group["pair_score"].transform("mean")

    frame["gold"] = frame["gold_pair"].map({0: "non_pair", 1: "pair"})
    frame["pred"] = frame["pred_pair"].map({0: "non_pair", 1: "pair"})
    frame["gold_id"] = 1 - frame["gold_pair"]
    frame["pred_id"] = 1 - frame["pred_pair"]
    frame["target_turn"] = frame["emotion_turn"]
    frame["candidate_turn"] = frame["cause_turn"]
    frame["model_name"] = model_name
    frame["model_family"] = "paceext_rpcl"
    frame["seed"] = int(seed)
    frame["modality"] = "t"
    frame["candidate_scope"] = "lower_triangle"
    frame["run_name"] = (
        frame["dataset"].str.lower()
        + f"_{model_name}_t_seed{seed}_lower_triangle"
    )
    frame["pair_key"] = (
        frame["dataset"].astype(str)
        + "::"
        + frame["split"].astype(str)
        + "::"
        + frame["dialogue_id"].astype(str)
        + "::t"
        + frame["emotion_turn"].astype(str)
        + "::c"
        + frame["cause_turn"].astype(str)
    )
    frame["scored_by_model"] = 1
    frame["score_label_space"] = "pair|non_pair"
    frame["source_score_column"] = "distance_adjusted_pair_score"
    frame["prob_pair"] = frame["pair_score"]
    frame["prob_non_pair"] = frame["non_pair_score"]
    frame["distance_abs"] = frame["distance"].abs()

    preferred = [
        "dataset",
        "split",
        "dialogue_id",
        "doc_id",
        "emotion_turn",
        "cause_turn",
        "target_turn",
        "candidate_turn",
        "gold",
        "pred",
        "gold_id",
        "pred_id",
        "gold_pair",
        "pred_pair",
        "pair_score",
        "non_pair_score",
        "pair_margin",
        "max_prob",
        "margin_top2",
        "raw_pair_score",
        "raw_non_pair_score",
        "raw_pair_margin",
        "raw_max_prob",
        "raw_margin_top2",
        "emotion_score",
        "cause_score",
        "pair_margin_to_threshold",
        "emotion_margin_to_gate",
        "cause_margin_to_gate",
        "min_gate_margin",
        "gate_score_product",
        "cell_entropy",
        "distance",
        "distance_abs",
        "distance_prior",
        "row_rank",
        "row_rank_fraction",
        "row_size",
        "row_pred_count",
        "row_top_score",
        "row_mean_score",
        "row_std_score",
        "row_score_gap_to_top",
        "row_mean_entropy",
        "dialogue_candidate_count",
        "dialogue_pred_count",
        "dialogue_mean_score",
        "pair_threshold_pass",
        "gate_pass",
        "logit_pair",
        "logit_non_pair",
        "raw_logit_margin",
        "model_name",
        "model_family",
        "seed",
        "modality",
        "candidate_scope",
        "run_name",
        "pair_key",
        "scored_by_model",
        "score_label_space",
        "source_score_column",
        "prob_pair",
        "prob_non_pair",
    ]
    missing = sorted(set(preferred) - set(frame.columns))
    if missing:
        raise AssertionError(f"Missing enriched columns: {missing}")
    return frame[preferred]


def collect_split(
    *,
    model: PaceExtModel,
    loader,
    config,
    dataset: str,
    split: str,
    decode: dict[str, object],
    model_name: str,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    model.eval()
    records: list[dict[str, object]] = []
    with torch.inference_mode():
        for data in loader:
            device_data = {
                key: value.to(config.device) if isinstance(value, torch.Tensor) else value
                for key, value in data.items()
            }
            _, output = model(**device_data)
            pair_logits, emotion_logits, cause_logits, masks = output[:4]
            pair_probs = torch.softmax(pair_logits, dim=-1)[..., 1].detach().cpu().numpy()
            pair_logits_np = pair_logits.detach().cpu().numpy()
            emotion_probs = torch.softmax(emotion_logits, dim=-1)
            neutral_id = int(getattr(config, "neutral_label_id", 0))
            emotion_scores = (
                1.0 - emotion_probs[..., neutral_id]
            ).detach().cpu().numpy()
            cause_scores = (
                torch.softmax(cause_logits, dim=-1)[..., 1]
                .detach()
                .cpu()
                .numpy()
            )
            masks_np = masks.detach().cpu().numpy().astype(bool)
            pairs_cpu = data["pairs"].detach().cpu().numpy()
            pair_nums = data["pair_nums"].detach().cpu().numpy()
            utterance_nums = data["utterance_nums"].detach().cpu().numpy()

            for batch_index, doc_id in enumerate(data["doc_ids"]):
                utterance_num = int(utterance_nums[batch_index])
                gold_matrix = np.zeros(
                    (utterance_num, utterance_num), dtype=np.int64
                )
                pair_num = int(pair_nums[batch_index])
                for row_index, column_index in pairs_cpu[batch_index, :pair_num]:
                    if row_index >= 0 and column_index >= 0:
                        gold_matrix[int(row_index), int(column_index)] = 1

                valid = masks_np[
                    batch_index, :utterance_num, :utterance_num
                ]
                rows, columns = np.where(valid)
                for row_index, column_index in zip(rows.tolist(), columns.tolist()):
                    records.append(
                        {
                            "dataset": dataset,
                            "split": split,
                            "dialogue_id": int(doc_id),
                            "doc_id": int(doc_id),
                            "emotion_turn": int(row_index + 1),
                            "cause_turn": int(column_index + 1),
                            "gold_pair": int(gold_matrix[row_index, column_index]),
                            "raw_pair_score": float(
                                pair_probs[
                                    batch_index, row_index, column_index
                                ]
                            ),
                            "emotion_score": float(
                                emotion_scores[batch_index, row_index]
                            ),
                            "cause_score": float(
                                cause_scores[batch_index, column_index]
                            ),
                            "distance": int(row_index - column_index),
                            "logit_non_pair": float(
                                pair_logits_np[
                                    batch_index, row_index, column_index, 0
                                ]
                            ),
                            "logit_pair": float(
                                pair_logits_np[
                                    batch_index, row_index, column_index, 1
                                ]
                            ),
                        }
                    )

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise RuntimeError(f"No pair cells exported for {dataset}/{split}/seed{seed}")
    if bool((frame["cause_turn"] > frame["emotion_turn"]).any()):
        raise AssertionError("lower_triangle export contains future-cause cells")
    if frame.duplicated(
        ["dataset", "split", "dialogue_id", "emotion_turn", "cause_turn"]
    ).any():
        raise AssertionError("Pair keys are not unique")

    metric_calculator = PaceExtMetricsCalculator(
        neutral_id=int(getattr(config, "neutral_label_id", 0)),
        dist_prior=getattr(config, "dist_prior", None),
    )
    raw_scores = frame["raw_pair_score"].to_numpy(dtype=np.float32)
    deltas = frame["distance"].to_numpy(dtype=np.int64)
    if bool(decode["dist_enable"]) and metric_calculator.dist_prior is not None:
        pair_scores = metric_calculator._apply_distance_prior(
            raw_scores,
            deltas,
            float(decode["dist_beta"]),
            metric_calculator.dist_prior,
        )
    else:
        pair_scores = raw_scores
    frame["pair_score"] = pair_scores.astype(np.float64)
    if metric_calculator.dist_prior is None:
        frame["distance_prior"] = 1.0
    else:
        clipped_delta = np.clip(
            deltas, 0, metric_calculator.dist_prior.size - 1
        )
        frame["distance_prior"] = metric_calculator.dist_prior[
            clipped_delta
        ].astype(np.float64)

    frame["pair_threshold_pass"] = (
        frame["pair_score"] >= float(decode["pair_threshold"])
    ).astype(int)
    if bool(decode["gate_enable"]):
        gate_pass = (
            (frame["emotion_score"] >= float(decode["emo_threshold"]))
            & (frame["cause_score"] >= float(decode["cause_threshold"]))
        )
    else:
        gate_pass = pd.Series(True, index=frame.index)
    frame["gate_pass"] = gate_pass.astype(int)
    frame["pred_pair"] = (
        (frame["pair_threshold_pass"] == 1) & (frame["gate_pass"] == 1)
    ).astype(int)
    frame["raw_logit_margin"] = frame["logit_pair"] - frame["logit_non_pair"]

    metrics = pair_metrics(
        frame["gold_pair"].to_numpy(), frame["pred_pair"].to_numpy()
    )
    metrics["pair_auprc_raw"] = float(
        average_precision_score(
            frame["gold_pair"].to_numpy(),
            frame["raw_pair_score"].to_numpy(),
        )
    )
    frame = enrich_features(
        frame,
        decode=decode,
        model_name=model_name,
        seed=seed,
    )
    frame = frame.sort_values(
        ["dialogue_id", "emotion_turn", "cause_turn"], kind="mergesort"
    ).reset_index(drop=True)
    return frame, metrics
