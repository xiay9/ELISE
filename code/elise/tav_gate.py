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


SPLIT_SEED = 20260724


GATE_FEATURES = (
    "llm_confidence",
    "router_score",
    "router_rank_fraction",
    "pair_score",
    "pair_margin",
    "raw_pair_margin",
    "emotion_score",
    "cause_score",
    "min_gate_margin",
    "cell_entropy",
    "distance",
    "distance_abs",
    "row_rank_fraction",
    "row_score_gap_to_top",
    "row_mean_entropy",
    "evidence_count",
    "evidence_has_target",
    "evidence_has_cause",
    "evidence_span",
)


@dataclass(frozen=True)
class GateCandidate:
    name: str
    family: str


@dataclass(frozen=True)
class ActionOutcome:
    route_rate: float
    accept_rate: float
    routed: int
    correct: int
    harmful: int

    @property
    def applied(self) -> int:
        return self.correct + self.harmful


def build_gate_estimator(candidate: GateCandidate):
    if candidate.family == "linear":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                max_iter=2000,
                class_weight="balanced",
                solver="liblinear",
                random_state=SPLIT_SEED,
            ),
        )
    return make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            max_leaf_nodes=7,
            learning_rate=0.05,
            max_iter=200,
            l2_regularization=2.0,
            min_samples_leaf=50,
            class_weight="balanced",
            early_stopping=False,
            random_state=SPLIT_SEED,
        ),
    )


def evidence_payload(value: object, target: int, cause: int) -> tuple[float, ...]:
    try:
        raw = json.loads(str(value))
        turns = sorted({int(item) for item in raw}) if isinstance(raw, list) else []
    except (TypeError, ValueError, json.JSONDecodeError):
        turns = []
    return (
        float(len(turns)),
        float(target in turns),
        float(cause in turns),
        float(max(turns) - min(turns)) if turns else 0.0,
    )


def action_pool(table: pd.DataFrame, action: str) -> pd.DataFrame:
    return table[
        table["pred_pair"].astype(int) == (0 if action == "add" else 1)
    ]


def action_labels(pool: pd.DataFrame, action: str) -> np.ndarray:
    gold = pool["gold_pair"].to_numpy(dtype=int)
    return gold if action == "add" else 1 - gold


def fixed_route_scores(pool: pd.DataFrame, action: str) -> np.ndarray:
    column = "pair_score" if action == "add" else "non_pair_score"
    return pool[column].to_numpy(dtype=float)


def score_diagnostics(
    labels: np.ndarray, scores: np.ndarray
) -> dict[str, float | int]:
    prevalence = float(labels.mean()) if len(labels) else 0.0
    result: dict[str, float | int] = {
        "support": int(len(labels)),
        "positives": int(labels.sum()),
        "prevalence": prevalence,
    }
    if np.unique(labels).size == 2:
        ap = float(average_precision_score(labels, scores))
        result.update(
            {
                "average_precision": ap,
                "roc_auc": float(roc_auc_score(labels, scores)),
                "brier": float(brier_score_loss(labels, scores)),
                "ap_lift": ap / prevalence if prevalence else math.nan,
            }
        )
    else:
        result.update(
            {
                "average_precision": math.nan,
                "roc_auc": math.nan,
                "brier": math.nan,
                "ap_lift": math.nan,
            }
        )
    return result


def stable_rank(
    pool: pd.DataFrame, score_column: str
) -> pd.Series:
    ordered = pool.sort_values(
        [score_column, "dialogue_id", "emotion_turn", "cause_turn"],
        ascending=[False, True, True, True],
        kind="mergesort",
    )
    ranks = pd.Series(
        np.arange(1, len(ordered) + 1, dtype=int), index=ordered.index
    )
    return ranks.reindex(pool.index)


def gate_action_frame(table: pd.DataFrame, action: str) -> pd.DataFrame:
    frame = action_pool(table, action).copy()
    frame["router_score"] = frame[f"router_{action}_score"].astype(float)
    frame["router_rank_fraction"] = (
        frame[f"router_{action}_rank"].astype(float)
        / np.maximum(
            frame[f"router_{action}_pool_size"].astype(float), 1.0
        )
    )
    verdict = "support_pair" if action == "add" else "reject_pair"
    frame["llm_consistent"] = (
        (frame["llm_available"].astype(int) == 1)
        & (frame["llm_verdict"].astype(str) == verdict)
    )
    frame["edit_correct"] = (
        frame["gold_pair"].astype(int) == (1 if action == "add" else 0)
    ).astype(int)
    for column in GATE_FEATURES:
        values = pd.to_numeric(frame[column], errors="raise")
        if not bool(np.isfinite(values.to_numpy(dtype=float)).all()):
            raise ValueError(f"Non-finite gate feature {column}")
        frame[column] = values.astype(float)
    return frame


def oof_gate_scores(
    pooled: pd.DataFrame, candidate: GateCandidate
) -> tuple[pd.Series, dict[str, float | int]]:
    consistent = pooled[pooled["llm_consistent"]].copy()
    labels = consistent["edit_correct"].to_numpy(dtype=int)
    groups = consistent["dialogue_id"].astype(str).to_numpy()
    if np.unique(labels).size != 2:
        raise ValueError("Gate target lacks two classes")
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=SPLIT_SEED
    )
    scores = pd.Series(0.0, index=pooled.index, dtype=float)
    consistent_scores = pd.Series(index=consistent.index, dtype=float)
    for fit_positions, heldout_positions in splitter.split(
        consistent, labels, groups
    ):
        estimator = build_gate_estimator(candidate)
        estimator.fit(
            consistent.iloc[fit_positions][list(GATE_FEATURES)],
            labels[fit_positions],
        )
        consistent_scores.loc[consistent.iloc[heldout_positions].index] = (
            estimator.predict_proba(
                consistent.iloc[heldout_positions][list(GATE_FEATURES)]
            )[:, 1]
        )
    if consistent_scores.isna().any():
        raise RuntimeError("Missing OOF gate scores")
    scores.loc[consistent.index] = consistent_scores
    return scores, score_diagnostics(
        labels, consistent_scores.to_numpy(dtype=float)
    )


def counts_metrics(tp: int, fp: int, fn: int, tn: int):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    pair_f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    non_pair_precision = tn / (tn + fn) if tn + fn else 0.0
    non_pair_recall = tn / (tn + fp) if tn + fp else 0.0
    non_pair_f1 = (
        2.0
        * non_pair_precision
        * non_pair_recall
        / (non_pair_precision + non_pair_recall)
        if non_pair_precision + non_pair_recall
        else 0.0
    )
    support = tp + fp + fn + tn
    return {
        "pair_precision": precision,
        "pair_recall": recall,
        "pair_f1": pair_f1,
        "non_pair_f1": non_pair_f1,
        "macro_f1": (pair_f1 + non_pair_f1) / 2.0,
        "accuracy": (tp + tn) / support if support else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "support": support,
    }


def base_counts(table: pd.DataFrame):
    return counts_metrics(
        tp=int(
            (
                (table["gold_pair"].astype(int) == 1)
                & (table["pred_pair"].astype(int) == 1)
            ).sum()
        ),
        fp=int(
            (
                (table["gold_pair"].astype(int) == 0)
                & (table["pred_pair"].astype(int) == 1)
            ).sum()
        ),
        fn=int(
            (
                (table["gold_pair"].astype(int) == 1)
                & (table["pred_pair"].astype(int) == 0)
            ).sum()
        ),
        tn=int(
            (
                (table["gold_pair"].astype(int) == 0)
                & (table["pred_pair"].astype(int) == 0)
            ).sum()
        ),
    )


def combine_one(base, add: ActionOutcome, delete: ActionOutcome):
    metrics = counts_metrics(
        tp=int(base["tp"]) + add.correct - delete.harmful,
        fp=int(base["fp"]) + add.harmful - delete.correct,
        fn=int(base["fn"]) - add.correct + delete.harmful,
        tn=int(base["tn"]) - add.harmful + delete.correct,
    )
    return {
        **metrics,
        "gain": float(metrics["pair_f1"]) - float(base["pair_f1"]),
        "routed": add.routed + delete.routed,
        "applied": add.applied + delete.applied,
        "correct_adds": add.correct,
        "harmful_adds": add.harmful,
        "correct_deletes": delete.correct,
        "harmful_deletes": delete.harmful,
    }


def policy_key(candidate: dict[str, Any]) -> tuple:
    return (
        int(float(candidate["min_gain"]) >= -1e-15),
        float(candidate["mean_pair_f1"]),
        float(candidate["min_gain"]),
        float(candidate["mean_macro_f1"]),
        -int(candidate["routed"]),
        -int(candidate["applied"]),
        -float(candidate["delete_route_rate"]),
        -float(candidate["add_route_rate"]),
        -float(candidate["delete_accept_rate"]),
        -float(candidate["add_accept_rate"]),
    )
