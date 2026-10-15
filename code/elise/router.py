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


SCORE_FEATURES = (
    "pair_score",
    "non_pair_score",
    "pair_margin",
    "max_prob",
    "margin_top2",
)


STRUCTURE_FEATURES = (
    "distance",
    "row_rank",
    "row_size",
    "row_pred_count",
    "row_rank_fraction",
)


V1_FEATURES = (
    "pair_score",
    "non_pair_score",
    "pair_margin",
    "max_prob",
    "margin_top2",
    "distance",
    "row_rank",
    "row_size",
    "row_pred_count",
)


PACEEXT_FEATURES = V1_FEATURES + (
    "raw_pair_score",
    "raw_pair_margin",
    "raw_max_prob",
    "raw_margin_top2",
    "raw_logit_margin",
    "emotion_score",
    "cause_score",
    "pair_margin_to_threshold",
    "emotion_margin_to_gate",
    "cause_margin_to_gate",
    "min_gate_margin",
    "gate_score_product",
    "cell_entropy",
    "distance_prior",
    "row_rank_fraction",
    "row_pred_fraction",
    "row_top_score",
    "row_mean_score",
    "row_std_score",
    "row_score_gap_to_top",
    "row_mean_entropy",
    "dialogue_pred_fraction",
    "pair_threshold_pass",
    "gate_pass",
)


FEATURE_SETS = {
    "score": SCORE_FEATURES,
    "structure": STRUCTURE_FEATURES,
    "v1": V1_FEATURES,
    "paceext": PACEEXT_FEATURES,
}


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    feature_set: str

    @property
    def features(self) -> tuple[str, ...]:
        return FEATURE_SETS[self.feature_set]


def build_estimator(candidate: Candidate):
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
    if candidate.family == "histgb":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                max_leaf_nodes=7,
                learning_rate=0.05,
                max_iter=200,
                l2_regularization=1.0,
                min_samples_leaf=20,
                class_weight="balanced",
                early_stopping=False,
                random_state=SPLIT_SEED,
            ),
        )
    raise ValueError(candidate.family)


def action_pool(table: pd.DataFrame, action: str) -> pd.DataFrame:
    return table[table["pred_pair"] == (0 if action == "add" else 1)]


def action_labels(pool: pd.DataFrame, action: str) -> np.ndarray:
    gold = pool["gold_pair"].to_numpy(dtype=int)
    return gold if action == "add" else 1 - gold


def fallback_scores(pool: pd.DataFrame, action: str) -> np.ndarray:
    column = "pair_score" if action == "add" else "non_pair_score"
    return pool[column].to_numpy(dtype=float)


def oof_scores(
    pool: pd.DataFrame,
    action: str,
    candidate: Candidate,
) -> tuple[pd.Series, dict[str, float | int]]:
    labels = action_labels(pool, action)
    groups = pool["dialogue_id"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SPLIT_SEED,
    )
    result = pd.Series(index=pool.index, dtype=float)
    fallback_folds = 0
    for fit_positions, heldout_positions in splitter.split(pool, labels, groups):
        fit_labels = labels[fit_positions]
        heldout = pool.iloc[heldout_positions]
        if np.unique(fit_labels).size < 2:
            fallback_folds += 1
            scores = fallback_scores(heldout, action)
        else:
            estimator = build_estimator(candidate)
            estimator.fit(
                pool.iloc[fit_positions][list(candidate.features)],
                fit_labels,
            )
            scores = estimator.predict_proba(
                heldout[list(candidate.features)]
            )[:, 1]
        result.loc[heldout.index] = scores
    if result.isna().any():
        raise RuntimeError(f"Missing OOF scores for {candidate.name}/{action}")

    prevalence = float(labels.mean()) if len(labels) else 0.0
    diagnostics: dict[str, float | int] = {
        "support": int(len(labels)),
        "positives": int(labels.sum()),
        "prevalence": prevalence,
        "fallback_folds": int(fallback_folds),
    }
    if np.unique(labels).size == 2:
        diagnostics.update(
            {
                "average_precision": float(
                    average_precision_score(labels, result.to_numpy())
                ),
                "roc_auc": float(roc_auc_score(labels, result.to_numpy())),
                "brier": float(
                    brier_score_loss(labels, result.to_numpy())
                ),
                "ap_lift": float(
                    average_precision_score(labels, result.to_numpy())
                    / prevalence
                )
                if prevalence > 0
                else math.nan,
            }
        )
    else:
        diagnostics.update(
            {
                "average_precision": math.nan,
                "roc_auc": math.nan,
                "brier": math.nan,
                "ap_lift": math.nan,
            }
        )
    return result, diagnostics
