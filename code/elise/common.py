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


DEFAULT_RATES = (
    0.0,
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.075,
    0.10,
    0.125,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
)


def selected_count(pool_size: int, rate: float) -> int:
    if rate <= 0 or pool_size <= 0:
        return 0
    return min(pool_size, int(round(pool_size * rate)))


def rank_action_candidates(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()
    if "pair_margin" not in ranked.columns:
        ranked["pair_margin"] = ranked["pair_score"] - ranked["non_pair_score"]
    if "row_rank" not in ranked.columns:
        ranked["row_rank"] = 0
    if "pair_key" not in ranked.columns:
        ranked["pair_key"] = ranked.index.astype(str)
    if "selector_add_score" not in ranked.columns:
        ranked["selector_add_score"] = ranked["pair_score"]
    if "selector_delete_score" not in ranked.columns:
        ranked["selector_delete_score"] = ranked["non_pair_score"]
    ranked["selector_add_rank"] = -1
    ranked["selector_delete_rank"] = -1
    ranked["selector_add_pool_size"] = int((ranked["pred_pair"] == 0).sum())
    ranked["selector_delete_pool_size"] = int((ranked["pred_pair"] == 1).sum())

    add_idx = ranked.index[ranked["pred_pair"] == 0]
    if len(add_idx):
        add_order = ranked.loc[add_idx].sort_values(
            ["selector_add_score", "pair_margin", "row_rank", "pair_key"],
            ascending=[False, False, True, True],
            kind="mergesort",
        )
        ranked.loc[add_order.index, "selector_add_rank"] = range(1, len(add_order) + 1)

    delete_idx = ranked.index[ranked["pred_pair"] == 1]
    if len(delete_idx):
        delete_order = ranked.loc[delete_idx].sort_values(
            ["selector_delete_score", "pair_score", "row_rank", "pair_key"],
            ascending=[False, True, False, True],
            kind="mergesort",
        )
        ranked.loc[delete_order.index, "selector_delete_rank"] = range(1, len(delete_order) + 1)
    return ranked
