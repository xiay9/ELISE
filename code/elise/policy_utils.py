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


def text_validation_oof_scored(
    module,
    source_tables: dict[str, dict[int, pd.DataFrame]],
    candidate,
    dataset_key: str,
) -> dict[int, pd.DataFrame]:
    scored = {
        seed: source_tables[dataset_key][seed].copy()
        for seed in module.SEEDS
    }
    for action in ("add", "delete"):
        frames = []
        for seed in module.SEEDS:
            frame = module.action_frame(scored[seed], action)
            frame["_row_key"] = [
                f"{seed}:{index}" for index in frame.index
            ]
            frames.append(frame)
        pooled = pd.concat(frames, ignore_index=True)
        scores, _ = module.oof_gate_scores(pooled, candidate)
        pooled["gate_score"] = scores.to_numpy(dtype=float)
        for seed in module.SEEDS:
            selected = pooled[
                pooled["seed"].astype(int) == int(seed)
            ]
            mapping = dict(
                zip(selected["_row_key"], selected["gate_score"])
            )
            frame = module.action_frame(scored[seed], action)
            scored[seed].loc[
                frame.index, f"{action}_gate_score"
            ] = [
                mapping[f"{seed}:{index}"] for index in frame.index
            ]
    for seed in module.SEEDS:
        table = scored[seed]
        table["add_gate_score"] = table["add_gate_score"].fillna(0.0)
        table["delete_gate_score"] = table[
            "delete_gate_score"
        ].fillna(0.0)
        add_mask = table["pred_pair"].astype(int) == 0
        table.loc[add_mask, "gate_score"] = table.loc[
            add_mask, "add_gate_score"
        ]
        table.loc[~add_mask, "gate_score"] = table.loc[
            ~add_mask, "delete_gate_score"
        ]
    return scored
