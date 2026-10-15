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

from common import DEFAULT_RATES, selected_count
POINTWISE_ROUTE_RATES = tuple(dict.fromkeys((*DEFAULT_RATES, 1.0)))
ACCEPT_RATES = POINTWISE_ROUTE_RATES
ACTIONS = ("add", "delete")
SEEDS = (42, 345, 456)


def finite_threshold(scores: Iterable[float], rate: float) -> float:
    """Return a finite descending-score cutoff for a target fraction."""
    values = np.asarray(list(scores), dtype=float)
    if not len(values):
        return 0.0
    if not bool(np.isfinite(values).all()):
        raise ValueError("Threshold calibration received non-finite scores")
    count = selected_count(len(values), float(rate))
    if count <= 0:
        return float(np.nextafter(values.max(), math.inf))
    if count >= len(values):
        return float(np.nextafter(values.min(), -math.inf))
    ordered = np.sort(values)[::-1]
    return float(ordered[count - 1])


def upper_tail_ecdf(
    validation_scores: Iterable[float], query_scores: Iterable[float]
) -> np.ndarray:
    """Validation survival percentile with smaller values for better scores."""
    validation = np.sort(np.asarray(list(validation_scores), dtype=float))
    query = np.asarray(list(query_scores), dtype=float)
    if not len(validation):
        raise ValueError("Cannot form an ECDF from an empty validation pool")
    if not bool(np.isfinite(validation).all() and np.isfinite(query).all()):
        raise ValueError("ECDF received non-finite scores")
    greater_or_equal = len(validation) - np.searchsorted(
        validation, query, side="left"
    )
    return greater_or_equal.astype(float) / float(len(validation))


def action_frame(module, table: pd.DataFrame, action: str) -> pd.DataFrame:
    if hasattr(module, "gate_action_frame"):
        frame = module.gate_action_frame(table, action)
        if "gate_score" in table:
            frame["gate_score"] = table.loc[
                frame.index, "gate_score"
            ].astype(float)
        return frame
    return module.action_frame(table, action)


def raw_router_column(chain: str, action: str) -> str:
    prefix = "selector" if chain == "text" else "router"
    return f"{prefix}_{action}_score"


def calibration_outcomes(
    module,
    table: pd.DataFrame,
    action: str,
) -> tuple[dict[tuple[float, float], Any], dict[tuple[float, float], dict[str, Any]]]:
    frame = action_frame(module, table, action)
    route_scores = frame["selector_score" if "selector_score" in frame else "router_score"]
    output = {}
    details = {}
    for route_rate in POINTWISE_ROUTE_RATES:
        route_threshold = finite_threshold(route_scores, route_rate)
        routed = frame[route_scores >= route_threshold]
        consistent = routed[routed["llm_consistent"]]
        for accept_rate in ACCEPT_RATES:
            gate_threshold = finite_threshold(
                consistent["gate_score"], accept_rate
            )
            accepted = consistent[
                consistent["gate_score"].astype(float) >= gate_threshold
            ]
            correct = int(accepted["edit_correct"].sum())
            key = (float(route_rate), float(accept_rate))
            output[key] = module.ActionOutcome(
                route_rate=float(route_rate),
                accept_rate=float(accept_rate),
                routed=int(len(routed)),
                correct=correct,
                harmful=int(len(accepted) - correct),
            )
            details[key] = {
                "route_threshold": route_threshold,
                "gate_threshold": gate_threshold,
                "pool": int(len(frame)),
                "routed": int(len(routed)),
                "consistent_routed": int(len(consistent)),
                "accepted": int(len(accepted)),
                "realized_route_rate": (
                    float(len(routed) / len(frame)) if len(frame) else 0.0
                ),
                "realized_accept_rate": (
                    float(len(accepted) / len(consistent))
                    if len(consistent)
                    else 0.0
                ),
            }
    return output, details


def search_pointwise_policy(
    module, tables: dict[int, pd.DataFrame]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bases = {seed: module.base_counts(table) for seed, table in tables.items()}
    outcomes = {}
    details = {}
    for seed, table in tables.items():
        outcomes[seed] = {}
        details[seed] = {}
        for action in ACTIONS:
            outcomes[seed][action], details[seed][action] = (
                calibration_outcomes(module, table, action)
            )
    best = None
    add_keys = next(iter(outcomes.values()))["add"].keys()
    delete_keys = next(iter(outcomes.values()))["delete"].keys()
    support = sum(int(value["support"]) for value in bases.values())
    for delete_key in delete_keys:
        for add_key in add_keys:
            run_rows = []
            for seed in SEEDS:
                run_rows.append(
                    {
                        "seed": seed,
                        **module.combine_one(
                            bases[seed],
                            outcomes[seed]["add"][add_key],
                            outcomes[seed]["delete"][delete_key],
                        ),
                    }
                )
            candidate = {
                "delete_route_rate": float(delete_key[0]),
                "delete_accept_rate": float(delete_key[1]),
                "add_route_rate": float(add_key[0]),
                "add_accept_rate": float(add_key[1]),
                "mean_pair_f1": float(
                    np.mean([row["pair_f1"] for row in run_rows])
                ),
                "mean_macro_f1": float(
                    np.mean([row["macro_f1"] for row in run_rows])
                ),
                "mean_gain": float(
                    np.mean([row["gain"] for row in run_rows])
                ),
                "min_gain": float(
                    np.min([row["gain"] for row in run_rows])
                ),
                "routed": int(sum(row["routed"] for row in run_rows)),
                "applied": int(sum(row["applied"] for row in run_rows)),
                "routed_fraction": float(
                    sum(row["routed"] for row in run_rows) / support
                ),
                "run_rows": run_rows,
            }
            if best is None or module.policy_key(candidate) > module.policy_key(best):
                best = candidate
    if best is None:
        raise RuntimeError("Pointwise policy search produced no candidate")
    threshold_rows = []
    for seed in SEEDS:
        for action in ACTIONS:
            key = (
                float(best[f"{action}_route_rate"]),
                float(best[f"{action}_accept_rate"]),
            )
            threshold_rows.append(
                {
                    "seed": seed,
                    "action": action,
                    "target_route_rate": key[0],
                    "target_accept_rate": key[1],
                    **details[seed][action][key],
                }
            )
    return best, threshold_rows


def threshold_for(
    thresholds: pd.DataFrame,
    chain: str,
    dataset_key: str,
    seed: int,
    action: str,
):
    selected = thresholds[
        (thresholds["chain"] == chain)
        & (thresholds["dataset_key"] == dataset_key)
        & (thresholds["seed"].astype(int) == int(seed))
        & (thresholds["action"] == action)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one threshold row for "
            f"{chain}/{dataset_key}/{seed}/{action}"
        )
    return next(selected.itertuples(index=False))


def apply_pointwise(
    *,
    chain: str,
    module,
    table: pd.DataFrame,
    dataset_key: str,
    seed: int,
    thresholds: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    pred = table["pred_pair"].astype(int).copy()
    routed = pd.Series(False, index=table.index, dtype=bool)
    accepted = pd.Series(False, index=table.index, dtype=bool)
    for action in ACTIONS:
        frozen = threshold_for(
            thresholds, chain, dataset_key, seed, action
        )
        frame = action_frame(module, table, action)
        score = raw_router_column(chain, action)
        action_routed = frame[
            table.loc[frame.index, score].astype(float)
            >= float(frozen.route_threshold)
        ]
        routed.loc[action_routed.index] = True
        consistent = action_routed[action_routed["llm_consistent"]]
        gate = table.loc[
            consistent.index, "pointwise_gate_score"
        ].astype(float)
        indices = consistent.index[gate >= float(frozen.gate_threshold)]
        accepted.loc[indices] = True
        pred.loc[indices] = 1 if action == "add" else 0
    return pred, routed, accepted
