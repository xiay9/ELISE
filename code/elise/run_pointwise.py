"""Fit validation-only ELISE policies, route candidates, or apply frozen edits."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import common
import pointwise
import policy_utils
import router
import tav_gate
import text_gate
from data_io import (DATA_ROOT, attach_responses, complete_gold_metrics,
                     dialogue_index, read_pairs, read_responses, sha256)

SEEDS = (42, 345, 456)
ACTIONS = ('add', 'delete')
ROUTER = router.Candidate('hgb_paceext', 'histgb', 'paceext')
text_gate.SEEDS = tav_gate.SEEDS = SEEDS


def gate_module(chain):
    return text_gate if chain == 'text' else tav_gate


def table_path(root, dataset, chain, seed, split):
    modality = 't' if chain == 'text' else 'tav'
    return root / dataset.lower() / f'rpcl_{modality}_seed{seed}_{split}.csv'


def rank_routes(table, chain):
    if chain == 'text':
        return common.rank_action_candidates(table)
    table = table.copy()
    for action in ACTIONS:
        pool = tav_gate.action_pool(table, action)
        score_column = f'router_{action}_score'
        table.loc[pool.index, score_column] = tav_gate.fixed_route_scores(pool, action)
        table.loc[pool.index, f'router_{action}_rank'] = tav_gate.stable_rank(table.loc[pool.index], score_column)
        table.loc[pool.index, f'router_{action}_pool_size'] = len(pool)
    return table


def fit_router(table, chain):
    table = table.copy()
    estimators = {}
    if chain == 'text':
        for action in ACTIONS:
            pool = router.action_pool(table, action)
            if pool.empty:
                raise ValueError(f'Validation {action} pool is empty')
            scores, _ = router.oof_scores(pool, action, ROUTER)
            table.loc[pool.index, f'selector_{action}_score'] = scores
            labels = router.action_labels(pool, action)
            estimator = None
            if np.unique(labels).size == 2:
                estimator = router.build_estimator(ROUTER)
                estimator.fit(pool[list(ROUTER.features)], labels)
            estimators[action] = estimator
    return rank_routes(table, chain), estimators


def score_router(table, chain, estimators):
    table = table.copy()
    if chain == 'text':
        for action in ACTIONS:
            pool = router.action_pool(table, action)
            if pool.empty:
                table[f'selector_{action}_score'] = table.get(f'selector_{action}_score', 0.0)
                continue
            estimator = estimators[action]
            scores = (router.fallback_scores(pool, action) if estimator is None else
                      estimator.predict_proba(pool[list(ROUTER.features)])[:, 1])
            table.loc[pool.index, f'selector_{action}_score'] = scores
    return rank_routes(table, chain)


def fit_gate_oof(tables, module, candidate):
    output = {seed: table.copy() for seed, table in tables.items()}
    for action in ACTIONS:
        frames = []
        for seed in SEEDS:
            frame = pointwise.action_frame(module, output[seed], action)
            frame['_row_key'] = [f'{seed}:{index}' for index in frame.index]
            frames.append(frame)
        pooled = pd.concat(frames, ignore_index=True)
        scores, _ = module.oof_gate_scores(pooled, candidate)
        lookup = dict(zip(pooled['_row_key'], scores.to_numpy(dtype=float)))
        for seed in SEEDS:
            frame = pointwise.action_frame(module, output[seed], action)
            output[seed].loc[frame.index, f'{action}_gate_score'] = [lookup[f'{seed}:{i}'] for i in frame.index]
    for seed, table in output.items():
        add = table.pred_pair.astype(int) == 0
        table['gate_score'] = np.where(add, table.add_gate_score, table.delete_gate_score)
    return output


def fit(args):
    if args.output.exists():
        raise FileExistsError('Use a new output directory for each validation fit')
    module = gate_module(args.chain)
    candidate = module.GateCandidate('linear_gate', 'linear')
    records = dialogue_index(args.dataset, 'valid', args.data_root)
    responses = read_responses(args.responses, args.dataset, 'valid', records)
    tables, router_models, references, inputs = {}, {}, {}, []
    for seed in SEEDS:
        path = table_path(args.pair_root, args.dataset, args.chain, seed, 'valid')
        raw = read_pairs(path, args.dataset, 'valid', seed, args.chain, fitting=True)
        # Every fit function below consumes validation labels only.
        routed, router_models[seed] = fit_router(raw, args.chain)
        tables[seed] = attach_responses(routed, responses, records, module)
        references[seed] = {}
        for action in ACTIONS:
            frame = pointwise.action_frame(module, tables[seed], action)
            column = 'selector_score' if args.chain == 'text' else 'router_score'
            references[seed][action] = frame[column].to_numpy(dtype=float)
        inputs.append({'file': path.name, 'sha256': sha256(path)})
    scored = fit_gate_oof(tables, module, candidate)
    policy, thresholds = pointwise.search_pointwise_policy(module, scored)
    gates = {}
    for action in ACTIONS:
        pooled = pd.concat([pointwise.action_frame(module, tables[s], action) for s in SEEDS], ignore_index=True)
        consistent = pooled[pooled.llm_consistent]
        estimator = (module.build_estimator(candidate) if args.chain == 'text' else
                     module.build_gate_estimator(candidate))
        estimator.fit(consistent[list(module.GATE_FEATURES)], consistent.edit_correct.to_numpy(dtype=int))
        gates[action] = estimator
    for row in thresholds:
        row.update(chain=args.chain, dataset_key=args.dataset.lower())
    payload = {'format_version': 1, 'dataset': args.dataset, 'chain': args.chain,
               'router_models': router_models, 'gates': gates,
               'validation_route_scores': references, 'thresholds': thresholds}
    args.output.mkdir(parents=True)
    model_path = args.output / 'model.pkl'
    with model_path.open('xb') as handle:
        pickle.dump(payload, handle, protocol=5)
    (args.output / 'validation.json').write_text(json.dumps({
        'format_version': 1, 'dataset': args.dataset, 'chain': args.chain,
        'seeds': list(SEEDS), 'fit_split': 'valid', 'inputs': inputs,
        'semantic_responses_sha256': sha256(args.responses),
        'model_sha256': sha256(model_path), 'policy': policy,
        'thresholds': thresholds, 'router_features': list(router.PACEEXT_FEATURES),
        'gate_features': list(module.GATE_FEATURES),
    }, indent=2) + '\n')
    print(f'Frozen validation policy and model: {args.output}')


def load_bundle(path):
    manifest = json.loads((path / 'validation.json').read_text())
    if manifest['fit_split'] != 'valid' or sha256(path / 'model.pkl') != manifest['model_sha256']:
        raise ValueError('Frozen validation model identity mismatch')
    with (path / 'model.pkl').open('rb') as handle:
        bundle = pickle.load(handle)
    if bundle['format_version'] != 1:
        raise ValueError('Unsupported bundle format')
    return bundle


def route_mask(table, chain, seed, dataset, thresholds):
    routed = pd.Series(False, index=table.index)
    for action in ACTIONS:
        frozen = pointwise.threshold_for(thresholds, chain, dataset.lower(), seed, action)
        pool = router.action_pool(table, action)
        routed.loc[pool.index] = pool[pointwise.raw_router_column(chain, action)] >= frozen.route_threshold
    return routed


def predict_table(table, bundle, seed, responses, records):
    chain, dataset = bundle['chain'], bundle['dataset']
    module = gate_module(chain)
    table = score_router(table, chain, bundle['router_models'][seed])
    table = attach_responses(table, responses, records, module)
    for action in ACTIONS:
        frame = pointwise.action_frame(module, table, action)
        if frame.empty:
            continue
        prefix = 'selector' if chain == 'text' else 'router'
        frame[f'{prefix}_rank_fraction'] = pointwise.upper_tail_ecdf(
            bundle['validation_route_scores'][seed][action], frame[f'{prefix}_score'])
        scores = bundle['gates'][action].predict_proba(frame[list(module.GATE_FEATURES)])[:, 1]
        table.loc[frame.index, 'pointwise_gate_score'] = np.where(frame.llm_consistent, scores, 0.0)
    pred, routed, accepted = pointwise.apply_pointwise(
        chain=chain, module=module, table=table, dataset_key=dataset.lower(), seed=seed,
        thresholds=pd.DataFrame(bundle['thresholds']))
    table['routed'], table['accepted'], table['final_pred_pair'] = routed.astype(int), accepted.astype(int), pred
    return table.drop(columns=[c for c in table if c.startswith('gold')])


def infer(args):
    bundle = load_bundle(args.bundle)
    dataset, chain = bundle['dataset'], bundle['chain']
    records = dialogue_index(dataset, 'test', args.data_root)
    responses = (read_responses(args.responses, dataset, 'test', records) if args.stage == 'predict' else {})
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for seed in SEEDS:
        path = table_path(args.pair_root, dataset, chain, seed, 'test')
        table = read_pairs(path, dataset, 'test', seed, chain)
        if args.stage == 'route':
            table = score_router(table, chain, bundle['router_models'][seed])
            table['routed'] = route_mask(table, chain, seed, dataset, pd.DataFrame(bundle['thresholds'])).astype(int)
            table = table.drop(columns=[c for c in table if c.startswith('gold')])
        else:
            table = predict_table(table, bundle, seed, responses, records)
            # Evaluation occurs only after all predictions have been formed.
            summaries.append({'seed': seed, 'encoder': complete_gold_metrics(table, records, 'pred_pair'),
                              'elise': complete_gold_metrics(table, records),
                              'routed': int(table.routed.sum()), 'accepted': int(table.accepted.sum())})
        table.to_csv(args.output / f'seed{seed}.csv', index=False, float_format='%.17g')
    if summaries:
        (args.output / 'metrics.json').write_text(json.dumps(summaries, indent=2) + '\n')
    print(f'{args.stage}: wrote outputs to {args.output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='stage', required=True)
    for stage in ['fit', 'route', 'predict']:
        command = sub.add_parser(stage)
        command.add_argument('--pair-root', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--data-root', type=Path, default=DATA_ROOT)
        if stage == 'fit':
            command.add_argument('--dataset', choices=['ECF', 'MECAD'], required=True)
            command.add_argument('--chain', choices=['text', 'multimodal'], required=True)
        else:
            command.add_argument('--bundle', type=Path, required=True)
        if stage != 'route':
            command.add_argument('--responses', type=Path, required=True)
    args = parser.parse_args()
    fit(args) if args.stage == 'fit' else infer(args)


if __name__ == '__main__':
    main()
