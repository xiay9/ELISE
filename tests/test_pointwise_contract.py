import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import pointwise
import run_pointwise as pipeline
import text_gate
from data_io import (attach_responses, complete_gold_metrics, read_responses,
                     request_for)
from semantic_verifier_prompt import build_prompt


class FixedGate:
    def predict_proba(self, frame):
        # Depend on the validation reference coordinate, so accidental use of
        # the current test pool changes the decisions in the partition test.
        rank = frame.iloc[:, 2].to_numpy()
        scores = 1.0 - 0.6 * rank
        return np.column_stack([1.0 - scores, scores])


def fixture(chain='text'):
    records = {'1': {'raw_doc_id': '1', 'doc_id': '1', 'pairs': [(2, 1), (1, 2)],
                     'utterances': [{'turn': 1, 'speaker': 'A', 'text': '发生了什么？'},
                                    {'turn': 2, 'speaker': 'B', 'text': 'I missed the train.'}]}}
    rows = []
    for i, (e, c, pred, score) in enumerate([(1, 1, 0, .3), (2, 1, 0, .8), (2, 2, 1, .4)]):
        row = {name: 0.25 for name in pipeline.router.PACEEXT_FEATURES}
        row.update(dataset='ECF', split='test', seed=42, modality='t' if chain == 'text' else 'tav',
                   dialogue_id='1', emotion_turn=e, cause_turn=c, pred_pair=pred,
                   gold_pair=0, pair_score=score, non_pair_score=1-score,
                   pair_key=f'ECF::test::1::t{e}::c{c}', row_rank=i+1,
                   row_size=3, row_pred_count=1, distance=e-c, distance_abs=abs(e-c))
        rows.append(row)
    table = pd.DataFrame(rows)
    thresholds = [dict(chain=chain, dataset_key='ecf', seed=42, action=a,
                       route_threshold=.1, gate_threshold=.65) for a in pipeline.ACTIONS]
    bundle = {'chain': chain, 'dataset': 'ECF', 'router_models': {42: {'add': None, 'delete': None}},
              'gates': {a: FixedGate() for a in pipeline.ACTIONS},
              'validation_route_scores': {42: {a: np.array([.1, .2, .5, .7, .9]) for a in pipeline.ACTIONS}},
              'thresholds': thresholds}
    responses = {('1', 1, 1): {'verdict': 'support_pair', 'confidence': .9, 'evidence_turns': [1]},
                 ('1', 2, 1): {'verdict': 'support_pair', 'confidence': .9, 'evidence_turns': [1, 2]},
                 ('1', 2, 2): {'verdict': 'reject_pair', 'confidence': .9, 'evidence_turns': [2]}}
    return table, bundle, records, responses


def test_prompt_is_action_and_label_blind():
    _, _, records, _ = fixture()
    base = {'emotion_turn': 2, 'cause_turn': 1}
    changed = dict(base, pred_pair=1, pair_score=.99, gold_pair=0, selector_action='delete')
    a = build_prompt(base, records['1'], 'blind', window=4, full_max=1_000_000)
    b = build_prompt(changed, records['1'], 'blind', window=4, full_max=1_000_000)
    assert a == b and '发生了什么？' in a and 'I missed the train.' in a


@pytest.mark.parametrize('chain', ['text', 'multimodal'])
def test_partition_and_order_invariance(chain):
    table, bundle, records, responses = fixture(chain)
    whole = pipeline.predict_table(table, bundle, 42, responses, records)
    pieces = pd.concat([pipeline.predict_table(table.iloc[[i]].copy(), bundle, 42, responses, records)
                        for i in [2, 0, 1]]).sort_index()
    np.testing.assert_array_equal(whole.final_pred_pair, pieces.final_pred_pair)
    np.testing.assert_allclose(whole.pointwise_gate_score, pieces.pointwise_gate_score)
    changed = table.assign(gold_pair=1)
    other = pipeline.predict_table(changed, bundle, 42, responses, records)
    np.testing.assert_array_equal(whole.final_pred_pair, other.final_pred_pair)


@pytest.mark.parametrize('chain', ['text', 'multimodal'])
def test_missing_uncertain_and_inconsistent_evidence_fail_closed(chain):
    table, bundle, records, _ = fixture(chain)
    for responses in [{}, {('1', 2, 1): {'verdict': 'uncertain', 'confidence': 1., 'evidence_turns': [1]}},
                      {('1', 2, 1): {'verdict': 'reject_pair', 'confidence': 1., 'evidence_turns': [1]}}]:
        result = pipeline.predict_table(table, bundle, 42, responses, records)
        np.testing.assert_array_equal(result.final_pred_pair, table.pred_pair)


def test_complete_gold_keeps_uneditable_positive():
    table, _, records, _ = fixture()
    table['final_pred_pair'] = [0, 1, 0]
    metrics = complete_gold_metrics(table, records)
    assert metrics['tp'] == 1 and metrics['fn'] == 1
    assert metrics['pair_f1'] == pytest.approx(2/3)


def test_response_contract_and_malformed_payload(tmp_path):
    _, _, records, _ = fixture()
    request = request_for(records['1'], 'ECF', 'test', 2, 1)
    row = dict(request, response_content='malformed', schema_valid=1, error='')
    path = tmp_path/'responses.jsonl'
    path.write_text(json.dumps(row)+'\n')
    assert read_responses(path, 'ECF', 'test', records)[('1', 2, 1)] is None
    row['messages_sha256'] = 'tampered'
    path.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError, match='messages_sha256'):
        read_responses(path, 'ECF', 'test', records)


def test_cutoff_ties_and_boundary_rates():
    scores = [.2, .7, .7, .9]
    assert pointwise.finite_threshold(scores, 0) > max(scores)
    assert pointwise.finite_threshold(scores, 1) < min(scores)
    threshold = pointwise.finite_threshold(scores, .5)
    assert sum(v >= threshold for v in scores) == 3
    np.testing.assert_allclose(pointwise.upper_tail_ecdf(scores, [.7, .9, 1.]), [.75, .25, 0.])


def test_oof_gate_keeps_seed_copies_in_same_fold(monkeypatch):
    real_splitter = text_gate.StratifiedGroupKFold
    observed = []
    class CheckedSplitter:
        def __init__(self, **kwargs):
            self.inner = real_splitter(**kwargs)
        def split(self, X, y, groups):
            for fit, held in self.inner.split(X, y, groups):
                assert set(groups[fit]).isdisjoint(groups[held])
                observed.append(len(held))
                yield fit, held
    monkeypatch.setattr(text_gate, 'StratifiedGroupKFold', CheckedSplitter)
    rows = []
    for seed in pipeline.SEEDS:
        for dialogue in range(30):
            row = {name: .1 + dialogue / 40 for name in text_gate.GATE_FEATURES}
            row.update(seed=seed, dialogue_id=str(dialogue), llm_consistent=True,
                       edit_correct=dialogue % 2)
            rows.append(row)
    scores, _ = text_gate.oof_gate_scores(pd.DataFrame(rows), text_gate.GateCandidate('linear', 'linear'))
    assert len(observed) == 5 and len(scores) == 90 and np.isfinite(scores).all()


@pytest.mark.parametrize('chain', ['text', 'multimodal'])
def test_validation_fit_freeze_and_test_execution(tmp_path, monkeypatch, chain):
    # Fictional dialogues and tiny grids exercise the public workflow without
    # using benchmark outcomes or running encoder/LLM inference.
    records = {}
    for i in range(30):
        records[str(i)] = {'raw_doc_id': str(i), 'doc_id': str(i),
                          'pairs': [(2, 1)] if i % 2 else [(2, 2)],
                          'utterances': [{'turn': 1, 'speaker': 'A', 'text': 'An event happened.'},
                                         {'turn': 2, 'speaker': 'B', 'text': 'I am upset.'}]}
    monkeypatch.setattr(pipeline, 'dialogue_index', lambda *a, **kw: records)
    monkeypatch.setattr(pointwise, 'POINTWISE_ROUTE_RATES', (0., .5, 1.))
    monkeypatch.setattr(pointwise, 'ACCEPT_RATES', (0., .5, 1.))
    pair_root = tmp_path/'pairs'
    (pair_root/'ecf').mkdir(parents=True)
    for split in ['valid', 'test']:
        response_path = tmp_path/f'{split}.jsonl'
        response_rows = []
        for i in range(30):
            for c in [1, 2]:
                request = request_for(records[str(i)], 'ECF', split, 2, c)
                payload = {'verdict': 'support_pair' if c == 1 else 'reject_pair',
                           'confidence': .6 + (i % 4)/10, 'evidence_turns': [c]}
                response_rows.append(dict(request, response_content=json.dumps(payload),
                                          schema_valid=1, error=''))
        response_path.write_text(''.join(json.dumps(r)+'\n' for r in response_rows))
        for seed in pipeline.SEEDS:
            rows = []
            for i in range(30):
                for c in [1, 2]:
                    row = {f: .1 + (i % 7)/10 for f in pipeline.router.PACEEXT_FEATURES}
                    row.update(dataset='ECF', split=split, seed=seed, candidate_scope='lower_triangle',
                               modality='t' if chain=='text' else 'tav', dialogue_id=str(i),
                               emotion_turn=2, cause_turn=c, pred_pair=c-1,
                               gold_pair=int((2,c) in records[str(i)]['pairs']),
                               row_size=2, row_pred_count=1, dialogue_pred_count=1,
                               dialogue_candidate_count=2, row_rank=c,
                               pair_key=f'ECF::{split}::{i}::t2::c{c}',
                               pair_score=.2 + (i % 6)/10, non_pair_score=.8-(i % 6)/10,
                               distance=2-c, distance_abs=2-c)
                    rows.append(row)
            pd.DataFrame(rows).to_csv(pipeline.table_path(pair_root,'ECF',chain,seed,split),index=False)
    bundle_path = tmp_path/'bundle'
    pipeline.fit(SimpleNamespace(output=bundle_path, chain=chain, dataset='ECF',
                                 responses=tmp_path/'valid.jsonl', pair_root=pair_root, data_root=tmp_path))
    bundle = pipeline.load_bundle(bundle_path)
    assert bundle['dataset'] == 'ECF' and len(bundle['thresholds']) == 6
    for stage in ['route', 'predict']:
        args = SimpleNamespace(stage=stage, bundle=bundle_path, pair_root=pair_root,
                               output=tmp_path/stage, data_root=tmp_path, responses=tmp_path/'test.jsonl')
        pipeline.infer(args)
        assert len(list(args.output.glob('seed*.csv'))) == 3
    metrics = json.loads((tmp_path/'predict/metrics.json').read_text())
    assert len(metrics) == 3 and all(0 <= r['elise']['pair_f1'] <= 1 for r in metrics)
