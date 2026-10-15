"""Dataset identities, primary pair tables, and action-blind response alignment."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from configs.paths import DATA_ROOT
from data import load_split
from semantic_verifier_prompt import (
    PROMPT_VERSION, RESPONSE_SCHEMA, SYSTEM_PROMPT, build_prompt,
    messages_sha256, semantic_request_id,
)
from run_llm_verifier_vllm import strict_response_error


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dialogue_index(dataset, split, data_root=DATA_ROOT):
    loaded = load_split(dataset, split, data_root=data_root, candidate_scope='full_matrix')
    index = {}
    for dialogue in loaded.dialogues:
        raw_id = str(dialogue.dialogue_id)
        doc_id = str(dialogue.metadata.get('doc_id', raw_id))
        record = {
            'raw_doc_id': raw_id, 'doc_id': doc_id,
            'pairs': [(int(e), int(c)) for e, c in dialogue.emotion_cause_pairs],
            'utterances': [{'turn': int(u.turn), 'speaker': str(u.speaker or ''),
                            'text': str(u.text or '')} for u in dialogue.utterances],
        }
        for key in {raw_id, doc_id}:
            if key in index and index[key]['raw_doc_id'] != raw_id:
                raise ValueError(f'Ambiguous dialogue identity: {key}')
            index[key] = record
    return index


def request_for(record, dataset, split, emotion_turn, cause_turn, scope='lower_triangle'):
    pair_key = f'{dataset}::{split}::{record["raw_doc_id"]}::t{emotion_turn}::c{cause_turn}'
    occurrence = {'emotion_turn': int(emotion_turn), 'cause_turn': int(cause_turn)}
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': build_prompt(occurrence, record, 'blind', window=4, full_max=1_000_000)},
    ]
    return {
        'request_id': semantic_request_id(condition='blind', candidate_scope=scope,
                                           cache_key=pair_key, messages=messages),
        'dataset': dataset, 'dataset_key': dataset.lower(), 'split': split,
        'dialogue_id': record['raw_doc_id'], **occurrence,
        'pair_key': pair_key, 'cache_key': pair_key,
        'candidate_scope': scope, 'candidate_source': 'selected',
        'condition': 'blind', 'emotion_label_mode': 'hidden',
        'prompt_version': PROMPT_VERSION, 'prompt_variant': PROMPT_VERSION,
        'response_schema': RESPONSE_SCHEMA, 'messages_sha256': messages_sha256(messages),
        'messages': messages, 'modality': 'text', 'seed': 42,
        'run_name': f'{dataset.lower()}_{split}_blind', 'model': 'blind_semantic_verifier',
    }


def read_pairs(path, dataset, split, seed, chain, *, fitting=False):
    table = pd.read_csv(path, dtype={'dialogue_id': str, 'pair_key': str})
    for column, expected in [('dataset', dataset), ('split', split),
                             ('modality', 't' if chain == 'text' else 'tav'),
                             ('candidate_scope', 'lower_triangle')]:
        if column not in table or set(table[column].astype(str)) != {expected}:
            raise ValueError(f'{path}: incorrect {column}; expected {expected}')
    if table.empty or set(table['seed'].astype(int)) != {seed}:
        raise ValueError(f'{path}: empty table or wrong seed')
    keys = ['dialogue_id', 'emotion_turn', 'cause_turn']
    if table.duplicated(keys).any() or (table.cause_turn > table.emotion_turn).any():
        raise ValueError(f'{path}: duplicate or out-of-support pair')
    if not set(table.pred_pair) <= {0, 1}:
        raise ValueError(f'{path}: predictions must be binary')
    if fitting:
        if split != 'valid' or 'gold_pair' not in table or not set(table.gold_pair) <= {0, 1}:
            raise ValueError('Only labeled validation data may enter fitting')
    else:
        # Gold is never consulted by the inference path. The legacy feature-frame
        # helper requires this placeholder to construct an unused edit_correct field.
        table = table.drop(columns=[c for c in table if c.startswith('gold')], errors='ignore')
        table['gold_pair'] = 0
    table['row_pred_fraction'] = table.row_pred_count / table.row_size.clip(lower=1)
    table['dialogue_pred_fraction'] = table.dialogue_pred_count / table.dialogue_candidate_count.clip(lower=1)
    table['distance_abs'] = table.distance.abs()
    return table


def read_responses(path, dataset, split, records):
    responses = {}
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('dataset') != dataset or row.get('split') != split:
                raise ValueError(f'{path}:{line_number}: dataset/split mismatch')
            record = records[str(row['dialogue_id'])]
            e, c = int(row['emotion_turn']), int(row['cause_turn'])
            scope = row.get('candidate_scope')
            if scope not in {'lower_triangle', 'full_matrix'}:
                raise ValueError(f'{path}:{line_number}: unsupported candidate scope')
            expected = request_for(record, dataset, split, e, c, scope)
            for field in ('request_id', 'cache_key', 'messages_sha256', 'prompt_version',
                          'condition', 'emotion_label_mode', 'response_schema'):
                if row.get(field) != expected[field]:
                    raise ValueError(f'{path}:{line_number}: response {field} mismatch')
            key = (record['raw_doc_id'], e, c)
            if key in responses:
                raise ValueError(f'{path}:{line_number}: duplicate semantic response')
            payload, error = strict_response_error(row.get('response_content', ''), 'compact_verification')
            available = not error and not row.get('error') and row.get('schema_valid') == 1
            responses[key] = payload if available else None
    return responses


def attach_responses(table, responses, records, module):
    table = table.copy()
    values = []
    for row in table.itertuples():
        record = records[str(row.dialogue_id)]
        payload = responses.get((record['raw_doc_id'], int(row.emotion_turn), int(row.cause_turn)))
        payload = payload or {}
        evidence = json.dumps(payload.get('evidence_turns', []))
        values.append((int(bool(payload)), payload.get('verdict', 'uncertain'),
                       float(payload.get('confidence', 0.0)), evidence,
                       *module.evidence_payload(evidence, row.emotion_turn, row.cause_turn)))
    columns = ['llm_available', 'llm_verdict', 'llm_confidence', 'llm_evidence_turns_json',
               'evidence_count', 'evidence_has_target', 'evidence_has_cause', 'evidence_span']
    for column, values_column in zip(columns, zip(*values)):
        table[column] = list(values_column)
    return table


def complete_gold_metrics(table, records, prediction='final_pred_pair'):
    """Score against all source gold pairs, including pairs outside editable support."""
    unique = {r['raw_doc_id']: r for r in records.values()}
    gold = {(key, e, c) for key, r in unique.items() for e, c in r['pairs']}
    predicted = set()
    for row in table.itertuples():
        if int(getattr(row, prediction)):
            record = records[str(row.dialogue_id)]
            predicted.add((record['raw_doc_id'], int(row.emotion_turn), int(row.cause_turn)))
    tp, fp, fn = len(gold & predicted), len(predicted - gold), len(gold - predicted)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'gold_pairs': len(gold),
            'pair_precision': tp / (tp + fp) if tp + fp else 0.0,
            'pair_recall': tp / (tp + fn) if tp + fn else 0.0,
            'pair_f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}
