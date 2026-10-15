#!/usr/bin/env python

import torch
import numpy as np
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

class PaceExtMetricsCalculator:
    """专门负责指标计算的类"""

    def __init__(
        self,
        neutral_id: int = 0,
        dist_prior=None,
        *,
        pair_role_class_names=None,
        pair_role_cause_class_id: int = 0,
        pair_role_ignore_index: int = -100,
        context_subtype_class_names=None,
    ):
        self.neutral_id = int(neutral_id)
        self.pair_role_class_names = list(pair_role_class_names or [])
        self.pair_role_enabled = bool(self.pair_role_class_names)
        self.pair_role_cause_class_id = int(pair_role_cause_class_id)
        self.pair_role_ignore_index = int(pair_role_ignore_index)
        self.context_subtype_class_names = list(context_subtype_class_names or [])
        self.context_subtype_enabled = bool(self.context_subtype_class_names)
        self.context_subtype_thresholds = [0.5] * len(self.context_subtype_class_names)
        self.dist_prior = None
        if dist_prior is not None:
            self.set_dist_prior(dist_prior)
        self.re_init()

    def set_dist_prior(self, dist_prior):
        prior = np.asarray(dist_prior, dtype=np.float32).reshape(-1)
        if prior.size == 0:
            self.dist_prior = None
            return
        prior = prior / float(np.sum(prior))
        self.dist_prior = prior

    def re_init(self):
        """重置统计数据"""
        self.preds = defaultdict(list)
        self.golds = defaultdict(list)
        self.auprc_scores = defaultdict(list)
        self.auprc_labels = defaultdict(list)
        # Cell-level buffers for pair thresholding / gating on valid/test.
        # Each entry corresponds to one valid (w, z) cell under the mask.
        self.pair_cell_scores = []
        self.pair_cell_labels = []
        self.pair_cell_deltas = []
        self.pair_cell_emo_scores = []    # P(emo != neutral) at position w
        self.pair_cell_cause_scores = []  # P(cause == 1) at position z
        self.pair_role_golds = []
        self.pair_role_preds = []
        self.context_subtype_golds = []
        self.context_subtype_scores = []
        self.last_decode = None
        self.last_context_subtype_thresholds = list(self.context_subtype_thresholds)
        self.keys = ['default']

    def add_output(self, data, output):
        """添加一个batch的预测结果"""
        ecp_predictions, emo_predictions, cause_predictions, masks = output[:4]
        context_subtype_predictions = output[4] if len(output) > 4 else None
        emo_pred = emo_predictions.argmax(-1).cpu().numpy()
        cause_pred = cause_predictions.argmax(-1).cpu().numpy()
        masks = masks.cpu().numpy()
        pair_probs = torch.softmax(ecp_predictions, dim=-1)
        pair_score_class = self.pair_role_cause_class_id if self.pair_role_enabled else 1
        pair_score_class = max(0, min(int(pair_score_class), int(pair_probs.shape[-1]) - 1))
        pair_scores = pair_probs[..., pair_score_class].detach().cpu().numpy()
        pair_argmax = ecp_predictions.argmax(-1).detach().cpu().numpy()
        cause_scores = torch.softmax(cause_predictions, dim=-1)[..., 1].detach().cpu().numpy()
        emo_probs = torch.softmax(emo_predictions, dim=-1)
        neutral_probs = emo_probs[..., self.neutral_id]
        emo_non_neu_scores = (1.0 - neutral_probs).detach().cpu().numpy()
        context_subtype_scores = None
        if self.context_subtype_enabled and context_subtype_predictions is not None:
            context_subtype_scores = torch.sigmoid(context_subtype_predictions).detach().cpu().numpy()

        for i in range(len(emo_pred)):
            mask = masks[i]
            doc_id = data['doc_ids'][i]
            utt_nums = int(data['utterance_nums'][i].item())
            pair_num = int(data['pair_nums'][i].item())
            pairs = data['pairs'][i]

            # 处理情感预测
            emo_pred_ = emo_pred[i, :utt_nums].tolist()
            emo_gold_ = data['labels'][i, :utt_nums].tolist()
            self.preds['emo'] += emo_pred_
            self.golds['emo'] += emo_gold_

            # 处理原因预测
            cause_pred_ = cause_pred[i, :utt_nums].tolist()
            cause_gold_ = data['cause_labels'][i, :utt_nums].tolist()
            self.preds['cause'] += cause_pred_
            self.golds['cause'] += cause_gold_
            self.auprc_scores['cause'] += cause_scores[i, :utt_nums].tolist()
            self.auprc_labels['cause'] += cause_gold_

            if context_subtype_scores is not None and 'context_subtype_labels' in data and 'context_subtype_mask' in data:
                role_mask = data['context_subtype_mask'][i, :utt_nums, :utt_nums].detach().cpu().numpy().astype(bool)
                if role_mask.any():
                    role_gold = data['context_subtype_labels'][i, :utt_nums, :utt_nums].detach().cpu().numpy()
                    self.context_subtype_golds += role_gold[role_mask].astype(np.int64).tolist()
                    self.context_subtype_scores += context_subtype_scores[i, :utt_nums, :utt_nums][role_mask].tolist()

            self.golds['ecp'] += [(doc_id, *w) for w in pairs[:pair_num].tolist()]

            # AUPRC: pair scores vs. gold matrix or pair_role labels (mask内)
            pair_scores_i = pair_scores[i]
            valid = mask.astype(bool)
            if self.pair_role_enabled and 'pair_role_labels' in data:
                label_matrix = data['pair_role_labels'][i].detach().cpu().numpy()
                valid = valid & (label_matrix != self.pair_role_ignore_index)
                gold_matrix = (label_matrix == self.pair_role_cause_class_id).astype(np.int64)
                if valid.any():
                    self.pair_role_golds += label_matrix[valid].astype(np.int64).tolist()
                    self.pair_role_preds += pair_argmax[i][valid].astype(np.int64).tolist()
            else:
                gold_matrix = np.zeros_like(mask, dtype=np.int64)
                pairs_np = pairs[:pair_num].cpu().numpy()
                for w, z in pairs_np:
                    if w >= 0 and z >= 0:
                        gold_matrix[w, z] = 1
            pair_scores_valid = pair_scores_i[valid]
            pair_labels_valid = gold_matrix[valid]
            self.auprc_scores['pair'] += pair_scores_valid.tolist()
            self.auprc_labels['pair'] += pair_labels_valid.tolist()

            # Store per-cell scores for threshold search / gating.
            rows, cols = np.where(valid)
            self.pair_cell_scores += pair_scores_valid.tolist()
            self.pair_cell_labels += pair_labels_valid.tolist()
            self.pair_cell_deltas += (rows - cols).tolist()
            self.pair_cell_emo_scores += emo_non_neu_scores[i, rows].tolist()
            self.pair_cell_cause_scores += cause_scores[i, cols].tolist()

    def _safe_auprc(self, labels, scores):
        if not labels:
            return 0.0
        if len(set(labels)) < 2:
            return 0.0
        return float(average_precision_score(labels, scores))

    def _pair_role_metrics(self):
        if not self.pair_role_enabled or not self.pair_role_golds:
            return {}
        labels = list(range(len(self.pair_role_class_names)))
        gold = np.asarray(self.pair_role_golds, dtype=np.int64)
        pred = np.asarray(self.pair_role_preds, dtype=np.int64)
        precision, recall, f1, support = precision_recall_fscore_support(
            gold,
            pred,
            labels=labels,
            zero_division=0,
        )
        macro = precision_recall_fscore_support(gold, pred, labels=labels, average='macro', zero_division=0)
        weighted = precision_recall_fscore_support(gold, pred, labels=labels, average='weighted', zero_division=0)
        micro = precision_recall_fscore_support(gold, pred, labels=labels, average='micro', zero_division=0)
        metrics = {
            'pair_role_accuracy': float(np.mean(gold == pred)),
            'pair_role_macro_precision': float(macro[0]),
            'pair_role_macro_recall': float(macro[1]),
            'pair_role_macro_f1': float(macro[2]),
            'pair_role_weighted_f1': float(weighted[2]),
            'pair_role_micro_f1': float(micro[2]),
            'pair_role_support': int(len(gold)),
        }
        for index, name in enumerate(self.pair_role_class_names):
            metrics[f'pair_role_{name}_precision'] = float(precision[index])
            metrics[f'pair_role_{name}_recall'] = float(recall[index])
            metrics[f'pair_role_{name}_f1'] = float(f1[index])
            metrics[f'pair_role_{name}_support'] = int(support[index])
        return metrics

    def _default_context_subtype_threshold_candidates(self):
        return np.round(np.arange(0.05, 0.951, 0.01), 2).tolist()

    def _context_subtype_metrics(self, *, tune_thresholds: bool = False, thresholds=None):
        if not self.context_subtype_enabled or not self.context_subtype_golds:
            return {}
        gold = np.asarray(self.context_subtype_golds, dtype=np.int64)
        scores = np.asarray(self.context_subtype_scores, dtype=np.float32)
        num_classes = len(self.context_subtype_class_names)
        if gold.ndim != 2 or scores.ndim != 2 or gold.shape[1] != num_classes:
            return {}

        if thresholds is None:
            thresholds = self.last_context_subtype_thresholds or self.context_subtype_thresholds
        thresholds = [float(x) for x in thresholds]
        if len(thresholds) != num_classes:
            thresholds = [0.5] * num_classes

        if tune_thresholds:
            candidates = self._default_context_subtype_threshold_candidates()
            tuned = []
            for class_index in range(num_classes):
                class_gold = gold[:, class_index]
                class_scores = scores[:, class_index]
                best_t = 0.5
                best_f1 = -1.0
                best_p = -1.0
                best_r = -1.0
                for threshold in candidates:
                    class_pred = (class_scores >= float(threshold)).astype(np.int64)
                    p, r, f1, _ = precision_recall_fscore_support(
                        class_gold,
                        class_pred,
                        average='binary',
                        zero_division=0,
                    )
                    if (f1 > best_f1) or (f1 == best_f1 and p > best_p) or (f1 == best_f1 and p == best_p and r > best_r):
                        best_t = float(threshold)
                        best_f1 = float(f1)
                        best_p = float(p)
                        best_r = float(r)
                tuned.append(best_t)
            thresholds = tuned

        thresholds_array = np.asarray(thresholds, dtype=np.float32).reshape(1, -1)
        pred = (scores >= thresholds_array).astype(np.int64)
        labels = list(range(num_classes))
        precision, recall, f1, support = precision_recall_fscore_support(
            gold,
            pred,
            labels=labels,
            zero_division=0,
        )
        macro = precision_recall_fscore_support(gold, pred, average='macro', zero_division=0)
        weighted = precision_recall_fscore_support(gold, pred, average='weighted', zero_division=0)
        micro = precision_recall_fscore_support(gold, pred, average='micro', zero_division=0)
        metrics = {
            'context_subtype_macro_precision': float(macro[0]),
            'context_subtype_macro_recall': float(macro[1]),
            'context_subtype_macro_f1': float(macro[2]),
            'context_subtype_weighted_f1': float(weighted[2]),
            'context_subtype_micro_f1': float(micro[2]),
            'context_subtype_exact_match': float(np.mean(np.all(gold == pred, axis=1))),
            'context_subtype_item_count': int(gold.shape[0]),
            'context_subtype_thresholds': [float(x) for x in thresholds],
        }
        for index, name in enumerate(self.context_subtype_class_names):
            metrics[f'context_subtype_{name}_precision'] = float(precision[index])
            metrics[f'context_subtype_{name}_recall'] = float(recall[index])
            metrics[f'context_subtype_{name}_f1'] = float(f1[index])
            metrics[f'context_subtype_{name}_support'] = int(support[index])
            metrics[f'context_subtype_{name}_auprc'] = self._safe_auprc(
                gold[:, index].astype(np.int64).tolist(),
                scores[:, index].astype(float).tolist(),
            )
        self.last_context_subtype_thresholds = [float(x) for x in thresholds]
        return metrics

    def _pair_prf(self, tp, fp, fn):
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        f = 2 * p * r / (p + r) if p + r > 0 else 0.0
        return p, r, f

    def _compute_pair_counts(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        emo_scores: np.ndarray,
        cause_scores: np.ndarray,
        pair_threshold: float,
        gate_enable: bool,
        emo_threshold: float,
        cause_threshold: float,
    ):
        pred = scores >= float(pair_threshold)
        if gate_enable:
            pred = pred & (emo_scores >= float(emo_threshold)) & (cause_scores >= float(cause_threshold))
        pos = labels == 1
        tp = int(np.sum(pred & pos))
        pred_pos = int(np.sum(pred))
        gold_pos = int(np.sum(pos))
        fp = pred_pos - tp
        fn = gold_pos - tp
        return tp, fp, fn

    def _ensure_candidates(self, values, default_values):
        if values is None:
            return list(default_values)
        if isinstance(values, (list, tuple, np.ndarray)):
            return [float(v) for v in values]
        return [float(values)]

    def _default_pair_threshold_candidates(self):
        # Include 0.5 in the candidate set (fixed argmax-equivalent threshold).
        return np.round(np.arange(0.05, 0.951, 0.01), 2).tolist()

    def _default_gate_threshold_candidates(self):
        # Coarse grid to avoid overfitting and keep search cheap.
        return [0.0, 0.3, 0.5, 0.7, 0.9]

    def _default_dist_beta_candidates(self):
        return np.round(np.arange(0.0, 5.0 + 1e-9, 0.25), 2).tolist()

    def _apply_distance_prior(self, scores: np.ndarray, deltas: np.ndarray, beta: float, prior: np.ndarray):
        beta = float(beta)
        if beta == 0.0:
            return scores
        eps = 1e-7
        p = np.clip(scores, eps, 1.0 - eps)
        logit = np.log(p / (1.0 - p))
        deltas = np.asarray(deltas, dtype=np.int64)
        deltas = np.clip(deltas, 0, prior.size - 1)
        bias = beta * np.log(np.clip(prior[deltas], 1e-12, 1.0))
        logit = logit + bias
        return (1.0 / (1.0 + np.exp(-logit))).astype(np.float32)

    def _tune_decode_thresholds(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        emo_scores: np.ndarray,
        cause_scores: np.ndarray,
        *,
        tune_pair_threshold: bool,
        pair_threshold: float,
        gate_enable: bool,
        tune_gate_threshold: bool,
        emo_threshold: float,
        cause_threshold: float,
        pair_candidates,
        emo_candidates,
        cause_candidates,
    ):
        chosen_pair_threshold = float(pair_threshold)
        chosen_emo_threshold = float(emo_threshold)
        chosen_cause_threshold = float(cause_threshold)

        if tune_pair_threshold:
            init_gate_emo = chosen_emo_threshold if gate_enable and not tune_gate_threshold else 0.0
            init_gate_cause = chosen_cause_threshold if gate_enable and not tune_gate_threshold else 0.0
            best_pair = self._search_best_pair_threshold(
                scores, labels, emo_scores, cause_scores,
                candidates=pair_candidates,
                gate_enable=gate_enable,
                emo_threshold=init_gate_emo,
                cause_threshold=init_gate_cause,
            )
            chosen_pair_threshold = float(best_pair['pair_threshold'])

        if gate_enable and tune_gate_threshold:
            best_gate = self._search_best_gate_thresholds(
                scores, labels, emo_scores, cause_scores,
                pair_threshold=chosen_pair_threshold,
                emo_candidates=emo_candidates,
                cause_candidates=cause_candidates,
            )
            chosen_emo_threshold = float(best_gate['emo_threshold'])
            chosen_cause_threshold = float(best_gate['cause_threshold'])

        if tune_pair_threshold and gate_enable:
            best_pair = self._search_best_pair_threshold(
                scores, labels, emo_scores, cause_scores,
                candidates=pair_candidates,
                gate_enable=True,
                emo_threshold=chosen_emo_threshold,
                cause_threshold=chosen_cause_threshold,
            )
            chosen_pair_threshold = float(best_pair['pair_threshold'])

        tp, fp, fn = self._compute_pair_counts(
            scores, labels, emo_scores, cause_scores,
            pair_threshold=chosen_pair_threshold,
            gate_enable=gate_enable,
            emo_threshold=chosen_emo_threshold,
            cause_threshold=chosen_cause_threshold,
        )
        p, r, f1 = self._pair_prf(tp, fp, fn)
        return {
            'pair_threshold': chosen_pair_threshold,
            'emo_threshold': chosen_emo_threshold,
            'cause_threshold': chosen_cause_threshold,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'p': p,
            'r': r,
            'f1': f1,
        }

    def _search_best_pair_threshold(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        emo_scores: np.ndarray,
        cause_scores: np.ndarray,
        candidates,
        gate_enable: bool,
        emo_threshold: float,
        cause_threshold: float,
    ):
        best = {
            'pair_threshold': float(candidates[0]),
            'tp': 0,
            'fp': 0,
            'fn': int(np.sum(labels == 1)),
            'p': 0.0,
            'r': 0.0,
            'f1': 0.0,
        }
        for t in candidates:
            tp, fp, fn = self._compute_pair_counts(
                scores, labels, emo_scores, cause_scores,
                pair_threshold=float(t),
                gate_enable=gate_enable,
                emo_threshold=float(emo_threshold),
                cause_threshold=float(cause_threshold),
            )
            p, r, f1 = self._pair_prf(tp, fp, fn)
            if (f1 > best['f1']) or (f1 == best['f1'] and p > best['p']) or (f1 == best['f1'] and p == best['p'] and r > best['r']):
                best.update({'pair_threshold': float(t), 'tp': tp, 'fp': fp, 'fn': fn, 'p': p, 'r': r, 'f1': f1})
        return best

    def _search_best_gate_thresholds(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        emo_scores: np.ndarray,
        cause_scores: np.ndarray,
        pair_threshold: float,
        emo_candidates,
        cause_candidates,
    ):
        best = {
            'emo_threshold': float(emo_candidates[0]),
            'cause_threshold': float(cause_candidates[0]),
            'tp': 0,
            'fp': 0,
            'fn': int(np.sum(labels == 1)),
            'p': 0.0,
            'r': 0.0,
            'f1': 0.0,
        }
        for emo_t in emo_candidates:
            for cause_t in cause_candidates:
                tp, fp, fn = self._compute_pair_counts(
                    scores, labels, emo_scores, cause_scores,
                    pair_threshold=float(pair_threshold),
                    gate_enable=True,
                    emo_threshold=float(emo_t),
                    cause_threshold=float(cause_t),
                )
                p, r, f1 = self._pair_prf(tp, fp, fn)
                if (f1 > best['f1']) or (f1 == best['f1'] and p > best['p']) or (f1 == best['f1'] and p == best['p'] and r > best['r']):
                    best.update({'emo_threshold': float(emo_t), 'cause_threshold': float(cause_t), 'tp': tp, 'fp': fp, 'fn': fn, 'p': p, 'r': r, 'f1': f1})
        return best

    def report_score(self):
        """计算并返回各项指标"""
        raise NotImplementedError("Use report_score_decode(...) instead.")

    def report_score_decode(
        self,
        *,
        tune_pair_threshold: bool = False,
        pair_threshold: float = 0.5,
        gate_enable: bool = False,
        tune_gate_threshold: bool = False,
        emo_threshold: float = 0.0,
        cause_threshold: float = 0.0,
        pair_threshold_candidates=None,
        emo_threshold_candidates=None,
        cause_threshold_candidates=None,
        dist_enable: bool = False,
        dist_beta_search: bool = False,
        dist_beta: float = 0.0,
        dist_beta_candidates=None,
        tune_context_subtype_thresholds: bool = False,
        context_subtype_thresholds=None,
    ):
        """
        Compute metrics with threshold decoding (and optional emo/cause gating).

        Notes:
          - If tune_* is True, thresholds are searched on the current split (typically dev/valid).
          - For test reporting, pass tune_* = False and fixed thresholds found on dev.
        """
        scores = np.asarray(self.pair_cell_scores, dtype=np.float32)
        labels = np.asarray(self.pair_cell_labels, dtype=np.int64)
        deltas = np.asarray(self.pair_cell_deltas, dtype=np.int64)
        emo_scores = np.asarray(self.pair_cell_emo_scores, dtype=np.float32)
        cause_scores = np.asarray(self.pair_cell_cause_scores, dtype=np.float32)

        if scores.size == 0:
            pair_role_metrics = self._pair_role_metrics()
            context_subtype_metrics = self._context_subtype_metrics(
                tune_thresholds=tune_context_subtype_thresholds,
                thresholds=context_subtype_thresholds,
            )
            metrics = {
                'p': 0.0,
                'r': 0.0,
                'default': 0.0,
                'emo': 0.0,
                'cause': 0.0,
                'pair_auprc': 0.0,
                'cause_auprc': 0.0,
                'pair_threshold': float(pair_threshold),
                'gate_enable': bool(gate_enable),
                'emo_threshold': float(emo_threshold),
                'cause_threshold': float(cause_threshold),
                'dist_enable': False,
                'dist_beta': 0.0,
                **pair_role_metrics,
                **context_subtype_metrics,
            }
            if 'pair_role_macro_f1' in pair_role_metrics:
                score = float(pair_role_metrics['pair_role_macro_f1'])
            else:
                score = float(context_subtype_metrics.get('context_subtype_macro_f1', 0.0))
            return score, ("No supervised pair cells.\n", metrics)

        pair_candidates = self._ensure_candidates(pair_threshold_candidates, self._default_pair_threshold_candidates())
        emo_candidates = self._ensure_candidates(emo_threshold_candidates, self._default_gate_threshold_candidates())
        cause_candidates = self._ensure_candidates(cause_threshold_candidates, self._default_gate_threshold_candidates())

        prior = self.dist_prior
        dist_enable = bool(dist_enable) and prior is not None and prior.size > 0
        dist_candidates = self._ensure_candidates(dist_beta_candidates, self._default_dist_beta_candidates())
        if 0.0 not in dist_candidates:
            dist_candidates = [0.0] + dist_candidates

        chosen_beta = float(dist_beta)
        best_decode = None
        if dist_enable and dist_beta_search:
            for beta in dist_candidates:
                scores_adj = self._apply_distance_prior(scores, deltas, beta, prior)
                tuned = self._tune_decode_thresholds(
                    scores_adj, labels, emo_scores, cause_scores,
                    tune_pair_threshold=tune_pair_threshold,
                    pair_threshold=pair_threshold,
                    gate_enable=gate_enable,
                    tune_gate_threshold=tune_gate_threshold,
                    emo_threshold=emo_threshold,
                    cause_threshold=cause_threshold,
                    pair_candidates=pair_candidates,
                    emo_candidates=emo_candidates,
                    cause_candidates=cause_candidates,
                )
                candidate = {
                    **tuned,
                    'dist_beta': float(beta),
                }
                if best_decode is None:
                    best_decode = candidate
                else:
                    if (candidate['f1'] > best_decode['f1']) or (candidate['f1'] == best_decode['f1'] and candidate['p'] > best_decode['p']) or (candidate['f1'] == best_decode['f1'] and candidate['p'] == best_decode['p'] and candidate['r'] > best_decode['r']):
                        best_decode = candidate
            chosen_beta = float(best_decode['dist_beta'])
        else:
            if not dist_enable:
                chosen_beta = 0.0

        scores_used = scores
        if dist_enable and chosen_beta != 0.0:
            scores_used = self._apply_distance_prior(scores, deltas, chosen_beta, prior)

        if best_decode is None or (not dist_enable or not dist_beta_search):
            best_decode = self._tune_decode_thresholds(
                scores_used, labels, emo_scores, cause_scores,
                tune_pair_threshold=tune_pair_threshold,
                pair_threshold=pair_threshold,
                gate_enable=gate_enable,
                tune_gate_threshold=tune_gate_threshold,
                emo_threshold=emo_threshold,
                cause_threshold=cause_threshold,
                pair_candidates=pair_candidates,
                emo_candidates=emo_candidates,
                cause_candidates=cause_candidates,
            )
            best_decode['dist_beta'] = chosen_beta

        tp = best_decode['tp']
        fp = best_decode['fp']
        fn = best_decode['fn']
        p = best_decode['p']
        r = best_decode['r']
        f = best_decode['f1']
        chosen_pair_threshold = float(best_decode['pair_threshold'])
        chosen_emo_threshold = float(best_decode['emo_threshold'])
        chosen_cause_threshold = float(best_decode['cause_threshold'])

        # 计算情感和原因分数
        gold_emo = [0 if w == self.neutral_id else 1 for w in self.golds['emo']]
        pred_emo = [0 if w == self.neutral_id else 1 for w in self.preds['emo']]
        emo = precision_recall_fscore_support(gold_emo, pred_emo, average='binary')
        cause = precision_recall_fscore_support(self.golds['cause'], self.preds['cause'], average='binary')

        pair_auprc = self._safe_auprc(self.auprc_labels['pair'], self.auprc_scores['pair'])
        cause_auprc = self._safe_auprc(self.auprc_labels['cause'], self.auprc_scores['cause'])

        # 生成结果字符串
        res = (f"Pair Pre. {p*100:.4f}\t Rec. {r*100:.4f}\tF1 {f*100:.4f}\n"
               f"TP {tp}\tPred. {tp+fp}\tGold. {tp+fn}\n"
               f"Decode: pair_th={chosen_pair_threshold:.3f}"
               f"\tgate={int(bool(gate_enable))}"
               f"\temo_th={chosen_emo_threshold:.3f}"
               f"\tcause_th={chosen_cause_threshold:.3f}"
               f"\tdist={int(bool(dist_enable))}"
               f"\tbeta={chosen_beta:.3f}\n"
               f"Emo: Pre. {emo[0]*100:.4f}\t Rec. {emo[1]*100:.4f}\tF1 {emo[2]*100:.4f}\n"
               f"Cause: Pre. {cause[0]*100:.4f}\t Rec. {cause[1]*100:.4f}\tF1 {cause[2]*100:.4f}\n")

        self.last_decode = {
            'pair_threshold': chosen_pair_threshold,
            'gate_enable': bool(gate_enable),
            'emo_threshold': chosen_emo_threshold,
            'cause_threshold': chosen_cause_threshold,
            'dist_enable': bool(dist_enable),
            'dist_beta': chosen_beta,
        }

        pair_role_metrics = self._pair_role_metrics()
        context_subtype_metrics = self._context_subtype_metrics(
            tune_thresholds=tune_context_subtype_thresholds,
            thresholds=context_subtype_thresholds,
        )
        if pair_role_metrics:
            res += (
                f"PairRole: Acc {pair_role_metrics['pair_role_accuracy']*100:.4f}\t"
                f"Macro-F1 {pair_role_metrics['pair_role_macro_f1']*100:.4f}\t"
                f"Weighted-F1 {pair_role_metrics['pair_role_weighted_f1']*100:.4f}\n"
            )
        if context_subtype_metrics:
            res += (
                f"ContextSubtype: Macro-F1 {context_subtype_metrics['context_subtype_macro_f1']*100:.4f}\t"
                f"Micro-F1 {context_subtype_metrics['context_subtype_micro_f1']*100:.4f}\t"
                f"Exact {context_subtype_metrics['context_subtype_exact_match']*100:.4f}\n"
            )

        metrics = {
            'p': p,
            'r': r,
            'default': f,
            'emo': emo[2],
            'cause': cause[2],
            'pair_auprc': pair_auprc,
            'cause_auprc': cause_auprc,
            'pair_threshold': chosen_pair_threshold,
            'gate_enable': bool(gate_enable),
            'emo_threshold': chosen_emo_threshold,
            'cause_threshold': chosen_cause_threshold,
            'dist_enable': bool(dist_enable),
            'dist_beta': chosen_beta,
            **pair_role_metrics,
            **context_subtype_metrics,
        }
        if self.pair_role_enabled and 'pair_role_macro_f1' in metrics:
            score = float(metrics['pair_role_macro_f1'])
        elif self.context_subtype_enabled and 'context_subtype_macro_f1' in metrics:
            score = float(metrics['context_subtype_macro_f1'])
        else:
            score = f
        return score, (res, metrics)

    def get_final_test_results(self):
        """获取test set的最终结果数据"""
        scores = np.asarray(self.pair_cell_scores, dtype=np.float32)
        labels = np.asarray(self.pair_cell_labels, dtype=np.int64)
        deltas = np.asarray(self.pair_cell_deltas, dtype=np.int64)
        emo_scores = np.asarray(self.pair_cell_emo_scores, dtype=np.float32)
        cause_scores = np.asarray(self.pair_cell_cause_scores, dtype=np.float32)

        decode = self.last_decode or {'pair_threshold': 0.5, 'gate_enable': False, 'emo_threshold': 0.0, 'cause_threshold': 0.0, 'dist_enable': False, 'dist_beta': 0.0}
        if bool(decode.get('dist_enable', False)) and self.dist_prior is not None and float(decode.get('dist_beta', 0.0)) != 0.0:
            scores = self._apply_distance_prior(scores, deltas, float(decode['dist_beta']), self.dist_prior)
        if scores.size == 0:
            return {
                'pair_f1': 0.0,
                'pair_precision': 0.0,
                'pair_recall': 0.0,
                'pair_auprc': 0.0,
                'tp': 0,
                'fp': 0,
                'fn': 0,
                **self._pair_role_metrics(),
                **self._context_subtype_metrics(thresholds=self.last_context_subtype_thresholds),
            }

        tp, fp, fn = self._compute_pair_counts(
            scores, labels, emo_scores, cause_scores,
            pair_threshold=decode['pair_threshold'],
            gate_enable=bool(decode['gate_enable']),
            emo_threshold=decode['emo_threshold'],
            cause_threshold=decode['cause_threshold'],
        )
        pair_p, pair_r, pair_f1 = self._pair_prf(tp, fp, fn)

        # 计算情感和原因分数
        gold_emo = [0 if w == self.neutral_id else 1 for w in self.golds['emo']]
        pred_emo = [0 if w == self.neutral_id else 1 for w in self.preds['emo']]
        emo = precision_recall_fscore_support(gold_emo, pred_emo, average='binary')
        cause = precision_recall_fscore_support(self.golds['cause'], self.preds['cause'], average='binary')

        pair_auprc = self._safe_auprc(self.auprc_labels['pair'], self.auprc_scores['pair'])
        cause_auprc = self._safe_auprc(self.auprc_labels['cause'], self.auprc_scores['cause'])

        return {
            'pair_f1': pair_f1,
            'pair_precision': pair_p,
            'pair_recall': pair_r,
            'pair_auprc': pair_auprc,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'emo_f1': emo[2],
            'emo_precision': emo[0],
            'emo_recall': emo[1],
            'cause_f1': cause[2],
            'cause_precision': cause[0],
            'cause_recall': cause[1],
            'cause_auprc': cause_auprc,
            'pair_threshold': decode['pair_threshold'],
            'gate_enable': bool(decode['gate_enable']),
            'emo_threshold': decode['emo_threshold'],
            'cause_threshold': decode['cause_threshold'],
            'dist_enable': bool(decode.get('dist_enable', False)),
            'dist_beta': float(decode.get('dist_beta', 0.0)),
            **self._pair_role_metrics(),
            **self._context_subtype_metrics(thresholds=self.last_context_subtype_thresholds),
        }


MetricsCalculator = PaceExtMetricsCalculator
