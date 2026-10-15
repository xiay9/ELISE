#!/use/bin/env python

"""
Name: trainer.py
"""

import os
import json
import time
import hashlib
import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
import wandb
from .validation import PaceExtMetricsCalculator
from .eng_metrics import count_parameters, collect_eng_metrics

class PaceExtTrainer:
    def __init__(self, model, config, train_loader, valid_loader, test_loader):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.save_dir = config.target_dir
        self.run_label = os.path.basename(os.path.normpath(self.save_dir)) or 'run'
        self.artifact_label = self._build_artifact_label()
        self.save_best_k = max(1, int(getattr(config, 'save_best_k', 1)))
        self.best_checkpoints = []

        # 用于记录最佳结果
        self.scores = []
        self.lines = []

        # 指标计算器
        self.metrics_calculator = self._new_metrics_calculator()
        self.best_pair_decode = None
        self.best_context_subtype_thresholds = None
        self.best_checkpoint_path = None
        self.wandb_run = None
        self.fixed_base_decode = getattr(self.config, 'teacher_fixed_base_decode', None)
        self.epoch_history = []

    def _new_metrics_calculator(self):
        neutral_id = int(getattr(self.config, 'neutral_label_id', 0))
        return PaceExtMetricsCalculator(
            neutral_id=neutral_id,
            dist_prior=getattr(self.config, 'dist_prior', None),
            pair_role_class_names=getattr(self.config, 'pair_role_class_names', None),
            pair_role_cause_class_id=int(getattr(self.config, 'pair_role_cause_class_id', 0)),
            pair_role_ignore_index=int(getattr(self.config, 'pair_role_ignore_index', -100)),
            context_subtype_class_names=getattr(self.config, 'context_subtype_class_names', None),
        )

    def _is_jsonable(self, value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return True
        if isinstance(value, (list, tuple)):
            return all(self._is_jsonable(v) for v in value)
        if isinstance(value, dict):
            return all(isinstance(k, str) and self._is_jsonable(v) for k, v in value.items())
        return False

    def _build_wandb_config(self):
        skip_keys = {'optimizer', 'scheduler', 'device', 'label_dict', 'speaker_dict'}
        config_dict = {}
        for key, value in self.config.items():
            if key in skip_keys:
                continue
            if self._is_jsonable(value):
                config_dict[key] = value
        return config_dict

    def _init_wandb(self):
        project = getattr(self.config, 'wandb_project', 'utdg')
        entity = getattr(self.config, 'wandb_entity', '')
        mode = getattr(self.config, 'wandb_mode', 'online')
        run_name = getattr(self.config, 'wandb_run_name', '')

        if not run_name:
            seed = getattr(self.config, 'seed', 'na')
            run_name = f"{self.run_label}_seed{seed}"

        kwargs = {
            'project': project,
            'name': run_name,
            'mode': mode,
            'config': self._build_wandb_config()
        }
        if entity:
            kwargs['entity'] = entity

        self.wandb_run = wandb.init(**kwargs)

    def _is_number(self, value):
        return isinstance(value, (int, float, np.integer, np.floating))

    def _to_jsonable(self, value):
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.item()
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(v) for v in value]
        return value

    def _build_artifact_label(self):
        dataset_name = str(getattr(self.config, 'dataset_name', '') or 'run').lower()
        seed = getattr(self.config, 'seed', None)
        seed_tag = f"seed{seed}" if seed is not None else "seedna"
        label_hash = hashlib.sha1(self.run_label.encode('utf-8')).hexdigest()[:10]
        return f"{dataset_name}_{seed_tag}_{label_hash}"

    def _dataset_stats(self):
        return {
            'train_size': len(self.train_loader.dataset),
            'valid_size': len(self.valid_loader.dataset),
            'test_size': len(self.test_loader.dataset),
            'train_batches': len(self.train_loader),
            'valid_batches': len(self.valid_loader),
            'test_batches': len(self.test_loader),
        }

    def _evaluate_loader_metrics(self, data_loader, *, override_decode=None):
        self.model.eval()
        metrics_calculator = self._new_metrics_calculator()
        for data in data_loader:
            with torch.no_grad():
                data = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
                _, output = self.model(**data)
                metrics_calculator.add_output(data, output)
        decode_kwargs = self._pair_decode_kwargs(tune_decode=False, override_decode=override_decode or self.best_pair_decode)
        metrics_calculator.report_score_decode(
            **decode_kwargs,
            tune_context_subtype_thresholds=False,
            context_subtype_thresholds=self.best_context_subtype_thresholds,
        )
        return metrics_calculator.get_final_test_results()

    def _pair_decode_kwargs(self, *, tune_decode: bool, override_decode=None):
        cfg = self.config
        gate_enable = bool(getattr(cfg, 'pair_gate_enable', False))
        tune_pair_threshold = bool(getattr(cfg, 'pair_threshold_search', True)) and bool(tune_decode)
        tune_gate_threshold = gate_enable and bool(getattr(cfg, 'pair_gate_search', True)) and bool(tune_decode)

        pair_threshold = float(getattr(cfg, 'pair_threshold', 0.5))
        emo_threshold = float(getattr(cfg, 'pair_gate_emo_threshold', 0.0))
        cause_threshold = float(getattr(cfg, 'pair_gate_cause_threshold', 0.0))

        pair_threshold_candidates = getattr(cfg, 'pair_threshold_candidates', None)
        emo_threshold_candidates = getattr(cfg, 'pair_gate_emo_threshold_candidates', None)
        cause_threshold_candidates = getattr(cfg, 'pair_gate_cause_threshold_candidates', None)

        dist_enable = bool(getattr(cfg, 'dist_prior_enable', True))
        dist_beta_search = dist_enable and bool(getattr(cfg, 'dist_prior_beta_search', True)) and bool(tune_decode)
        dist_beta = float(getattr(cfg, 'dist_prior_beta', 0.0))

        dist_beta_candidates = getattr(cfg, 'dist_prior_beta_candidates', None)
        if dist_beta_candidates is None:
            beta_min = float(getattr(cfg, 'dist_prior_beta_min', 0.0))
            beta_max = float(getattr(cfg, 'dist_prior_beta_max', 5.0))
            beta_step = float(getattr(cfg, 'dist_prior_beta_step', 0.25))
            if beta_step <= 0:
                beta_step = 0.25
            dist_beta_candidates = np.arange(beta_min, beta_max + 1e-9, beta_step).round(6).tolist()

        if override_decode:
            if 'gate_enable' in override_decode:
                gate_enable = bool(override_decode['gate_enable'])
            if 'pair_threshold' in override_decode:
                pair_threshold = float(override_decode['pair_threshold'])
            if 'emo_threshold' in override_decode:
                emo_threshold = float(override_decode['emo_threshold'])
            if 'cause_threshold' in override_decode:
                cause_threshold = float(override_decode['cause_threshold'])
            if 'dist_enable' in override_decode:
                dist_enable = bool(override_decode['dist_enable'])
            if 'dist_beta' in override_decode:
                dist_beta = float(override_decode['dist_beta'])

        return {
            'tune_pair_threshold': tune_pair_threshold,
            'pair_threshold': pair_threshold,
            'gate_enable': gate_enable,
            'tune_gate_threshold': tune_gate_threshold,
            'emo_threshold': emo_threshold,
            'cause_threshold': cause_threshold,
            'pair_threshold_candidates': pair_threshold_candidates,
            'emo_threshold_candidates': emo_threshold_candidates,
            'cause_threshold_candidates': cause_threshold_candidates,
            'dist_enable': dist_enable,
            'dist_beta_search': dist_beta_search,
            'dist_beta': dist_beta,
            'dist_beta_candidates': dist_beta_candidates,
        }

    def _save_run_json(self, payload):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f"run_{self.artifact_label}_{timestamp}.json"
        path = os.path.join(self.save_dir, filename)
        payload = self._to_jsonable(payload)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)
        return path

    def _seed_tag(self):
        seed = getattr(self, 'used_seed', None)
        if seed is None:
            seed = getattr(self.config, 'seed', None)
        return f"seed{seed}" if seed is not None else "seedna"

    def _format_f1(self, score):
        return f"{score:.4f}"

    def _checkpoint_path(self, score):
        score_tag = self._format_f1(score)
        return os.path.join(
            self.save_dir,
            f"best_{self.artifact_label}_f1_{score_tag}.pt"
        )

    def train(self):
        best_score, best_iter = float('-inf'), -1
        self._init_wandb()
        enable_eng = bool(getattr(self.config, 'eng_metrics', False))
        run_constants = {
            'seed': getattr(self, 'used_seed', None),
            'device': str(self.config.device),
            'config': self._build_wandb_config(),
            'data': self._dataset_stats()
        }
        if enable_eng:
            run_constants['params'] = count_parameters(self.model)

        total_steps = self.config.epoch_size * len(self.train_loader)
        progress = tqdm(total=total_steps)
        for epoch in range(self.config.epoch_size):
            self.model.global_epoch = epoch
            self.global_epoch = epoch

            # 训练和评估
            train_loss, train_loss_components = self.train_step(progress, epoch)
            score, (res, metrics) = self.evaluate_step()
            teacher_loss_weight = None
            if hasattr(self.model, "get_teacher_loss_weight"):
                teacher_loss_weight = float(self.model.get_teacher_loss_weight())

            # 重置统计数据并记录结果
            self.metrics_calculator.re_init()
            self.add_instance(score, res)
            primary_metric_name = (
                'pair_role_macro_f1'
                if metrics and 'pair_role_macro_f1' in metrics
                else (
                    'context_subtype_macro_f1'
                    if metrics and 'context_subtype_macro_f1' in metrics
                    else 'pair_f1'
                )
            )
            epoch_record = {
                'epoch': epoch,
                'train_loss': train_loss,
                'primary_metric': primary_metric_name,
                'val_primary_score': score,
                'val_pair_f1': metrics.get('default', score) if metrics else score,
                'teacher_loss_weight': teacher_loss_weight,
            }
            if metrics:
                epoch_record['valid_metrics'] = self._to_jsonable(metrics)
            if train_loss_components:
                epoch_record['train_loss_components'] = self._to_jsonable(train_loss_components)
            self.epoch_history.append(epoch_record)

            # 保存最佳模型（按配置保留Top-K）
            should_save = False
            if len(self.best_checkpoints) < self.save_best_k:
                should_save = True
            else:
                worst_score = min(self.best_checkpoints, key=lambda x: x[0])[0]
                should_save = score > worst_score

            if should_save:
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)

                save_path = self._checkpoint_path(score)
                torch.save(
                    {
                        'epoch': epoch,
                        'model': self.model.cpu().state_dict(),
                        'score': score,
                        'pair_decode': {
                            'pair_threshold': metrics.get('pair_threshold', 0.5),
                            'gate_enable': bool(metrics.get('gate_enable', False)),
                            'emo_threshold': metrics.get('emo_threshold', 0.0),
                            'cause_threshold': metrics.get('cause_threshold', 0.0),
                            'dist_enable': bool(metrics.get('dist_enable', False)),
                            'dist_beta': metrics.get('dist_beta', 0.0),
                        } if metrics else None,
                        'context_subtype_thresholds': metrics.get('context_subtype_thresholds') if metrics else None,
                    },
                    save_path
                )
                self.model.to(self.config.device)

                self.best_checkpoints.append((score, save_path))
                self.best_checkpoints.sort(reverse=True, key=lambda x: x[0])
                while len(self.best_checkpoints) > self.save_best_k:
                    _, drop_path = self.best_checkpoints.pop()
                    # 若由于分数四舍五入导致路径相同，避免误删仍在使用的checkpoint
                    if any(path == drop_path for _, path in self.best_checkpoints):
                        continue
                    if os.path.exists(drop_path):
                        os.remove(drop_path)

            if score > best_score:
                best_score, best_iter = score, epoch
                if should_save:
                    self.best_checkpoint_path = save_path
                if metrics:
                    self.best_pair_decode = {
                        'pair_threshold': metrics.get('pair_threshold', 0.5),
                        'gate_enable': bool(metrics.get('gate_enable', False)),
                        'emo_threshold': metrics.get('emo_threshold', 0.0),
                        'cause_threshold': metrics.get('cause_threshold', 0.0),
                        'dist_enable': bool(metrics.get('dist_enable', False)),
                        'dist_beta': metrics.get('dist_beta', 0.0),
                    }
                    self.best_context_subtype_thresholds = metrics.get('context_subtype_thresholds')

            if self.wandb_run is not None:
                log_data = {
                    'train/loss': train_loss,
                    'val/primary_score': score,
                }
                if metrics:
                    log_data['val/pair_f1'] = metrics.get('default', score)
                    if 'pair_role_macro_f1' in metrics:
                        log_data['val/pair_role_macro_f1'] = metrics['pair_role_macro_f1']
                    if 'pair_role_accuracy' in metrics:
                        log_data['val/pair_role_accuracy'] = metrics['pair_role_accuracy']
                    if 'context_subtype_macro_f1' in metrics:
                        log_data['val/context_subtype_macro_f1'] = metrics['context_subtype_macro_f1']
                    if 'context_subtype_micro_f1' in metrics:
                        log_data['val/context_subtype_micro_f1'] = metrics['context_subtype_micro_f1']
                    if 'p' in metrics:
                        log_data['val/pair_precision'] = metrics['p']
                    if 'r' in metrics:
                        log_data['val/pair_recall'] = metrics['r']
                    if 'emo' in metrics:
                        log_data['val/emo_f1'] = metrics['emo']
                    if 'cause' in metrics:
                        log_data['val/cause_f1'] = metrics['cause']
                    if 'pair_auprc' in metrics:
                        log_data['val/pair_auprc'] = metrics['pair_auprc']
                    if 'cause_auprc' in metrics:
                        log_data['val/cause_auprc'] = metrics['cause_auprc']
                wandb.log(log_data, step=epoch)

            # 早停
            if epoch - best_iter > self.config.patience:
                break
        progress.close()

        # 最终评估 - 获取test set结果
        best_path = self.best_checkpoint_path
        if best_path is None and self.best_checkpoints:
            best_path = self.best_checkpoints[0][1]
        score, res = self.final_evaluate(best_path)
        self.final_score, self.final_res = score, res

        # 获取test set的详细结果数据
        test_result_data = self.metrics_calculator.get_final_test_results()
        train_metrics = self._evaluate_loader_metrics(self.train_loader)
        valid_metrics = self._evaluate_loader_metrics(self.valid_loader)
        fixed_base_decode_metrics = None
        if self.fixed_base_decode:
            fixed_base_decode_metrics = {
                'train': self._evaluate_loader_metrics(self.train_loader, override_decode=self.fixed_base_decode),
                'valid': self._evaluate_loader_metrics(self.valid_loader, override_decode=self.fixed_base_decode),
                'test': self._evaluate_loader_metrics(self.test_loader, override_decode=self.fixed_base_decode),
                'decode': self.fixed_base_decode,
            }
        latency_stats = {}
        flops_stats = {}
        if enable_eng:
            latency_stats, flops_stats = collect_eng_metrics(
                self.model,
                self.test_loader,
                self.config.device
            )

        primary_test_metric = (
            test_result_data.get('pair_role_macro_f1')
            if bool(getattr(self.config, 'pair_role_enable', False))
            else (
                test_result_data.get('context_subtype_macro_f1')
                if bool(getattr(self.config, 'context_subtype_enable', False))
                else test_result_data.get('pair_f1')
            )
        )
        if best_path and primary_test_metric is not None:
            test_tag = self._format_f1(primary_test_metric)
            new_path = os.path.join(
                self.save_dir,
                f"best_{self.artifact_label}_f1_{test_tag}.pt"
            )
            if os.path.abspath(new_path) != os.path.abspath(best_path):
                if os.path.exists(new_path):
                    os.remove(new_path)
                if os.path.exists(best_path):
                    os.rename(best_path, new_path)
                self.best_checkpoints = [
                    (score, new_path if path == best_path else path)
                    for score, path in self.best_checkpoints
                ]
                if self.best_checkpoint_path == best_path:
                    self.best_checkpoint_path = new_path
                best_path = new_path

        test_result_data['seed'] = getattr(self, 'used_seed', None)
        if self.wandb_run is not None:
            for key, value in test_result_data.items():
                if self._is_number(value):
                    wandb.run.summary[f"test/{key}"] = float(value)
            if latency_stats:
                for key, value in latency_stats.items():
                    if self._is_number(value):
                        wandb.run.summary[f"test/{key}"] = float(value)
            if flops_stats:
                for key, value in flops_stats.items():
                    if self._is_number(value):
                        wandb.run.summary[f"test/{key}"] = float(value)
            run_url = getattr(self.wandb_run, 'url', None)
            wandb.finish()
        run_payload = {
            'run_label': self.run_label,
            'best': {
                'primary_metric': (
                    'pair_role_macro_f1'
                    if bool(getattr(self.config, 'pair_role_enable', False))
                    else (
                        'context_subtype_macro_f1'
                        if bool(getattr(self.config, 'context_subtype_enable', False))
                        else 'pair_f1'
                    )
                ),
                'val_primary_score': best_score,
                'val_pair_f1': best_score if not bool(getattr(self.config, 'pair_role_enable', False)) and not bool(getattr(self.config, 'context_subtype_enable', False)) else None,
                'best_epoch': best_iter,
                'best_checkpoint': best_path,
                'pair_decode': self.best_pair_decode,
                'context_subtype_thresholds': self.best_context_subtype_thresholds,
            },
            'metrics': {
                'train': train_metrics,
                'valid': valid_metrics,
                'test': test_result_data
            },
            'epoch_history': self.epoch_history,
            **run_constants
        }
        if latency_stats:
            run_payload['latency'] = latency_stats
        if flops_stats:
            run_payload['flops'] = flops_stats
        if fixed_base_decode_metrics is not None:
            run_payload['metrics_fixed_base_decode'] = fixed_base_decode_metrics
        if self.wandb_run is not None:
            run_payload['wandb'] = {
                'run_id': getattr(self.wandb_run, 'id', None),
                'run_name': getattr(self.wandb_run, 'name', None)
            }
        self._save_run_json(run_payload)
        return test_result_data

    def train_step(self, progress=None, epoch=0):
        self.model.train()
        losses = []
        component_sums = {}
        component_steps = 0

        for data in self.train_loader:
            # 将数据移到GPU
            data = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
            loss, _ = self.model(**data)
            losses.append(loss.item())
            latest_components = getattr(self.model, 'latest_loss_components', None)
            if isinstance(latest_components, dict) and latest_components:
                component_steps += 1
                for key, value in latest_components.items():
                    component_sums[key] = component_sums.get(key, 0.0) + float(value)

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.config.optimizer.step()
            if getattr(self.config, "scheduler", None) is not None:
                self.config.scheduler.step()
            self.config.optimizer.zero_grad(set_to_none=True)

            if progress is not None:
                desc = f"Epoch {epoch + 1}/{self.config.epoch_size} loss:{np.mean(losses):.4f}"
                progress.set_description(desc)
                progress.update(1)
        averaged_components = {}
        if component_steps:
            averaged_components = {
                key: value / component_steps
                for key, value in component_sums.items()
            }
        return (float(np.mean(losses)) if losses else 0.0), averaged_components

    def evaluate_step(self, dataLoader=None):
        self.model.eval()
        dataLoader = self.valid_loader if dataLoader is None else dataLoader

        for data in dataLoader:
            with torch.no_grad():
                # 将数据移到GPU
                data = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
                loss, output = self.model(**data)
                self.metrics_calculator.add_output(data, output)

        tune_decode = dataLoader is self.valid_loader
        decode_kwargs = self._pair_decode_kwargs(tune_decode=tune_decode, override_decode=(None if tune_decode else self.best_pair_decode))
        return self.metrics_calculator.report_score_decode(
            **decode_kwargs,
            tune_context_subtype_thresholds=bool(getattr(self.config, 'context_subtype_threshold_search', True)) and tune_decode,
            context_subtype_thresholds=None if tune_decode else self.best_context_subtype_thresholds,
        )

    def final_evaluate(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        score, res = self.evaluate_step(self.test_loader)
        return score, res

    def add_instance(self, score, res):
        self.scores.append(score)
        self.lines.append(res)


MyTrainer = PaceExtTrainer
