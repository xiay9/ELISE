#!/use/bin/env python


import os
import random

import torch
import numpy as np
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from configs.paths import CACHE_ROOT, DATA_ROOT, OUTPUT_ROOT


def _resolve_absolute_path(base_dir, path_value):
    path_text = str(path_value)
    if os.path.isabs(path_text):
        return path_text
    return os.path.join(base_dir, path_text)


def _dataset_slug(dataset_name):
    return str(dataset_name).strip().lower()


def update_config(config):
    data_root = config.get('data_root')
    if not data_root:
        data_root = str(DATA_ROOT)
    if not os.path.isabs(data_root):
        data_root = os.path.join(config.data_dir, data_root)
    config['data_root'] = data_root

    dataset_root = os.path.join(data_root, str(config.dataset_name))
    # Registry caches are read-only fallbacks; new caches belong to this project.
    if 'preprocessed_dir' not in config:
        config['preprocessed_dir'] = str(CACHE_ROOT / 'preprocessed' / str(config.dataset_name))
        config.setdefault('preprocessed_fallback_dir', os.path.join(dataset_root, 'preprocessed'))
    if 'preprocessed_dir' in config:
        config['preprocessed_dir'] = _resolve_absolute_path(config.data_dir, config['preprocessed_dir'])

    target_root = config.get('target_root')
    if target_root is not None:
        target_root = _resolve_absolute_path(config.data_dir, target_root)
        config['target_root'] = target_root

    if 'target_dir' in config and config.get('target_dir'):
        config['target_dir'] = _resolve_absolute_path(config.data_dir, config['target_dir'])
    else:
        if target_root is None:
            target_root = str(OUTPUT_ROOT / 'paceext')
            config['target_root'] = target_root
        target_subdir = config.get('target_subdir') or f"{_dataset_slug(config.dataset_name)}_base"
        config['target_dir'] = os.path.join(target_root, target_subdir)

    if not os.path.exists(config.preprocessed_dir):
        os.makedirs(config.preprocessed_dir)
    if not os.path.exists(config.target_dir):
        os.makedirs(config.target_dir)
    return config

def set_seed(seed):
    # 使用配置文件中的固定种子值
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    random.seed(seed)  # Python random module.
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    try:
        torch.use_deterministic_algorithms(True)
    except TypeError:
        torch.use_deterministic_algorithms(True)

    return seed  # 返回使用的种子


def load_params_bert(config, model, fold_data):
    no_decay = ['bias', 'LayerNorm.weight']
    bert_params = set(model.bert.parameters())
    other_params = list(set(model.parameters()) - bert_params)
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)], 'lr': float(config.bert_lr), 'weight_decay': float(config.weight_decay)},
        {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)], 'lr': float(config.bert_lr), 'weight_decay': 0.0},
        {'params': other_params, 'lr': float(config.learning_rate), 'weight_decay': float(config.weight_decay)},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, eps=float(config.adam_epsilon))

    total_steps = int(config.epoch_size) * int(len(fold_data))
    warmup_steps = int(getattr(config, 'warmup_steps', 0) or 0)
    if warmup_steps <= 0:
        warmup_proportion = float(getattr(config, 'warmup_proportion', 0.0) or 0.0)
        warmup_steps = int(total_steps * warmup_proportion)
    warmup_steps = max(0, min(warmup_steps, total_steps))

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    config.optimizer = optimizer
    config.scheduler = scheduler

    return config
