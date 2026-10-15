#!/usr/bin/env python

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
import random
import string
import sys
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_CPP_MIN_VLOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("WANDB_CONSOLE", "off")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = CODE_ROOT.parent
for path in (CODE_ROOT, REPO_ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

logging.basicConfig(level=logging.ERROR, force=True)
warnings.filterwarnings("ignore")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PaceExt with the unified dataset layout.")
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "encoders" / "ecf_t.yaml"),
        help="YAML config path.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Optional dataset override, e.g. ECF, MEC4, MECAD, ConvECPE, IemoMECP.",
    )
    parser.add_argument(
        "--epoch-size",
        type=int,
        default=None,
        help="Optional epoch override for smoke runs.",
    )
    parser.add_argument(
        "--dialogue-fraction",
        type=float,
        default=None,
        help="Optional fraction of dialogues per split, e.g. 0.01 for 1%% smoke runs.",
    )
    parser.add_argument(
        "--train-dialogue-fraction",
        type=float,
        default=None,
        help="Optional fraction of train dialogues only.",
    )
    parser.add_argument(
        "--valid-dialogue-fraction",
        type=float,
        default=None,
        help="Optional fraction of valid dialogues only.",
    )
    parser.add_argument(
        "--test-dialogue-fraction",
        type=float,
        default=None,
        help="Optional fraction of test dialogues only.",
    )
    parser.add_argument(
        "--use-audio",
        dest="use_audio",
        action="store_true",
        default=None,
        help="Enable audio modality. Defaults to the config value.",
    )
    parser.add_argument(
        "--no-audio",
        dest="use_audio",
        action="store_false",
        help="Disable audio modality by zeroing audio features in the dataloader.",
    )
    parser.add_argument(
        "--use-video",
        dest="use_video",
        action="store_true",
        default=None,
        help="Enable video modality. Defaults to the config value.",
    )
    parser.add_argument(
        "--no-video",
        dest="use_video",
        action="store_false",
        help="Disable video modality by zeroing video features in the dataloader.",
    )
    parser.add_argument(
        "--use-text",
        dest="use_text",
        action="store_true",
        default=None,
        help="Enable text features. Defaults to the config value.",
    )
    parser.add_argument(
        "--no-text",
        dest="use_text",
        action="store_false",
        help="Disable text modality by zeroing text features before multimodal fusion.",
    )
    parser.add_argument(
        "--use-audio-summary",
        dest="use_audio_summary",
        action="store_true",
        default=None,
        help="Append audio summary jsonl content to each utterance.",
    )
    parser.add_argument(
        "--no-audio-summary",
        dest="use_audio_summary",
        action="store_false",
        help="Disable audio summary augmentation.",
    )
    parser.add_argument(
        "--use-video-summary",
        dest="use_video_summary",
        action="store_true",
        default=None,
        help="Append video summary jsonl content to each utterance.",
    )
    parser.add_argument(
        "--no-video-summary",
        dest="use_video_summary",
        action="store_false",
        help="Disable video summary augmentation.",
    )
    parser.add_argument("--audio-summary-train-path", default=None, help="Override train audio summary jsonl path.")
    parser.add_argument("--audio-summary-valid-path", default=None, help="Override valid audio summary jsonl path.")
    parser.add_argument("--audio-summary-test-path", default=None, help="Override test audio summary jsonl path.")
    parser.add_argument("--video-summary-train-path", default=None, help="Override train video summary jsonl path.")
    parser.add_argument("--video-summary-valid-path", default=None, help="Override valid video summary jsonl path.")
    parser.add_argument("--video-summary-test-path", default=None, help="Override test video summary jsonl path.")
    parser.add_argument("-cd", "--cuda-index", type=int, default=None, help="CUDA device index.")
    parser.add_argument("--seed", nargs="+", default=None, help="Space separated seeds, e.g. --seed 42 123.")
    parser.add_argument("--output-dir", default=None, help="Absolute output directory override.")
    parser.add_argument("--wandb-mode", default=None, help="Optional wandb mode override, e.g. disabled.")
    parser.add_argument(
        "--eng-metrics",
        dest="eng_metrics",
        action="store_true",
        default=None,
        help="Enable engineering metrics collection. Defaults to the config value.",
    )
    parser.add_argument(
        "--no-eng-metrics",
        dest="eng_metrics",
        action="store_false",
        help="Disable engineering metrics collection.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional dataloader worker override.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional wandb run name override.",
    )
    parser.add_argument(
        "--pair-role-task",
        choices=["none", "3class", "pair_role", "cause_context_none"],
        default=None,
        help="Train pair roles: emo_cause / emo_context / non_pair.",
    )
    parser.add_argument(
        "--pair-role-labels-path",
        default=None,
        help="CSV path with split/dialogue_id/target_emotion_turn/cause_turn/taxonomy_label columns.",
    )
    parser.add_argument(
        "--pair-role-loss-weight",
        type=float,
        default=None,
        help="Optional multiplier for the pair-role loss.",
    )
    parser.add_argument(
        "--pair-role-related-loss-weight",
        type=float,
        default=None,
        help="Optional auxiliary loss weight for related(emo_cause+emo_context) vs non_pair.",
    )
    parser.add_argument(
        "--context-subtype-task",
        choices=["none", "4tag", "context_subtype"],
        default=None,
        help="Train pair-level context subtype auxiliary tags for emo_context pairs.",
    )
    parser.add_argument(
        "--context-subtype-loss-weight",
        type=float,
        default=None,
        help="Optional multiplier for the context-subtype loss.",
    )
    parser.add_argument("--verifiability-task", dest="pair_role_task", choices=["none", "3class"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--verifiability-labels-path", dest="pair_role_labels_path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--verifiability-loss-weight", dest="pair_role_loss_weight", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cause-role-task", dest="context_subtype_task", choices=["none", "4tag", "5tag", "5class", "cause5"], default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cause-role-loss-weight", dest="context_subtype_loss_weight", type=float, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def _resolve_overrides(args: argparse.Namespace, seed: int | None) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if args.dataset_name is not None:
        overrides["dataset_name"] = args.dataset_name
    if args.epoch_size is not None:
        overrides["epoch_size"] = args.epoch_size
    if args.dialogue_fraction is not None:
        overrides["dialogue_fraction"] = args.dialogue_fraction
    if args.train_dialogue_fraction is not None:
        overrides["train_dialogue_fraction"] = args.train_dialogue_fraction
    if args.valid_dialogue_fraction is not None:
        overrides["valid_dialogue_fraction"] = args.valid_dialogue_fraction
    if args.test_dialogue_fraction is not None:
        overrides["test_dialogue_fraction"] = args.test_dialogue_fraction
    if args.use_audio is not None:
        overrides["use_audio"] = args.use_audio
    if args.use_video is not None:
        overrides["use_video"] = args.use_video
    if args.use_text is not None:
        overrides["use_text"] = args.use_text
    if args.use_audio_summary is not None:
        overrides["use_audio_summary"] = args.use_audio_summary
    if args.use_video_summary is not None:
        overrides["use_video_summary"] = args.use_video_summary
    if args.audio_summary_train_path is not None:
        overrides["audio_summary_train_path"] = args.audio_summary_train_path
    if args.audio_summary_valid_path is not None:
        overrides["audio_summary_valid_path"] = args.audio_summary_valid_path
    if args.audio_summary_test_path is not None:
        overrides["audio_summary_test_path"] = args.audio_summary_test_path
    if args.video_summary_train_path is not None:
        overrides["video_summary_train_path"] = args.video_summary_train_path
    if args.video_summary_valid_path is not None:
        overrides["video_summary_valid_path"] = args.video_summary_valid_path
    if args.video_summary_test_path is not None:
        overrides["video_summary_test_path"] = args.video_summary_test_path
    if args.cuda_index is not None:
        overrides["cuda_index"] = args.cuda_index
    if args.wandb_mode is not None:
        overrides["wandb_mode"] = args.wandb_mode
    if args.eng_metrics is not None:
        overrides["eng_metrics"] = args.eng_metrics
    if args.num_workers is not None:
        overrides["num_workers"] = args.num_workers
    if args.wandb_run_name is not None:
        overrides["wandb_run_name"] = args.wandb_run_name
    if args.pair_role_task is not None:
        overrides["pair_role_task"] = args.pair_role_task
    if args.pair_role_labels_path is not None:
        overrides["pair_role_labels_path"] = args.pair_role_labels_path
    if args.pair_role_loss_weight is not None:
        overrides["pair_role_loss_weight"] = args.pair_role_loss_weight
    if args.pair_role_related_loss_weight is not None:
        overrides["pair_role_related_loss_weight"] = args.pair_role_related_loss_weight
    if args.context_subtype_task is not None:
        overrides["context_subtype_task"] = args.context_subtype_task
    if args.context_subtype_loss_weight is not None:
        overrides["context_subtype_loss_weight"] = args.context_subtype_loss_weight
    if seed is not None:
        overrides["seed"] = seed
    return overrides


def _run_single(config_path: str | Path, args: argparse.Namespace, seed: int | None) -> None:
    import torch
    import transformers

    from paceext import PaceExtModel, PaceExtTrainer, load_paceext_config, make_paceext_datamodule
    from paceext.core.utils import load_params_bert, set_seed

    transformers.logging.set_verbosity_error()
    config = load_paceext_config(config_path, _resolve_overrides(args, seed))
    if args.output_dir is not None:
        config.target_dir = str(Path(args.output_dir).expanduser().resolve())
        Path(config.target_dir).mkdir(parents=True, exist_ok=True)

    random_str = "".join(random.sample(string.ascii_letters + string.digits, 8))
    config.save_name = f"bert_{random_str}_{config.seed}_{{}}.pt"

    used_seed = set_seed(config.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Encoder training requires an allocated CUDA GPU.")
    config.device = torch.device(f"cuda:{config.cuda_index}")

    print(f"seed: {used_seed}")
    print(f"dataset: {config.dataset_name}")
    print(f"backbone: {config.backbone}")
    print(f"batch_size: {config.batch_size}")
    print(f"use_audio: {bool(getattr(config, 'use_audio', True))}")
    print(f"use_video: {bool(getattr(config, 'use_video', True))}")
    print(f"use_text: {bool(getattr(config, 'use_text', True))}")
    print(f"use_audio_summary: {bool(getattr(config, 'use_audio_summary', False))}")
    print(f"use_video_summary: {bool(getattr(config, 'use_video_summary', False))}")
    print(f"pair_role_task: {getattr(config, 'pair_role_task', 'none')}")
    print(f"pair_role_labels_path: {getattr(config, 'pair_role_labels_path', None)}")
    print(f"context_subtype_task: {getattr(config, 'context_subtype_task', 'none')}")
    print(f"target_dir: {config.target_dir}")
    print(f"timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.backbone,
        padding_side="right",
        use_fast=False,
    )
    train_loader, valid_loader, test_loader, config = make_paceext_datamodule(config, tokenizer)
    if bool(getattr(config, "pair_role_enable", False)):
        print(f"pair_role_classes: {list(getattr(config, 'pair_role_class_names', []))}")
        print(f"pair_role_train_counts: {getattr(config, 'pair_role_train_label_counts', {})}")
        print(f"pair_role_all_counts: {getattr(config, 'pair_role_all_label_counts', {})}")
        print(f"pair_role_class_weights: {getattr(config, 'pair_role_class_weights', [])}")
        print(f"pair_role_related_loss_weight: {getattr(config, 'pair_role_related_loss_weight', 0.0)}")
        print(f"pair_role_related_class_weights: {getattr(config, 'pair_role_related_class_weights', [])}")
    if bool(getattr(config, "context_subtype_enable", False)):
        print(f"context_subtype_classes: {list(getattr(config, 'context_subtype_class_names', []))}")
        print(f"context_subtype_train_item_count: {getattr(config, 'context_subtype_train_item_count', 0)}")
        print(f"context_subtype_all_item_count: {getattr(config, 'context_subtype_all_item_count', 0)}")
        print(f"context_subtype_train_counts: {getattr(config, 'context_subtype_train_label_counts', {})}")
        print(f"context_subtype_all_counts: {getattr(config, 'context_subtype_all_label_counts', {})}")
        print(f"context_subtype_pos_weights: {getattr(config, 'context_subtype_pos_weights', [])}")
    model = PaceExtModel(config, tokenizer).to(config.device)
    config = load_params_bert(config, model, train_loader)
    trainer = PaceExtTrainer(model, config, train_loader, valid_loader, test_loader)
    trainer.used_seed = used_seed
    trainer.train()


def main() -> None:
    args = _parse_args()
    if args.seed:
        for raw_seed in args.seed:
            _run_single(args.config, args, int(raw_seed))
        return
    _run_single(args.config, args, None)


if __name__ == "__main__":
    main()
