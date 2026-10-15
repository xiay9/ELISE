"""Export the primary lower-triangle diagnostics from a trained encoder."""
import argparse
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=[42, 345, 456], required=True)
    parser.add_argument('--device', default='cuda:0', help='Logical index within allocated GPU visibility')
    parser.add_argument('--splits', nargs='+', choices=['valid', 'test'], default=['valid', 'test'])
    args = parser.parse_args()
    import torch
    from transformers import AutoTokenizer
    from paceext import PaceExtModel, load_paceext_config, make_paceext_datamodule
    from paceext.core.utils import set_seed
    from encoder_features import collect_split, normalize_decode
    if not torch.cuda.is_available() or not args.device.startswith('cuda:'):
        raise RuntimeError('Encoder export requires an allocated CUDA GPU')
    config = load_paceext_config(args.config, {'seed': args.seed, 'num_workers': 0, 'wandb_mode': 'disabled'})
    config.device = torch.device(args.device)
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.backbone, padding_side='right', use_fast=False)
    _, valid, test, config = make_paceext_datamodule(config, tokenizer)
    model = PaceExtModel(config, tokenizer).to(config.device)
    checkpoint = torch.load(args.checkpoint, map_location=config.device, weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    decode = normalize_decode(checkpoint['pair_decode'])
    modality = 'tav' if config.use_audio and config.use_video else 't'
    destination = args.output / config.dataset_name.lower()
    destination.mkdir(parents=True, exist_ok=True)
    for split, loader in [('valid', valid), ('test', test)]:
        if split not in args.splits:
            continue
        table, _ = collect_split(model=model, loader=loader, config=config,
                                dataset=config.dataset_name, split=split, decode=decode,
                                model_name='rpcl', seed=args.seed)
        table['modality'] = modality
        table['run_name'] = f'{config.dataset_name.lower()}_rpcl_{modality}_seed{args.seed}'
        output = destination / f'rpcl_{modality}_seed{args.seed}_{split}.csv'
        with output.open('x') as handle:
            table.to_csv(handle, index=False, float_format='%.17g')
        print(output)


if __name__ == '__main__':
    main()
