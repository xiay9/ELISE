"""Build action-blind requests from exported candidates or frozen routed candidates."""
import argparse
import json
from pathlib import Path

import pandas as pd
from data_io import DATA_ROOT, dialogue_index, request_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=Path, nargs='+', required=True)
    parser.add_argument('--dataset', choices=['ECF', 'MECAD'], required=True)
    parser.add_argument('--split', choices=['valid', 'test'], required=True)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--routed-only', action='store_true')
    args = parser.parse_args()
    records = dialogue_index(args.dataset, args.split, args.data_root)
    requests = {}
    for path in args.pairs:
        frame = pd.read_csv(path, dtype={'dialogue_id': str})
        if set(frame['split']) != {args.split} or set(frame['dataset']) != {args.dataset}:
            raise ValueError(f'Dataset/split mismatch: {path}')
        if args.routed_only:
            frame = frame[frame['routed'].astype(int) == 1]
        for row in frame.itertuples():
            record = records[str(row.dialogue_id)]
            request = request_for(record, args.dataset, args.split, row.emotion_turn, row.cause_turn)
            requests[request['request_id']] = request
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        for key in sorted(requests):
            handle.write(json.dumps(requests[key], ensure_ascii=False) + '\n')
    print(f'Wrote {len(requests)} unique action-blind requests to {args.output}')


if __name__ == '__main__':
    main()
