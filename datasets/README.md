# ECF and MECAD data

This directory includes the exact text/annotation splits used by this project, stable dialogue-ID maps, and the twelve audio/video feature files bound to the primary TAV experiments. It contains 20 data files (approximately 215 MiB), plus this documentation and the checksum manifest.

| Dataset | Split | Dialogues | Utterances | Gold pairs |
|---|---|---:|---:|---:|
| ECF | train | 1,001 | 9,966 | 7,055 |
| ECF | valid | 112 | 1,087 | 866 |
| ECF | test | 261 | 2,566 | 1,873 |
| MECAD | train | 684 | 7,516 | 5,788 |
| MECAD | valid | 126 | 1,168 | 977 |
| MECAD | test | 179 | 1,832 | 1,312 |

These are complete source annotations; gold pairs outside the lower triangle are retained. The `valid` split is the validation split, not a second test split.

```text
datasets/
  ECF/
    dataset/{train,valid,test}.txt
    ecf_raw_to_doc_id.json
    cache/{audio,video}_features_{train,valid,test}.pt
  MECAD/
    {train,valid,test}_data_pair.json
    mecad_raw_to_doc_id.json
    cache/{audio,video}_features_{train,valid,test}.pt
  MANIFEST.json
```

The ID maps are JSON exports of the original maps. They preserve numeric encoder IDs and raw dialogue IDs. Feature `.pt` files retain their original bytes; they contain dictionaries of utterance feature vectors. The dataloader applies its existing missing-feature handling; the released feature hashes identify the input condition exactly. No extra `dev` copies are needed because the adapters also resolve `valid`.

Verify the data from the repository root:

```bash
python code/scripts/verify_datasets.py
```

Original sources and dataset terms:

- [ECF / MECPE](https://github.com/NUSTM/MECPE): Wang et al., *Multimodal Emotion-Cause Pair Extraction in Conversations*, IEEE Transactions on Affective Computing, DOI [10.1109/TAFFC.2022.3226559](https://doi.org/10.1109/TAFFC.2022.3226559). The upstream project specifies research-only use. Original multimedia is available through the upstream dataset distribution.
- [MECAD-MECTEC](https://github.com/redifinition/MECAD-MECTEC): the original MECAD dataset and annotation tools; see also [M3HG](https://github.com/redifinition/M3HG) for the accompanying research implementation and citation.

Please cite the dataset authors and follow their original terms. This repository does not assign a new license to third-party data or pretrained features. Original multimedia clips, unrelated feature variants, LLM-generated summaries and experiment-specific prediction caches are not included.
