# ELISE: Selective Revision with LLM Verification for Multimodal Emotion–Cause Pair Extraction

Official implementation of **ELISE (Encoder–LLM Integrated Selective Editing)**, a post-encoder selective revision framework for multimodal emotion–cause pair extraction (MECPE).

ELISE keeps a fixed structured encoder as the default predictor and invokes a large language model only for selected candidate relations. LLM judgments are treated as auxiliary semantic evidence rather than replacement labels, and a separately trained authorization gate determines whether a proposed local edit should be applied.

## Overview

Multimodal emotion–cause pair extraction aims to identify causal links between emotional utterances and their causes in conversations.

Strong structured encoders still produce false-positive relations and miss valid causes. Applying an LLM to every candidate pair is expensive and may also modify predictions that are already correct.

ELISE addresses this problem with a three-stage post-encoder revision pipeline:

1. **Candidate prioritization**  
   Encoder-side diagnostics identify predictions that are most worth reviewing.

2. **Action-blind semantic verification**  
   A text-only LLM verifier evaluates whether a candidate cause supports the target emotional state. The verifier does **not** observe the encoder decision, encoder scores, or contemplated edit.

3. **Edit authorization**  
   Action-specific gates combine encoder, routing, and verifier evidence to estimate whether a proposed flip is likely to correct the base prediction.

Only routed, action-consistent, and gate-authorized candidates are modified. All other encoder predictions are preserved.

```text
Fixed MECPE encoder
        |
        v
Candidate prioritization
        |
        v
Action-blind LLM verification
        |
        v
Edit authorization gate
        |
        v
Sparse authorized edits
        |
        v
Final pair matrix
```

## Code and data

This release provides the primary **Pointwise ELISE** workflow for ECF and MECAD with text and multimodal fixed encoders. Routing and authorization thresholds are selected on validation data and applied independently to test candidates.

| Component | Location |
|---|---|
| Fixed encoder training and data loading | `code/paceext/`, `code/data/` |
| Encoder diagnostics export | `code/elise/export_encoder.py` |
| Action-blind requests and verifier | `code/elise/build_requests.py`, `code/elise/run_llm_verifier_vllm.py` |
| Router, gates and Pointwise policy | `code/elise/router.py`, `code/elise/text_gate.py`, `code/elise/tav_gate.py`, `code/elise/pointwise.py` |
| Validation fit, test routing and frozen prediction | `code/elise/run_pointwise.py` |
| Four encoder configurations and fixed verifier settings | `configs/` |
| Dataset splits, dialogue-ID maps and audio/video features | `datasets/` |
| Contract and workflow tests | `tests/` |

The included data comprise all six train/validation/test splits, two dialogue-ID maps and twelve feature tensor files (about 215 MiB). See [dataset documentation](datasets/README.md) for upstream sources, terms, statistics and checksums. Pretrained weights and original multimedia clips are obtained separately.

## Getting started

Use Python 3.12. The [running guide](docs/USAGE.md) covers encoder training, diagnostic export, validation verification, policy fitting, selective test verification and evaluation. Start by checking the included data:

```bash
python code/scripts/verify_datasets.py
```

The CPU router/gate workflow requires `requirements-core.txt`. Encoder and verifier dependencies are listed separately in `requirements-encoder.txt` and `requirements-verifier.txt`. CUDA is required for neural model execution.

- [Usage and commands](docs/USAGE.md)
- [Input and feature contracts](docs/DATA_FORMAT.md)
- [Reproduction settings and validation scope](docs/REPRODUCIBILITY.md)
- [Dataset sources and checksums](datasets/README.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

This release focuses on the primary method. Manuscript sources, paper PDFs, policy-comparison and attribution-control experiment files, historical result tables, trained checkpoints and verifier-response archives are not part of this code release.
