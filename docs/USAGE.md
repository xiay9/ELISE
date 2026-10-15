# Running ELISE

Run commands from the repository root with Python 3.12. Reuse an existing environment where practical. Install `requirements-encoder.txt` for encoder training/export and `requirements-verifier.txt` for semantic verification; the router/gate workflow needs only `requirements-core.txt`.

```bash
python -m pip install -r requirements-encoder.txt
python -m pip install -r requirements-verifier.txt
export PYTHONPATH="$PWD/code:$PWD/code/elise"
export ELISE_ARTIFACT_ROOT=/path/to/large-storage/elise
export HF_HOME="$ELISE_ARTIFACT_ROOT/huggingface"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
python code/scripts/verify_datasets.py
```

`ELISE_DATA_ROOT` defaults to the included `datasets/` directory. Generated preprocessing caches, checkpoints and results go under `ELISE_ARTIFACT_ROOT`; use an external storage location with sufficient capacity. Preserve the allocator's `CUDA_VISIBLE_DEVICES`. Training/export and LLM inference require CUDA; the scikit-learn router and gate fitting run on CPUs. Assign independent dataset/modality/seed runs to separate allocated logical GPU indices when several GPUs are available. Set CPU thread/worker counts to the allocation rather than the machine's total core count.

## 1. Train and export fixed encoders

Four configuration templates cover ECF/MECAD and text (`t`)/text–audio–video (`tav`). The primary seeds are 42, 345 and 456. The text backbone identifier can be replaced by a local RoBERTa snapshot in each YAML file. Keep the same snapshot across seeds.

Example for one ECF text run:

```bash
python code/scripts/run_paceext.py \
  --config configs/encoders/ecf_t.yaml --seed 42 \
  --output-dir "$ELISE_ARTIFACT_ROOT/encoders/ecf_t_seed42"

python code/elise/export_encoder.py \
  --config configs/encoders/ecf_t.yaml --seed 42 \
  --checkpoint /path/to/best_checkpoint.pt \
  --output "$ELISE_ARTIFACT_ROOT/pairs"
```

Use the best validation checkpoint reported by training. Repeat for all three seeds and the other configurations. `--device cuda:0` on the exporter denotes logical device zero within the allocation. Exported filenames follow `pairs/ecf/rpcl_t_seed42_valid.csv` and `pairs/ecf/rpcl_t_seed42_test.csv`; multimodal files use `rpcl_tav`.

## 2. Acquire validation semantic evidence

Validation policy selection evaluates multiple operating thresholds. Build the union of validation candidates across seeds, then run the action-blind verifier:

```bash
python code/elise/build_requests.py \
  --dataset ECF --split valid \
  --pairs "$ELISE_ARTIFACT_ROOT/pairs/ecf/"rpcl_t_seed*_valid.csv \
  --output "$ELISE_ARTIFACT_ROOT/requests/ecf_valid_requests.jsonl"

python code/elise/run_llm_verifier_vllm.py \
  --input "$ELISE_ARTIFACT_ROOT/requests/ecf_valid_requests.jsonl" \
  --output "$ELISE_ARTIFACT_ROOT/responses/ecf_valid_results.jsonl" \
  --model /path/to/Qwen3.6-27B-snapshot \
  --max-model-len 2048 --max-tokens 128 --batch-size 64 \
  --structured-output-schema compact_verification --no-enable-thinking
```

Use Qwen3.6-27B revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`, listed in `configs/verifier.json`. The model weights are obtained separately. Consult the [official model card](https://huggingface.co/Qwen/Qwen3.6-27B) and [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.6-27B) for serving support. The runner uses the official offline `LLM` and structured-output APIs. Its `--tensor-parallel-size` is configurable; the recorded paper setup used one A100-80GB. Select memory settings appropriate to the allocated GPUs.

`--preflight-only` validates request identities and schemas without loading vLLM. Inference writes contract/runtime sidecars beside the JSONL output. Prompts contain the complete dialogue and speaker identities, with no proposed action, encoder scores or gold emotion labels. Duplicate requests across seeds are merged before inference.

## 3. Fit the validation policy

```bash
python code/elise/run_pointwise.py fit \
  --dataset ECF --chain text \
  --pair-root "$ELISE_ARTIFACT_ROOT/pairs" \
  --responses "$ELISE_ARTIFACT_ROOT/responses/ecf_valid_results.jsonl" \
  --output "$ELISE_ARTIFACT_ROOT/policies/ecf_text"
```

Use `--chain multimodal` for the TAV chain. Fit requires all three validation seeds. It writes `validation.json` with input hashes, policy and cutoffs, and a checksummed `model.pkl` containing fitted estimators and validation reference distributions. Use only locally produced, trusted model bundles. Existing output directories are rejected to keep each fit distinct.

The text chain uses the selected seven-leaf histogram gradient boosting router; the TAV chain uses the direct action-oriented encoder score. Both use separate linear add/delete gates. The release retains the selected primary architectures and the original Pointwise rate search. It does not rerun the earlier architecture-selection study.

## 4. Route test candidates before verification

```bash
python code/elise/run_pointwise.py route \
  --bundle "$ELISE_ARTIFACT_ROOT/policies/ecf_text" \
  --pair-root "$ELISE_ARTIFACT_ROOT/pairs" \
  --output "$ELISE_ARTIFACT_ROOT/routed/ecf_text"

python code/elise/build_requests.py \
  --dataset ECF --split test --routed-only \
  --pairs "$ELISE_ARTIFACT_ROOT/routed/ecf_text/"seed*.csv \
  --output "$ELISE_ARTIFACT_ROOT/requests/ecf_test_requests.jsonl"

python code/elise/run_llm_verifier_vllm.py \
  --input "$ELISE_ARTIFACT_ROOT/requests/ecf_test_requests.jsonl" \
  --output "$ELISE_ARTIFACT_ROOT/responses/ecf_test_results.jsonl" \
  --model /path/to/Qwen3.6-27B-snapshot \
  --max-model-len 2048 --max-tokens 128 --batch-size 64 \
  --structured-output-schema compact_verification --no-enable-thinking
```

## 5. Apply frozen edits and evaluate

```bash
python code/elise/run_pointwise.py predict \
  --bundle "$ELISE_ARTIFACT_ROOT/policies/ecf_text" \
  --pair-root "$ELISE_ARTIFACT_ROOT/pairs" \
  --responses "$ELISE_ARTIFACT_ROOT/responses/ecf_test_results.jsonl" \
  --output "$ELISE_ARTIFACT_ROOT/predictions/ecf_text"
```

The model never fits on test labels. Missing, malformed, uncertain or action-inconsistent semantic responses preserve the encoder decision. Each candidate uses frozen route/gate thresholds and the validation reference distribution. Final metrics use all gold pairs from the source dataset, including relations outside the encoder's editable lower triangle.

Repeat for MECAD and the multimodal chain. The main workflow generates new predictions and metrics; it does not require any private historical result table, fitted policy, encoder checkpoint or LLM-response archive.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
```

The tests use fictional examples and mocked model responses. They cover inference-pool invariance, label independence, grouped validation folds, fail-closed responses, complete-gold scoring, request contracts and a small validation-fit/frozen-test workflow.
