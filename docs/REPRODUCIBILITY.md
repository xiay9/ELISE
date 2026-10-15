# Implementation and reproduction scope

This release contains the primary Pointwise method and the fixed encoder implementation, with ECF/MECAD dataset splits and the recorded feature tensors. Shared numerical primitives were extracted from the experimental code; filesystem-specific orchestration was replaced with explicit CLI paths. The original encoder implementation retains the internal package name `paceext` for checkpoint compatibility.

The primary router/gate code preserves the feature order, estimator settings, validation grouping, rate-to-cutoff conversion, inclusive threshold comparisons and action-consistency rule. The gate is fitted on action-consistent validation examples. OOF gate folds group all seed copies of a dialogue. Router OOF scoring follows the original per-encoder-seed procedure. Gate training uses the original validation routing coordinates; test-time Pointwise scoring evaluates the routing coordinate against the frozen validation score distribution.

The validation operating-policy search uses the original encoder candidate support. Reported final Pair F1 is recomputed against the complete source gold relation set. Out-of-support gold relations remain false negatives. The release fixes the selected primary model families and does not include the historical architecture selection, policy-comparison experiments, no-verifier controls or action-only controls.

## Recorded settings

- Datasets: ECF and MECAD; encoder seeds: 42, 345, 456.
- Primary support: ordered pairs with cause turn no later than emotion turn.
- Text router: 33 encoder diagnostic fields, seven-leaf histogram gradient boosting.
- TAV router: `pair_score` for additions and `non_pair_score` for deletions.
- Authorization: 19 fields, median imputation, standardization, balanced logistic regression, `C=1`, `solver=liblinear`, at most 2,000 iterations.
- Validation folds: five shuffled `StratifiedGroupKFold` folds, random state 20260724.
- Verifier: Qwen3.6-27B, fixed revision in `configs/verifier.json`, text-only action-blind prompt, compact JSON schema, temperature 0, top-p 1, 128 output tokens, 2,048 context tokens, disabled thinking.
- Recorded numerical stack: NumPy 1.26.4, pandas 2.3.3 and scikit-learn 1.8.0. The dependency file permits NumPy 2.x for development; use NumPy 1.26.4 when targeting the recorded environment.
- Recorded verifier stack: vLLM 0.19.1 and Transformers 5.7.0.

## Assets and verification

`datasets/MANIFEST.json` binds every released split, ID map and feature file by SHA-256. The twelve feature tensor hashes were checked against the primary TAV run's recorded feature audit. Each vector has 768 dimensions in the inspected validation tensors. Original audio/video clips and pretrained model weights are obtained separately from their upstream sources; encoder training consumes the included feature vectors.

The release preparation verified all six dataset splits, the data-file hashes, source parity of extracted primitives, CLI startup, encoder-module imports, and unit/integration contracts. It did not retrain the encoders or rerun the 27B verifier. New full runs can differ with hardware, numerical-library versions or pretrained-model revisions; the release does not promise bit-identical paper scores from an unpinned environment.

Paper sources/PDFs, experiment-result CSV/JSON files, generated policies/models, verification request/response archives, logs and manuscript table-generation code are outside this source release.
