# Input and feature contracts

The exporter produces one CSV per dataset, encoder modality, seed and split. The main workflow expects `rpcl_{t,tav}_seed{42,345,456}_{valid,test}.csv` under `PAIR_ROOT/{ecf,mecad}/`.

Each row identifies an ordered emotion-turn/cause-turn pair. Turn indices are one-based. `dialogue_id` is the stable numeric encoder ID; the included JSON maps bind it to the source raw dialogue ID. `pair_key` identifies the dataset, split and pair.

Required identity fields are `dataset`, `split`, `seed`, `modality`, `candidate_scope`, `dialogue_id`, `emotion_turn`, `cause_turn`, `pair_key`, and `pred_pair`. The primary support is `lower_triangle`. Validation also requires binary `gold_pair`. The inference loader discards supplied gold columns before router/gate scoring. Complete gold annotations are loaded separately for final evaluation.

Encoder exports include raw and distance-adjusted probabilities, node probabilities, margins, entropy, row statistics and dialogue counts. The public exporter preserves the original diagnostic definitions. The router loader derives `row_pred_fraction`, `dialogue_pred_fraction` and `distance_abs` from those diagnostics.

## Text-router fields (ordered)

1. `pair_score`
2. `non_pair_score`
3. `pair_margin`
4. `max_prob`
5. `margin_top2`
6. `distance`
7. `row_rank`
8. `row_size`
9. `row_pred_count`
10. `raw_pair_score`
11. `raw_pair_margin`
12. `raw_max_prob`
13. `raw_margin_top2`
14. `raw_logit_margin`
15. `emotion_score`
16. `cause_score`
17. `pair_margin_to_threshold`
18. `emotion_margin_to_gate`
19. `cause_margin_to_gate`
20. `min_gate_margin`
21. `gate_score_product`
22. `cell_entropy`
23. `distance_prior`
24. `row_rank_fraction`
25. `row_pred_fraction`
26. `row_top_score`
27. `row_mean_score`
28. `row_std_score`
29. `row_score_gap_to_top`
30. `row_mean_entropy`
31. `dialogue_pred_fraction`
32. `pair_threshold_pass`
33. `gate_pass`

## Authorization-gate fields (ordered)

| Position | Text chain | TAV chain |
|---:|---|---|
| 1 | `llm_confidence` | `llm_confidence` |
| 2 | `selector_score` | `router_score` |
| 3 | `selector_rank_fraction` | `router_rank_fraction` |
| 4 | `pair_score` | `pair_score` |
| 5 | `pair_margin` | `pair_margin` |
| 6 | `raw_pair_margin` | `raw_pair_margin` |
| 7 | `emotion_score` | `emotion_score` |
| 8 | `cause_score` | `cause_score` |
| 9 | `min_gate_margin` | `min_gate_margin` |
| 10 | `cell_entropy` | `cell_entropy` |
| 11 | `distance` | `distance` |
| 12 | `distance_abs` | `distance_abs` |
| 13 | `row_rank_fraction` | `row_rank_fraction` |
| 14 | `row_score_gap_to_top` | `row_score_gap_to_top` |
| 15 | `row_mean_entropy` | `row_mean_entropy` |
| 16 | `evidence_count` | `evidence_count` |
| 17 | `evidence_has_target` | `evidence_has_target` |
| 18 | `evidence_has_cause` | `evidence_has_cause` |
| 19 | `evidence_span` | `evidence_span` |

The TAV router uses the encoder pair score for additions and the non-pair score for deletions. It does not fit a second text router.

For Pointwise inference, the rank-fraction field is the upper-tail empirical coordinate of the raw routing score in the frozen validation action pool. It is not recomputed from the current test batch. Evidence indices are parsed as integers, deduplicated and sorted before computing count, target/cause indicators and span.

## Semantic responses

The verifier returns exactly `verdict`, `confidence` and `evidence_turns`. `verdict` is one of `support_pair`, `reject_pair`, or `uncertain`; confidence is in [0, 1]; evidence turns are integer indices.

Request JSONL includes a schema, prompt version, request ID, semantic pair key and SHA-256 of the messages. The inference runner returns the corresponding metadata and strict JSON response, with contract/runtime sidecars. The primary workflow rebuilds expected prompt identities from the included dataset, checks those bindings, and joins responses by raw dialogue ID and pair indices. Missing or invalid semantic evidence leaves the base decision unchanged. A response for another split, prompt or pair is rejected.

Request files, response files, fitted model bundles and prediction CSVs are generated at run time and are not included in the repository.
