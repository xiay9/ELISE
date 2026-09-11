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

In short:

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
