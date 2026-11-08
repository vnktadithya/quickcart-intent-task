# CLAUDE.md — QuickCart intent task

## Context
- Take-home for the Yellow.ai Applied ML Intern role. Task spec: README.md in this repo. Read it fully before any work.
- Goal: improve the whole intent-extraction system (taxonomy, data, evaluation, model) and prove every change with numbers.
- Deadline: submission ready by 7 pm Nov 8. Discussion round Nov 10: Adi must be able to explain every line and decision.
- Implementation plan: phases 0–8 below. Work them in order.

## How we work
1. One phase at a time. Do not start the next phase until Adi says so.
2. Before writing code or running anything, state in 3–5 lines: what you will do, why, and which file it produces. Wait for "go".
3. After each phase:
   - write results to `results/` (CSV, one row per experiment or per class, clear column names, before/after columns)
   - update `NOTES.md` (what was done, why, what the numbers mean, concepts in plain words, anything that went wrong)
   - add an entry to `DECISIONS.md` for every choice made
   - commit: `phaseN: <what changed>`
4. Explain concepts at the depth of "why it works", not just "what it is". No filler.
5. If Adi asks "why", answer with the evidence (number + file), not with popularity.

## Hard rules
- Never edit `data/test.csv`. Never train, tune or select on test. Test is used once, in Phase 7, to write predictions.
- Never try to infer test labels.
- Do not modify `baseline/`. Fixed versions go in `src/`, so the original bugs stay visible for before/after.
- Every reported number must come from a script in this repo that re-runs end to end. Fixed random seeds.
- Model selection uses the checked dev subset (`data/dev_checked.csv`) once it exists. Silver-dev numbers are reported beside it, never used alone to choose.
- Always report macro-F1 (`average="macro"`) and per-class P/R/F1 with `labels=` in taxonomy order.
- Hosted LLMs: open-source models only, offline only (labelling/auditing), only data from this repo. Save raw LLM outputs to `results/llm_raw/`. Adi verifies every row of the checked set himself.
- Final system must run on 4 CPU cores, 15 GB RAM, one message at a time.

## Latency measurement (use exactly this)
- `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `torch.set_num_threads(4)` where torch is used.
- One message per call, 20 warm-up calls, then 500 timed calls on dev messages. Report p50 and p95 ms and the CPU model (`lscpu`).
- Write to `results/latency.csv`: model, p50_ms, p95_ms, hardware, threads, notes.

## Phases
0. Setup: venv, run baseline, first commit "starter as received".
1. Fix evaluation: `src/evaluate.py` (label order, true macro-F1, labeled confusion matrix, slices: length, Hinglish, multi-request, contains-taxonomy-example). Re-score starter → `results/baseline_fixed.csv`, `results/eval_bugs.md`.
2. Label noise: 5-fold CV disagreement on train; stratified 250-row dev sample pre-labelled by an open-source LLM using the 8 README rules, then checked by Adi → `data/dev_checked.csv`, `results/label_noise.csv` (agreement + 95% Wilson interval, error types).
3. Taxonomy v2: remove duplicate/rule-breaking examples, fill `promo_code_issue` and `account_help`, intent-distance matrix vs confusion → `data/taxonomy_v2.json`, `results/taxonomy_before_after.csv`.
4. Model comparison: fixed matcher; word+char TF-IDF + logistic regression; small sentence embeddings (MiniLM, BGE-small, E5-small, multilingual MiniLM) + logistic regression → `results/experiments.csv`, `results/latency.csv`, `results/per_class_<model>.csv`.
5. Out-of-scope decision: trained class vs confidence threshold; PR sweep → `results/oos_threshold.csv`.
6. (Optional) Clean labels: relabel CV-flagged train rows, retrain, compare on checked set only → `results/clean_vs_silver.csv`.
7. Predict + benchmark: `submissions/<name>_predictions.csv`, format check script, latency.
8. Write-up: 2 pages from NOTES.md and results/.

## Prompt log
- `prompt_log.md` is a required deliverable. After each new prompt from Adi, append it verbatim under the next number with `Note: TODO (Adi)`. Adi writes the note and marks wrong/rejected answers. Never tidy or rewrite past entries.
- If you (Claude) got something wrong and it was caught, say so plainly in NOTES.md under "Where the AI was wrong".
