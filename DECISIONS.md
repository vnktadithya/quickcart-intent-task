# DECISIONS — QuickCart intent task

One entry per decision. No entry without evidence.

## Template
### D<n> — <short title>
- **Phase:**
- **Question:**
- **Options considered:**
- **Evidence:** (numbers + file)
- **Decision:**
- **What would change it:**

## D0 — Choose models on a checked dev subset, not silver dev
- **Phase:** planning
- **Question:** what set decides the final model?
- **Options considered:** silver dev (4,000 rows); hand-checked dev subset (~250 rows).
- **Evidence:** silver labels contain non-random errors (README; examples in NOTES.md #11). A model trained on silver train scores 0.86 on silver dev partly by reproducing the same errors.
- **Decision:** select on the checked subset; report silver-dev numbers beside it.
- **What would change it:** the checked subset agreeing with silver dev above ~95%, which would make silver dev a fair judge.

## D1 — Label order and macro-F1 definition
- **Phase:** 1
- **Question:** how to report per-class and headline numbers so they are correct and comparable across experiments?
- **Options considered:** keep starter format; sklearn default (alphabetical) labels; explicit taxonomy order.
- **Evidence:** starter's per-class rows are shifted (`results/eval_bugs.md` #1); its "macro-F1" 0.446 is weighted, true macro 0.4105 (`results/baseline_fixed.csv`).
- **Decision:** always `labels=intents` in taxonomy order; headline = `average="macro"`; confusion matrix with names.
- **What would change it:** the graders' scoring script using a different order (irrelevant to values, only layout).

## D2 — Ranking metrics are computed after the threshold
- **Phase:** 1
- **Question:** should top-3/MRR use raw scores (starter) or the deployed decision?
- **Evidence:** MRR 0.6655 raw vs 0.5996 after threshold (`results/eval_bugs.md` #9).
- **Decision:** report after-threshold (out_of_scope ranked first when best < threshold) and keep raw only for comparison.
- **What would change it:** if Phase 5 replaces the threshold with a trained out_of_scope class, ranking comes straight from model probabilities.

## D3 — Slice definitions are heuristics
- **Phase:** 1
- **Question:** how to define Hinglish / multi-request slices without labels?
- **Options considered:** word list / regex; LLM tagging; manual tagging.
- **Evidence:** Hinglish 413/4000 (10.3%), multi 915/4000 (22.9%) (`results/baseline_fixed.csv`); no validation yet.
- **Decision:** word list + regex now, documented as unvalidated; revisit on the 250-row checked set in Phase 2.
- **What would change it:** the checked set showing the heuristics mislabel many rows.
