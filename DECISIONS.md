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

## D4 — Labeller model: a larger hosted open-weights model, not the local 7B
- **Phase:** 2
- **Question:** which LLM pre-labels the 250 dev rows (and later relabels train)?
- **Options considered:** local Qwen2.5-7B-Instruct (Ollama, CPU only); hosted 70B-class open-weights model on Groq.
- **Evidence:** README allows hosted APIs serving open-source models; no GPU on this machine, so a larger model locally is not practical. No accuracy evidence yet, the model is picked by pilot.
- **Decision:** use a 70B-class open model through Groq (exact model id to be set after the key works); keep local Qwen2.5-7B as a second opinion. Raw outputs to `results/llm_raw/`, temperature 0.
- **What would change it:** the pilot showing a different candidate agrees more with Adi's labels; the key/rate limits not working.

## D5 — Minimum 15 rows per class in the 250-row dev sample
- **Phase:** 2
- **Question:** proportional sampling or a per-class floor?
- **Options considered:** proportional (promo_code_issue 12.9, wrong_item 13.1, account_help 13.2, change_address 14.1 rows); floor of 15.
- **Evidence:** the hidden test is scored per class and by macro-F1; promo_code_issue and account_help are the two classes the starter could not predict at all (recall 0, `results/baseline_fixed.csv`), and their train labels look the worst (`results/cv_disagreement_by_class.csv`: flag rates 61% and 71% vs 14% overall). Phases 3-5 are mostly about them, so the checked set must be able to show whether they improve. 13 rows cannot.
- **Decision:** floor of 15; weights (class size / sampled size) saved in `data/dev_sample.csv` so overall agreement is re-weighted to the dev distribution. Cost: 15 vs 12.9 is a small change, headline unbiased after weighting.
- **What would change it:** none of the per-class numbers being used for decisions.

## D6 — No manual labelling of the full 250 rows (Adi's call)
- **Phase:** 2
- **Question:** who labels the checked set?
- **Decision (Adi):** do not hand-label 250 rows; show LLM suggestions (no blind labelling). Conflicts with the CLAUDE.md rule "Adi verifies every row", so the scope of Adi's review is still open (see the Phase 2 reply).
- **What would change it:** the review scope agreed with Adi.
