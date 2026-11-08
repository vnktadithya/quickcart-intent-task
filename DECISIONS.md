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
- **Evidence:** README allows hosted APIs serving open-source models; no GPU here. Groq had no Llama-70B, so both `openai/gpt-oss-120b` and `qwen/qwen3.8-27b` were run. Against Adi's checked labels (`results/label_noise.csv`): all 250 rows qwen 96.8% [93.8, 98.4] vs gpt-oss 93.2% [89.4, 95.7]; on the 50 hard rows qwen 42/50 vs gpt-oss 33/50. The all-250 intervals overlap, but the hard-row gap is large.
- **Decision:** qwen3.8-27b is the primary labeller from now on (Phase 6 relabelling), gpt-oss-120b the second opinion; no local 7B. Raw outputs in `results/llm_raw/`, temperature 0. The separate 40-row pilot was dropped: Adi's 50 adjudications served as the model test.
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
- **Decision (Adi):** not 250 hand labels, and not blind. Two models label all 250; Adi labelled only the 50 rows where silver and the models did not all agree (CLAUDE.md updated by Adi). The 200 consensus rows are used as-is and called "not human-checked" in the write-up.
- **Evidence:** 200/250 rows had silver = both models; among the 50 reviewed, Adi's label differed from silver on 30 and sided with silver against both models on 20, so the models are not a copy of silver (`data/dev_review.csv`).
- **What would change it:** auditing a random sample of the 200 consensus rows would bound the shared-mistake risk; not done.

## D7 — What counts as the "checked" label
- **Phase:** 2
- **Question:** how to build `data/dev_checked.csv` from 50 human labels + 200 consensus rows?
- **Decision:** checked_intent = Adi's `my_label` where reviewed, else the consensus (silver = both models). Column `human_checked` marks which. Used for model selection from Phase 4; weights kept for re-weighting to the dev mix.
- **Evidence:** silver agrees with it on 88.0% [83.4, 91.5] (`results/label_noise.csv`).
- **What would change it:** evidence that consensus rows contain errors (random audit of a few of the 200).

## D8 — Taxonomy v2: which edits, and are the small ones worth keeping?
- **Phase:** 3
- **Question:** what to change in the taxonomy, and does it help?
- **Options considered:** (a) fill only the two empty intents; (b) cleanups only; (c) both.
- **Evidence:** `results/taxonomy_before_after.csv` (silver dev macro-F1, thr 0.10, train-only vectoriser): v1 0.4152; fill-only 0.5118; cleanups-only 0.4219; both 0.5183. Checked dev (n=250): 0.4437 / 0.5584 / 0.4680 / 0.5827.
- **Decision:** keep both. Filling the empty intents is the real gain; the cleanups add 0.007 / 0.025 (checked) and are required by rules 1 and 4 and by the duplicate-text defect (`results/eval_bugs.md` #6), so they stay even though their measured effect is small. Full edit list with reasons: `results/taxonomy_changes.csv`.
- **What would change it:** a learned model (Phase 4) that does not use taxonomy examples makes the taxonomy matter only for the matcher and the LLM prompts.

## D9 — New examples are hand-written paraphrases, not copied from data
- **Phase:** 3
- **Question:** where do the 16 examples for promo_code_issue / account_help come from?
- **Options considered:** copy train rows; pick train rows the CV model agrees with; write paraphrases.
- **Evidence:** 25% of dev rows already contain a taxonomy example verbatim (`results/eval_bugs.md` #7); copying train rows would raise that overlap and inflate matcher scores. Train rows for these two classes are also the noisiest (CV flag rate 61% / 71%, `results/cv_disagreement_by_class.csv`).
- **Decision:** write paraphrases in the same style (typos removed, some Hinglish), informed by CV-agreed train rows. Disclosed in NOTES.md.
- **What would change it:** an audit showing the paraphrases are unrepresentative of the test phrasing.

## D10 — Intent-distance measure
- **Phase:** 3
- **Decision:** cosine between TF-IDF centroids (taxonomy examples, and train rows per class), compared with confusion by Spearman rank correlation. Rho 0.32-0.48 for the matcher, -0.07 for CV confusion (`results/intent_distance_correlation.csv`). Used for interpretation only, no model choice depends on it.

## D11 — Final model family: word+char TF-IDF + logistic regression
- **Phase:** 4
- **Question:** which model goes into production (4 cores, 15 GB, one message at a time)?
- **Options considered:** taxonomy matcher v2; TF-IDF word+char LR; MiniLM-L6, BGE-small, E5-small, multilingual MiniLM, each + LR.
- **Evidence:** `results/experiments.csv`: checked macro-F1 0.9013 [0.849, 0.937], highest; silver dev 0.8448, highest by 4-6 points; p50 2.24 ms vs 10-18 ms for embeddings (`results/latency.csv`). The best embedding (BGE-small) is statistically tied on the 250 checked rows (delta -0.009 [-0.057, +0.044]) but 8x slower and 4 points lower on silver dev. Selection rule from the plan: highest checked macro-F1 under about 20 ms; embeddings had to beat TF-IDF by more than the interval, and none did.
- **Decision:** TF-IDF word+char LR is the working model for Phases 5-7. Embeddings are not used.
- **What would change it:** a tuned or fine-tuned embedding that beats TF-IDF on checked labels by more than the interval; real chat data with fewer template phrases.

## D12 — Hyper-parameters stay fixed
- **Phase:** 4
- **Decision:** C=10 for every logistic regression; TF-IDF word (1,2)/min_df 2 + char_wb (2,5)/min_df 3; no search. Reason: the only trustworthy labels are 250 rows, and tuning on them would overfit the selection; silver dev alone must not choose (CLAUDE.md).
- **What would change it:** a cross-validated search inside train (silver) with the final choice still checked on the checked set; listed as a next step.

## D13 — Embedding models and their licences
- **Phase:** 4
- **Evidence:** `results/embedding_models.csv` (licence read from the Hugging Face model card metadata): all-MiniLM-L6-v2 apache-2.0; bge-small-en-v1.5 mit; e5-small-v2 mit; paraphrase-multilingual-MiniLM-L12-v2 apache-2.0. Models were only used offline in experiments and are not in the final system.
