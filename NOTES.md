# NOTES — QuickCart intent task

Running notebook. Updated at the end of every phase. Plain words; every number points to the file it came from.

## Phase 0 — Findings from the planning read (to be reproduced by scripts in Phase 1)
These were found by running the starter code during planning. They are not evidence until a script in this repo reproduces them.

1. Per-class report is mislabeled: `eval.py` passes `target_names` in taxonomy order, but sklearn sorts labels alphabetically. The row printed as "order_status, n=212" is really `account_help`.
2. "macro_f1" is weighted F1. Reported 0.446; true macro-F1 0.411.
3. Confusion matrix is saved without a label order.
4. `promo_code_issue` and `account_help` have no taxonomy examples → zero vectors → never predicted (419 dev rows, 10.5%).
5. Threshold 0.10 sends 2,161 / 4,000 dev rows to `out_of_scope` (its precision 0.16).
6. Taxonomy contradictions: "where's my food" in two intents; "my order is incomplete" in two intents; "an item is missing from my order" under `wrong_item`; "cancel my food order, taking too long" under `delivery_delay`.
7. Messages contain a taxonomy example verbatim: dev 1,001/4,000, train 7,867/30,000, test 746/3,000.
8. TF-IDF vocabulary/IDF fitted on train + dev (dev leaks into features).
9. Top-3 and MRR ignore the out-of-scope threshold.
10. Train under-represents `promo_code_issue` (1.1% vs 5.2% of dev) and `account_help` (1.3% vs 5.3%).
11. Silver labels break the rules (examples in a 15-row sample).
12. Quick check: word+char TF-IDF + logistic regression trained on train → dev accuracy 0.86, macro-F1 0.85, ~1.7 ms/message (planning sandbox). Dev shares train's noise, so this is optimistic.

## Phase 1 — Fix evaluation (`src/evaluate.py`)
**What was done.** Wrote `src/evaluate.py` and re-scored the unmodified starter on dev. Outputs: `results/baseline_fixed.csv` (overall, per-class and per-slice rows, with the starter's reported values beside the fixed ones), `results/baseline_confusion.csv` (labelled), `results/eval_bugs.md` (each bug, evidence, size of effect). Re-run: `python -m src.evaluate`.

**Numbers (dev, silver labels, n=4000).**
| | starter reported | fixed |
|---|---|---|
| accuracy (top-1) | 0.4233 | 0.4233 |
| "macro-F1" | 0.446 | **0.4105** (the 0.446 is weighted F1) |
| top-3 | 0.7192 | 0.7208 (after threshold) |
| MRR | 0.6652 | 0.5996 (after threshold) |
- Fixed per-class: `promo_code_issue` and `account_help` have exactly 0 recall (419 rows, 10.5%). The starter's report showed the zeros under `order_status` and `change_address`, so it hid the real failure.
- `out_of_scope`: 2,161 of 4,000 rows predicted (54%); precision 0.16, recall 0.91.
- Dev leak in the vectoriser (train+dev fitted): removing dev from the fit changes accuracy 0.4233 -> 0.4303 and macro-F1 0.4105 -> 0.4152. Small, and not in the "optimistic" direction. So bug 8 is real but does not explain the low score.
- Verbatim taxonomy example inside the message: train 7,867, dev 1,001 (25%), test 746. On those dev rows accuracy is 0.695, on the rest 0.332. So even the 0.42 flatters the matcher on unseen phrasing.
- Other slices (heuristic, see limits): Hinglish accuracy 0.308 vs 0.437 English-only; multi-request 0.319 vs 0.454 single. Length matters little (0.38-0.44).

**Concepts in plain words.**
- *Macro vs weighted F1:* macro averages the 12 per-class F1s equally, so a dead class counts as much as a big one. Weighted averages by class size, so big classes hide failures. Macro is the right headline here because the test is scored per class.
- *Why the per-class rows were shifted:* sklearn's `classification_report` sorts labels alphabetically; the starter then attached names in taxonomy order. Passing `labels=` fixes order and names together.
- *Ranking after the threshold:* the deployed bot outputs `out_of_scope` when the best score is below 0.10; ranking metrics that ignore this describe a system nobody runs. MRR falls 0.666 -> 0.600 once it is counted.

**Limits / things not verified.**
- Hinglish and multi-request slices are word-list/regex heuristics, not validated against human labels ("and" over-counts multi-request).
- Shipped top-3/MRR (0.7192/0.6652) differs from my raw recompute (0.7200/0.6655) by <0.001; cause not investigated (likely tie-breaking).
- Everything is on silver dev labels.

**Went wrong.** My first attempt at writing the script via a shell heredoc failed (quoting error) and left junk in a draft line; I rewrote it with the file-write tool. No numbers were affected. The plan's figure of "true macro-F1 0.411" was reproduced (0.4105).

## Phase 2 — Label noise
**What was done.**
1. `src/cv_flags.py` (seed 42): 5-fold out-of-fold predictions on train (word+char TF-IDF + logistic regression). A row is flagged when the held-out prediction differs from the silver label. -> `results/cv_flags_train.csv` (4,230 of 30,000 = 14.1%), `results/cv_disagreement_by_class.csv`.
2. `src/sample_dev.py` (seed 42): 250-row dev sample, proportional by class with a floor of 15 per class, weights kept. -> `data/dev_sample.csv`.
3. `src/llm_label.py`: two open-weights models on Groq (`openai/gpt-oss-120b`, `qwen/qwen3.8-27b`) labelled the 250 rows using the 8 rules, temperature 0. Raw outputs: `results/llm_raw/*.jsonl`. Only messages from this repo were sent.
4. `src/label_noise.py --stage review`: rows where silver and both models do not all agree (50) -> `data/dev_review.csv`. Adi labelled these 50 himself (not blind; he saw silver + model labels) in column `my_label`, with an `error_type` where he disagreed with silver.
5. `src/label_noise.py --stage final` -> `data/dev_checked.csv`, `results/label_noise.csv`.

**Checked label of a row** = Adi's label for the 50 reviewed rows; for the other 200 rows (silver = both models) the label they all agree on. **Those 200 were not human-checked.**

**Numbers (`results/label_noise.csv`).**
- Silver agrees with the checked label on **220 / 250 = 88.0%**, 95% Wilson interval **[83.4%, 91.5%]**; re-weighted to the dev class mix 88.1% [83.5%, 91.6%]. So about 12% of silver dev labels are wrong, and more if any of the 200 unchecked rows are.
- Per silver class (n = 15-33, intervals are wide): out_of_scope 23/23, order_status 31/32, refund_request 26/27, payment_failed 18/19, wrong_item 14/15, delivery_delay 28/33, cancel_order 19/24 (79%), missing_item 13/17 (76%), promo_code_issue / change_address / account_help / talk_to_agent 12/15 each (80%).
- Of the 30 wrong silver labels, Adi tagged 11 as rule violations (rule 6 should be out_of_scope: 5; rule 1 first request: 3; rule 5 agent vs other request: 2; rule 2 change of mind: 1) and 19 as plain "silver label wrong". The silver->checked pairs are mostly one-offs (largest: cancel_order->delivery_delay 3), i.e. the errors are spread out, not one dominant confusion. Everything labelled out_of_scope in silver was right (23/23), but 5 messages that silver gave another label should be out_of_scope.
- Labellers vs the checked label, all 250 rows: silver 88.0%, gpt-oss-120b 93.2% [89.4, 95.7], qwen3.8-27b **96.8%** [93.8, 98.4]. On the 50 hard rows only (where somebody disagreed): silver 20/50, gpt-oss 33/50, qwen 42/50 (84%). Caveat: the 50 are exactly the rows where silver differed from at least one model, and the all-agree rows are scored as correct for all three labellers, so the all-250 figures are upper bounds. Qwen was the better labeller on this evidence.

**What this means.** Silver dev is mostly right (88%) but not a gold standard: a model that learns silver perfectly would top out near 88% on checked labels, and silver-dev differences of a point or two between models are inside the noise. The two open-weights models agree with each other (90.8%) far more than either agrees with silver, which makes them useful as a noise detector; they are not a copy of silver or of each other, because Adi sided with silver against both models on 20 rows.

**Open question (not resolved).** In train, the CV flag rate for `promo_code_issue` (61%) and `account_help` (71%) is far above the overall 14%, and a quick look showed many of those train rows are clearly not about promos / accounts. In dev the same two classes agree with the checked label at 80% (12/15). Two explanations fit: train labels for these two classes are much worse than dev's, or the model simply has too few rows (325 and 388) to learn them. The 15-row dev numbers cannot separate them. An LLM audit of a train sample could.

**Concepts in plain words.**
- *Cross-validated disagreement:* train on 4/5 of the data, predict the held-out 1/5, rotate. Every row is predicted by a model that never saw it, so a disagreement is a candidate error, not proof (label or model may be wrong).
- *Wilson interval:* the range of true agreement rates consistent with k/n in the sample. For 220/250 it is 83.4-91.5%. Unlike the textbook k/n +- 2*sqrt(p(1-p)/n), it behaves for small n and for rates near 0 or 1.
- *Why weights:* the 15-row floor over-samples four small classes, so the plain 250-row rate is slightly off the dev mix; weight = dev class size / sampled size corrects it (effect here: 88.0% -> 88.1%).

**Limits.** 200 of 250 checked labels are model-and-silver consensus, not human-verified; shared mistakes are possible. Per-class intervals are 20-30 points wide. One reviewer (Adi), no second annotator, so no inter-annotator agreement. Model names/licences were taken from Groq's catalogue and not verified (qwen3.8-27b in particular).

**Where the work went wrong.** My first plan for Phase 2 leaned on a local 7B model without checking that the README allows hosted open-weights APIs; Adi caught it (prompt 6). The Groq 401s were a stale `.env` file on disk, not a code bug (prompt 9). Twice a shell heredoc failed on quoting and I moved the code into files.

## Concepts
(Add as they come up: macro vs weighted F1, Wilson interval, calibration, cross-validated disagreement, char n-grams, ...)

## Where the AI was wrong
(Record every caught mistake: what it said, how it was caught, what was right.)
