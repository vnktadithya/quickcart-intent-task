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

## Phase 2 — Label noise (IN PROGRESS: 2a and sampling done; LLM labelling blocked)
**Done so far.**
- `src/cv_flags.py` (seed 42): 5-fold out-of-fold predictions on train with word+char TF-IDF + logistic regression. A row is flagged when the held-out prediction differs from its silver label. Outputs: `results/cv_flags_train.csv` (4,230 flagged rows), `results/cv_disagreement_by_class.csv`.
- `src/sample_dev.py` (seed 42): 250-row dev sample, proportional by class with a floor of 15 per class, weights saved for re-weighting. Output: `data/dev_sample.csv`.

**Numbers.** 4,230 of 30,000 train rows (14.1%) are flagged. The flag rate is not the noise rate: it also counts the model's own mistakes.
| silver class | flag rate | what the model says instead (top) |
|---|---|---|
| out_of_scope | 6.3% | talk_to_agent (83) |
| order_status | 9.2% | delivery_delay (232) |
| change_address | 10.8% | account_help (35) |
| delivery_delay | 14.6% | order_status (355) |
| talk_to_agent | 19.3% | out_of_scope (259) |
| wrong_item | 20.0% | missing_item (105) |
| **promo_code_issue** | **60.6%** | payment_failed (91) |
| **account_help** | **70.9%** | change_address (148) |

**What I saw in a random look at the flagged rows (8 per class, not a measurement).**
- `promo_code_issue` rows in train are mostly NOT about promo codes ("net banking payment unsuccessful", "bruh pls connect me to a human", "the order was supposed to be here already"). Likewise `account_help` rows are mostly address-change messages. By the 8 rules these silver labels are wrong, so here the label, not the model, looks wrong. The two intents the starter could not predict are also the two whose train labels look worst.
- `talk_to_agent` -> `out_of_scope` flags are small talk ("call my mom", "recommend a good movie") that rule 6 says is `out_of_scope`.
- `delivery_delay` -> `order_status` flags are status questions ("has the rider been assigned yet") that rule 4 says are `order_status`.
These are impressions from 8-line samples. The 250-row audit is what will give the rate and interval.

**Blocked.** The `GROQ_API_KEY` in `.env` returns HTTP 401 "Invalid API Key" from Groq (key format looks right: `gsk_` prefix). The LLM pre-labelling cannot run until a valid key is supplied.

## Concepts
(Add as they come up: macro vs weighted F1, Wilson interval, calibration, cross-validated disagreement, char n-grams, ...)

## Where the AI was wrong
(Record every caught mistake: what it said, how it was caught, what was right.)
