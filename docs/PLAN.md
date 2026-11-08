# QuickCart Intent Task — Implementation Plan

Nov 8, 2026 · @Adi

## The task in one page

Improve a 12-intent support-chatbot classifier for a fictional app (QuickCart) and prove, with numbers, what was wrong and what helped. Accuracy is one signal; the depth and honesty of the reasoning matter more. Source: the task [README](https://github.com/Abhivanth-yellow/quickcart-intent-task).

**Scope:** intent extraction only. Entities, responses, dialogue flow and UI are out.

| Item | What it is |
| --- | --- |
| `data/taxonomy.json` | 12 intents with a few example messages each; two intents have none |
| `data/train.csv` | 30,000 rows, silver labels (automatic, with non-random errors) |
| `data/dev.csv` | 4,000 rows, silver labels |
| `data/test.csv` | 3,000 rows, no labels; scored on hidden gold labels |
| `baseline/` | TF-IDF centroid matcher plus an eval script, both deliberately flawed |

**Scoring:** per-class precision, recall and F1, macro-F1 and accuracy on the hidden gold test set, plus a read of the write-up, prompt log and code.

**Labelling rules for gold labels (the target to model):**

1. One label per message: the first request asked decides it.
2. A change of mind: the last request wins.
3. Cancel plus refund: not yet delivered is `cancel_order`; delivered is `refund_request`.
4. A complaint with no request is labelled by the problem: late, absent or different item.
5. Asking only for a human is `talk_to_agent`; any other request present wins.
6. Anything not about QuickCart orders, accounts or payments is `out_of_scope`, including greetings or thanks alone and services QuickCart lacks (gift cards, loyalty points, tax invoices, tipping, restaurant feedback).
7. Filler, emotion and unrelated detail never change the label.
8. Order IDs, amounts and item names never change the label.

**Constraints:** the final system must run at inference on 4 CPU cores and 15 GB RAM, one message at a time in real time. GPUs and hosted APIs serving open-source models are allowed for experiments and labelling; say what runs where. Report measured latency per message and the hardware. Fine-tuning is allowed but should be a small part of the story. Never edit or train on `test.csv`.

**Deliverables, due EOD Nov 9 (own target: 7 pm Nov 8):**

1. GitHub repo with a clear commit history.
2. `submissions/<name>_predictions.csv`: 3,000 rows, columns `id,intent`, intents from the 12 names.
3. Write-up, at most 2 pages.
4. Findings and benchmarks as readable CSVs or tables in `results/`.
5. `prompt_log.md`: every AI prompt in order, a one-line note each, wrong or rejected answers marked, untidied.

## What is already wrong

The starter's per-class report is mislabeled and its "macro-F1" is a weighted F1; every starter number needs recomputing before it is a baseline. I verified each finding below by running the repo code; each still needs its own reproducible script in your repo.

| # | Finding | Evidence | Effect |
| --- | --- | --- | --- |
| 1 | Per-class report rows belong to the wrong classes | `eval.py` passes `target_names` in taxonomy order, but sklearn orders labels alphabetically | "order\_status, n=212" is really `account_help`; every per-class row in `baseline_dev.json` is shifted |
| 2 | "macro\_f1" is weighted F1 | `f1_score(..., average="weighted")` | Reported 0.446; true macro-F1 0.411 |
| 3 | Confusion matrix saved with no label order | `confusion_matrix(y, pred)` with no `labels=` | Rows and columns cannot be read correctly |
| 4 | Two intents can never be predicted | `promo_code_issue` and `account_help` have no examples, so they get zero vectors | 419 dev rows (10.5%) score 0 recall |
| 5 | The out-of-scope threshold swallows half the data | Threshold 0.10 on TF-IDF cosine | 2,161 of 4,000 dev rows predicted `out_of_scope`; its precision 0.16 |
| 6 | Taxonomy contradicts itself and the rules | "where's my food" under both `order_status` and `delivery_delay`; "my order is incomplete" under both `missing_item` and `wrong_item`; "an item is missing" under `wrong_item`; "cancel my food order, taking too long" under `delivery_delay` | Overlapping centroids; examples that break rules 1 and 4 |
| 7 | Data was built from taxonomy examples | Verbatim taxonomy example inside the message: dev 1,001 of 4,000 (25%), train 7,867 of 30,000, test 746 of 3,000 | Matching scores on those rows overstate generalisation |
| 8 | Dev text leaks into the vectorizer | TF-IDF vocabulary and IDF fitted on train plus dev | Mild optimism in dev numbers |
| 9 | Top-3 and MRR ignore the threshold | Ranks use raw scores; the out-of-scope decision is skipped | Ranking metrics do not describe the deployed system |
| 10 | Train under-represents two classes | `promo_code_issue` 1.1% of train vs 5.2% of dev; `account_help` 1.3% vs 5.3% | Class imbalance and a train/dev distribution shift |
| 11 | Silver labels break the rules | In a 15-row train sample: "half my groceries are missing plus cancelling order" labeled `wrong_item` (rule 1 says `missing_item`); "my internet is sloow" labeled `talk_to_agent` (rule 6 says `out_of_scope`) | Noise is systematic, so a model can learn it |

No message repeats across train, dev and test, and no train message carries two labels.

**Why dev cannot be the judge:** a quick word-plus-character TF-IDF logistic regression trained on train reaches 0.86 accuracy and 0.85 macro-F1 on dev at about 1.7 ms per message (sandbox CPU, not your box). Dev labels come from the same noisy process as train. Part of that score is the model reproducing silver errors, so final choices are made on a hand-checked subset (Phase 2).

## System design

All heavy work (label auditing, LLM labelling, model comparison) happens offline; production runs one small model that answers in milliseconds on 4 cores.

```
OFFLINE (GPU / hosted open-source LLM allowed)
  Silver data -> Label audit (CV flags, LLM pre-labels, Adi checks) -> Checked data (relabeled train, 250 checked dev) -> Train and select (on checked set, under latency cap)
                                                                                                    |
                                                                                            trained model file
                                                                                                    v
ONLINE (4 CPU cores, 15 GB RAM, one message at a time)
  Message in -> Normalise text -> Classifier (word+char TF-IDF or small embedding, linear) -> Confidence check (intent, or out_of_scope below threshold)
```

The classifier box stays open until Phase 4 picks it with evidence. The leading candidate is word-plus-character TF-IDF with logistic regression: it trains in about 22 seconds and answers in about 1.7 ms on the sandbox CPU. Character n-grams help with typos and Hinglish spellings. Small sentence embeddings are the alternative; they must beat it on the checked set by more than the confidence interval to justify the extra latency. Any hosted LLM is used only offline for labelling, never at inference.

## Phased plan to 7 pm

About 7 hours of work. Each phase ends with a commit, a results file, and an update to `NOTES.md` and `DECISIONS.md`. If time runs short, cut Phase 6 first, then trim Phase 4 to two models.

| Phase | Time | Work | Outputs | Decision it settles |
| --- | --- | --- | --- | --- |
| 0. Setup | 20 min | Fork the repo, create a venv, run the baseline, add `CLAUDE.md`, `NOTES.md`, `DECISIONS.md`, `prompt_log.md` | First commit: "starter as received" | None |
| 1. Fix evaluation | 40 min | Write `src/evaluate.py`: fixed label order, true macro-F1, labeled confusion matrix, slices by message length, Hinglish, multi-request and contains-taxonomy-example. Re-score the starter. | `results/baseline_fixed.csv`, `results/eval_bugs.md` | Which numbers are trustworthy, and the reporting format for every later experiment |
| 2. Measure label noise | 60 min | (a) Five-fold cross-validated disagreement on train to flag suspicious labels. (b) Draw a stratified 250-row dev sample, pre-label it with an open-source LLM using the 8 rules, then check every row yourself. Report silver-vs-checked agreement with a 95% Wilson interval, and group error types. | `data/dev_checked.csv`, `results/label_noise.csv` | How far to trust silver labels; which classes are noisiest; the set used to pick the final model |
| 3. Fix the taxonomy | 30 min | Remove duplicate and rule-breaking examples, add descriptions and examples for `promo_code_issue` and `account_help`, compute intent-to-intent distances and compare them with the confusion matrix. Re-score the matcher. | `data/taxonomy_v2.json`, `results/taxonomy_before_after.csv` | Whether the taxonomy alone can fix the zero-recall intents |
| 4. Compare models | 90 min | Same splits and metrics for: fixed matcher; word-plus-character TF-IDF logistic regression; small sentence embeddings (MiniLM, BGE-small, E5-small, a multilingual MiniLM) plus logistic regression. Time each at 4 CPU threads. | `results/experiments.csv`, `results/latency.csv`, `results/per_class_<model>.csv` | Final model, chosen on the checked subset within the latency budget |
| 5. Out-of-scope decision | 30 min | Treat "unknown" as its own decision: compare a trained `out_of_scope` class with a confidence threshold, and pick the threshold from a precision-recall sweep. | `results/oos_threshold.csv` | How `out_of_scope` is produced and at what threshold |
| 6. Clean labels (optional) | 45 min | Relabel the train rows flagged in Phase 2 with the LLM and the rules, retrain, and compare on the checked subset only. | `results/clean_vs_silver.csv` | Whether cleaner labels beat a bigger model |
| 7. Predict and benchmark | 20 min | Write test predictions, run a format check (3,000 rows, valid names, matching ids), benchmark latency per message. | `submissions/<name>_predictions.csv` | None |
| 8. Write-up | 60 min | Two pages from `NOTES.md` and `results/`; finalise `prompt_log.md` untidied. | `WRITEUP.md` or PDF | None |

**How the final model is chosen:** highest macro-F1 on the checked dev subset, with its confidence interval, among models that answer in under about 20 ms per message on 4 cores. Silver-dev numbers are reported beside it, not used to choose.

**Why checked labels, not silver:** the hidden test uses gold labels that follow the 8 rules. A 250-row check gives roughly ±5 percentage points on accuracy at 95% confidence, which is wide, so small gaps between models stay "not distinguishable" in the write-up.

## Notes and logging system

Four files in the repo carry everything you need to explain the work on Nov 10. `CLAUDE.md` tells Claude Code to update them at the end of every phase and to explain each step before running it.

| File | What goes in it | Updated |
| --- | --- | --- |
| `NOTES.md` | Per phase: what was done, why, what the numbers mean, concepts explained in plain words (macro vs weighted F1, calibration, Wilson interval), and where an AI tool was wrong | End of every phase |
| `DECISIONS.md` | One entry per decision: options, evidence (numbers and file), choice, what would change it | Every time a choice is made |
| `prompt_log.md` | Every prompt to any AI tool, in order, with a one-line note on what you did with the answer; wrong or rejected answers marked | After every prompt, including this planning chat |
| `results/*.csv` | One row per experiment or per class, clear column names, before and after columns | When an experiment finishes |

**Rule for every decision:** no choice goes in without a number and the file it came from. "It is popular" is not evidence; the README says so.

**Rule for understanding:** after each phase, read `NOTES.md` and say the phase's conclusion out loud in two sentences. If you can't, ask Claude Code to explain before moving on.

## Write-up outline and discussion prep

Two pages, in the order the README asks for, each claim with its number.

1. **Biggest problems found:** evaluation bugs (with corrected numbers), taxonomy gaps and overlaps, label noise rate with its interval, train/dev shift.
2. **What changed, before and after:** one table, one row per change, silver-dev and checked-dev columns.
3. **What did not work:** with numbers.
4. **Limits of the evaluation:** silver dev, small checked set, wide intervals, the 25% taxonomy-template overlap.
5. **Latency and compute:** what runs where (experiments vs production), measured ms per message and hardware, estimate for 4 cores and 15 GB, any hosted API used and its production cost, latency and reliability.
6. **Next steps:** concrete, with expected numbers.

**Questions to be ready for on Nov 10:**

- Why this model over a bigger one? Show the latency-vs-macro-F1 row.
- How do you know a label was wrong and not the model? Show the checked-subset method.
- How was the `out_of_scope` threshold set, and what does it trade off?
- Which class is still worst, and why?
- Where did an AI tool get something wrong? Point to the marked entries in `prompt_log.md`.
- What would you do with one more day?
