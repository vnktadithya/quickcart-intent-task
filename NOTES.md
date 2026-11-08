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

## Concepts
(Add as they come up: macro vs weighted F1, Wilson interval, calibration, cross-validated disagreement, char n-grams, ...)

## Where the AI was wrong
(Record every caught mistake: what it said, how it was caught, what was right.)
