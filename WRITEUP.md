# QuickCart intent task: write-up (Adithya)

**Result.** A 3.5 MB word+char TF-IDF + logistic-regression model, with `out_of_scope` as a trained class, lifts macro-F1 from a true **0.41** (starter reports 0.446) to **0.845** on silver dev and **0.901** on a hand-checked sample, at **2.3 ms per message** on 4 threads. Every number comes from a script in `src/` and a CSV in `results/` (fixed seeds); reasoning is in `NOTES.md` and `DECISIONS.md`.

## 1. Biggest problems found
1. **The starter's evaluation was wrong** (`results/eval_bugs.md`). Per-class rows were attached to the wrong class (sklearn sorts labels alphabetically, the starter used taxonomy order), which hid the two dead intents. "macro-F1" was weighted F1: 0.446 reported, **0.4105** true. The confusion matrix had no labels, and MRR ignored the threshold (0.665, 0.600 once applied).
2. **Two intents were unpredictable.** `promo_code_issue` and `account_help` had no taxonomy examples: recall 0 on 419 dev rows (10.5%). The 0.10 threshold sent 54% of dev to `out_of_scope` (precision 0.16).
3. **Silver labels are about 12% wrong.** On a 250-row stratified dev sample silver agrees with the checked label on **220/250 = 88.0%** (95% Wilson interval 83.4-91.5%). Errors are scattered (largest single confusion: 3 rows); 11 of the 30 break rules 1, 2, 5 or 6. In train, cross-validated disagreement flags 61% of `promo_code_issue` and 71% of `account_help` rows (14% overall), and many are plainly not about promos or accounts.
4. **Taxonomy defects:** two texts under two intents, two rule-breaking examples. 25% of dev messages contain a taxonomy example verbatim (accuracy 0.70 on those vs 0.33 on the rest): taxonomy and data are not independent.

## 2. What changed (macro-F1; silver dev n=4,000, checked n=250)
| Step | Silver dev | Checked |
|---|---|---|
| Starter, true macro-F1 | 0.4105 | 0.440 |
| + vectoriser fitted on train only | 0.4152 | 0.444 |
| + taxonomy v2 (dedupe, rule fixes, 16 examples for the two empty intents) | 0.5183 | 0.583 |
| + argmax instead of the 0.10 threshold | 0.5952 | 0.657 |
| + learn from 30,000 train rows: **TF-IDF + LR** | **0.8448** | **0.901** [0.849, 0.937] |

- **Taxonomy ablation:** filling the two empty intents gives +0.097, the cleanups alone +0.007. I kept both (the rules require them).
- **Out-of-scope** (`results/oos_summary.csv`): a trained class (0.845) beats a confidence threshold on an 11-class model (0.759). Average precision for spotting out-of-scope: 0.90 from P(`out_of_scope`), 0.22 and 0.09 from confidence. A linear model is confident on familiar words ("swiggy gave me the wrong order" gets 0.99 for `wrong_item`), so confidence does not mean "in scope".
- **Labelling:** two open-weights models on Groq labelled the 250 rows; I labelled the 50 where silver and the models disagreed. qwen3.8-27b matched the checked label on 96.8%, gpt-oss-120b on 93.2%.

## 3. What did not work
- **Sentence embeddings** (MiniLM-L6, BGE-small, E5-small, multilingual MiniLM + LR): 0.78-0.80 silver-dev macro-F1 vs 0.845, and 4-8x slower. The best (BGE-small) is statistically tied on the checked set (-0.009, interval [-0.057, +0.044]).
- **"Multilingual" did not help Hinglish:** accuracy 0.758 vs 0.804 on English-only; TF-IDF with character n-grams scored 0.848 vs 0.854.
- **Confidence thresholds** (above). The dev-leak fix changed accuracy by only +0.007.
- **Still weak:** `account_help` (recall 0.65) and multi-request messages (accuracy 0.78 vs 0.875).
- **Not done:** Phase 6, relabelling flagged train rows, so "cleaner labels vs a bigger model" is untested.

## 4. Limits of the evaluation
- A model that learned silver perfectly would top out near 88% against checked labels, so silver-dev gaps of a point or two are noise.
- The checked set is small (about ±5 points). 200 of its 250 rows are consensus rows (silver = both models) that no human looked at; I labelled the other 50, not blind, as the only annotator. 0.901 is probably optimistic.
- The data is synthetic and template-based, which likely favours TF-IDF. No hyper-parameter tuning (too few checked rows). Hinglish and multi-request slices are unvalidated heuristics.
- On 60 messages I wrote, the trained class catches 22 of 30 novel out-of-scope ones; misses are other companies' complaints (3) and short unfamiliar messages. Test performance is unknown.

## 5. Latency and compute (4 cores, 15 GB budget)
Protocol: 4 threads, one message per call, 20 warm-up, 500 timed dev messages (`results/latency.csv`).

| Model | p50 / p95 ms |
|---|---|
| **Final: TF-IDF + LR**, loaded from disk | **2.3 / 3.2** (3 runs: 2.27-2.68 / 3.03-3.59) |
| Taxonomy matcher | 0.9 / 1.5 |
| MiniLM-L6 + LR | 10.1 / 13.8 |
| BGE-small, E5-small, multilingual MiniLM + LR | 17-18 / 25-26 |

- **Hardware:** a 12th-gen i5-1235U laptop (10 cores / 12 threads, 15.7 GB, no GPU) limited to 4 threads, **not** a dedicated 4-core box. My unmeasured estimate for a 4-vCPU cloud machine is 3-5 ms. The model file is 3.5 MB (the 300 MB seen in benchmarking includes PyTorch, not needed in production).
- **What runs where:** embedding experiments ran offline on this laptop. Hosted open-weights models on Groq (gpt-oss-120b, qwen3.8-27b) were used **offline only** for labelling (about 500 calls on repo messages, free tier, roughly 8 rows a minute). In production an LLM call would add a network round trip, per-call cost, rate limits and an outside dependency; production is the single CPU model.

## 6. Next steps
1. Relabel the 4,230 flagged train rows with qwen3.8-27b (about 45 minutes), retrain, compare on the checked set only. Headroom is the 12% label noise; I cannot promise a number.
2. Class weights or more data for `account_help` and `promo_code_issue` (cheap to test).
3. A competitor-name rule (Swiggy, Zomato, Dominos): fixing the 3 company misses would lift novel out-of-scope recall on my probes from 0.73 to 0.83.
4. A first-request splitter for multi-request messages.
5. Check more of the 200 consensus rows (or add a second annotator) to tighten the interval, then repeat model selection.

*AI use.* Claude Code was used throughout. Mistakes caught: a local 7B labeller proposed without checking that hosted open-weights APIs are allowed; a garbled import line; a format checker that crashed on a wrong-column file; a miscount of probe misses (5 company cases, really 3). See `prompt_log.md` and `NOTES.md`.
