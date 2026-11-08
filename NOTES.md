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

## Phase 3 — Taxonomy v2
**What was done.**
- `src/build_taxonomy_v2.py` -> `data/taxonomy_v2.json` and `results/taxonomy_changes.csv` (every edit with its reason). Edits: 3 examples removed (the duplicate "where's my food" under delivery_delay; the duplicate "my order is incomplete" under wrong_item; the ambiguous "charged twice, please refund one payment" under payment_failed), 2 moved to the intent the rules give ("an item is missing from my order": wrong_item -> missing_item; "cancel my food order, taking too long": delivery_delay -> cancel_order), 16 examples added for the two empty intents (8 each), plus a one-line description and a "not this" note per intent for humans / LLM prompts (the TF-IDF matcher ignores them).
- `src/matcher.py`: the starter's method (TF-IDF centroid per intent, cosine, threshold) with the taxonomy file, vectoriser corpus and threshold as arguments. `baseline/` is untouched. Its experiment A reproduces the starter exactly (0.4233 / 0.4105).
- `src/taxonomy_eval.py` -> `results/taxonomy_before_after.csv`, `results/intent_distance.csv`, `results/intent_distance_correlation.csv`.

**Before / after (`results/taxonomy_before_after.csv`; same method, only the taxonomy changes).**
| experiment | silver dev macro-F1 (n=4000) | checked dev macro-F1 (n=250) | zero-recall classes |
|---|---|---|---|
| A v1 as shipped (vectoriser on train+dev, thr 0.10) | 0.4105 | 0.4399 | promo_code_issue, account_help |
| B v1, vectoriser on train only, thr 0.10 | 0.4152 | 0.4437 | same two |
| **C v2**, train-only, thr 0.10 | **0.5183** | **0.5827** | none |
| F ablation: v1 + fill the 2 empty intents only | 0.5118 | 0.5584 | none |
| G ablation: v2 cleanups only (2 intents left empty) | 0.4219 | 0.4680 | same two |
| D v1, train-only, no threshold (argmax) | 0.4797 | 0.5084 | same two |
| E v2, train-only, no threshold (argmax) | 0.5952 | 0.6570 | none |
- Filling the two empty intents does almost all of the work: B->F +0.097, B->G +0.007. The 3 removals + 2 moves are justified by the rules but are worth less than one point of macro-F1 on their own.
- v2 F1 for the newly filled classes (silver dev, thr 0.10): promo_code_issue 0.000 -> 0.603, account_help 0.000 -> 0.514. Cost: delivery_delay F1 0.395 -> 0.355 and payment_failed 0.408 -> 0.391, because each lost an example (delivery_delay and payment_failed now have 8 and 7).
- The threshold is still the biggest problem the taxonomy cannot fix: at 0.10, 47% of dev rows are still sent to out_of_scope under v2, vs 7% with plain argmax (D, E). Macro-F1 under argmax is 0.595. This is a diagnostic only: the real threshold is decided in Phase 5 with a precision-recall sweep, not by looking at this table.
- Checked-set (n=250) numbers move the same way as silver-dev but carry roughly +-6 points of uncertainty and 200 of the 250 rows are consensus (easy) rows.

**Intent distance vs confusion (`results/intent_distance.csv`, `..._correlation.csv`).** Distance = cosine similarity between intent centroids (taxonomy examples, or the mean of all train rows per class) in TF-IDF space. Confusion rate of a pair = (a->b + b->a errors) / (n_a + n_b).
| comparison | pairs | Spearman rho | p |
|---|---|---|---|
| taxonomy-v1 similarity vs matcher confusion (no threshold), excluding the 2 empty intents | 45 | 0.34 | 0.02 |
| taxonomy-v2 similarity vs matcher confusion | 66 | 0.32 | 0.01 |
| train-centroid similarity vs matcher confusion | 66 | 0.48 | <0.001 |
| train-centroid similarity vs CV confusion on train | 66 | -0.07 | 0.56 |
- Examples of what the matrix shows: order_status <-> delivery_delay was the most similar non-empty pair in v1 (0.26; 0.12 after v2), missing_item <-> wrong_item 0.196 -> 0.056 after removing the duplicates, and its matcher confusion fell 14.4% -> 6.6%. change_address <-> account_help is the closest pair by train data (0.80) and the top CV confusion (8.1%): the two intents are semantically adjacent (address and phone/profile edits) and a boundary rule for them is needed ("address = change_address; phone/name/profile = account_help", already in the v2 "not this" notes).
- So taxonomy similarity explains part of the matcher's confusions (rho about 0.3-0.5, weak-to-moderate), but train-centroid similarity does NOT predict a trained model's confusions (rho -0.07). My guess, not verified: train centroids are all close (0.4-0.8) because the messages share generic filler words, and CV confusions are driven by label noise rather than geometry.

**Concepts in plain words.**
- *Ablation:* undo one part of a change and re-measure, to see which part earned the gain (here F and G split v2 into its two kinds of edit).
- *Centroid:* the average vector of an intent's examples. Cosine similarity between two centroids says how alike the two intents' wording is.
- *Spearman rho:* correlation of ranks, in -1..1; 0 means no relation. With 45-66 pairs, 0.3 is detectable but weak.

**Limits / honesty.**
- The 16 new examples are my own paraphrases. I wrote them after reading train rows for those two classes that the CV model agreed with (selected as the cleaner ones), and earlier in Phase 2 I had displayed some dev review rows, so their wording may echo dev phrasing (e.g. coupon applied but total unchanged). No dev/test row was copied. Because this tilts toward the data's style, the large gain for the two classes may be optimistic for truly new phrasing. Check: Phase 4's learned models do not depend on these examples.
- The "contains-taxonomy-example" slice was not recomputed for v2.
- 8 examples per new intent is a small sample; the other intents have 6-14.

**Where the AI was wrong.** I wrote a garbled import line in `src/build_taxonomy_v2.py` before running it (caught by reading it back, fixed before the first run). Earlier I called the heredoc failures quoting errors; the fix was moving code into files.

## Phase 4 — Model comparison
**What was done.** Same train data (30,000 silver rows), same two eval sets (silver dev 4,000; checked dev 250), same metric code (`src/evaluate.py`), fixed hyper-parameters for every model (nothing tuned on dev or checked). Scripts: `src/models.py`, `src/embed_cache.py` (embeds train + dev once; test is not embedded), `src/run_models.py`, `src/latency.py`. Outputs: `results/experiments.csv`, `results/per_class_<model>.csv` (7 files), `results/experiments_slices.csv`, `results/latency.csv`, `results/embedding_models.csv`.

**Results.**
| model | silver dev macro-F1 | checked macro-F1 [95% CI] | checked delta vs TF-IDF [95% CI] | p50 / p95 ms (4 threads) |
|---|---|---|---|---|
| taxonomy matcher v2, thr 0.10 | 0.5183 | 0.5827 [0.504, 0.644] | -0.319 [-0.400, -0.238] | 0.92 / 1.45 |
| taxonomy matcher v2, argmax | 0.5952 | 0.6570 [0.588, 0.716] | -0.244 [-0.316, -0.176] | (same as above) |
| **word+char TF-IDF + LR** | **0.8448** | **0.9013 [0.849, 0.937]** | 0 | **2.24 / 3.19** |
| MiniLM-L6 + LR | 0.7955 | 0.8801 [0.829, 0.916] | -0.021 [-0.070, +0.032] | 10.08 / 13.77 |
| BGE-small + LR | 0.8042 | 0.8926 [0.842, 0.932] | -0.009 [-0.057, +0.044] | 17.68 / 25.20 |
| E5-small-v2 + LR | 0.7924 | 0.8554 [0.801, 0.901] | -0.046 [-0.094, +0.000] | 18.32 / 26.41 |
| multilingual MiniLM + LR | 0.7808 | 0.8743 [0.823, 0.912] | -0.027 [-0.071, +0.022] | 17.43 / 25.05 |
- Learning from the 30,000 train rows beats matching against taxonomy examples by +0.25 to +0.32 macro-F1 on checked (CI excludes 0): same compute class, far more signal.
- No embedding model beats TF-IDF+LR. On the checked set the gaps to TF-IDF are inside the interval (BGE-small -0.009 is statistically indistinguishable; E5 is borderline), but on silver dev (n=4000, much tighter) all four are 4-6 points worse (0.78-0.80 vs 0.845), and they are 4-8x slower. Silver-dev gaps are partly "who reproduces the silver errors better", so I use them as supporting, not deciding, evidence (D0).
- Slices (silver dev accuracy, `results/experiments_slices.csv`): TF-IDF is 0.848 on Hinglish vs 0.854 English-only: char n-grams cope with romanised Hindi. The multilingual MiniLM is the WORST on Hinglish (0.758 vs 0.804 English-only), so "multilingual" did not help here. Multi-request messages are hardest for all models (TF-IDF 0.780 vs 0.875 single-request); long messages too (0.804). On "contains a taxonomy example" vs not, TF-IDF is 0.843 vs 0.857, so it does not lean on the taxonomy-overlap shortcut (the matcher does: 0.764 vs 0.547).
- Per class (TF-IDF, silver dev F1): lowest are `account_help` 0.750 (recall 0.651), `promo_code_issue` 0.821, `talk_to_agent` 0.818, `payment_failed` 0.822. The two classes that were dead in the starter are the weakest learned classes: they have only 1.1-1.3% of train rows, and Phase 2 showed their train labels are the noisiest.

**Concepts in plain words.**
- *Frozen embedding + logistic regression:* the transformer turns a sentence into 384 numbers and is not updated; only a linear classifier on top is trained. Fast to train (6-10 s), but the embedding was built to capture general meaning, not the difference between "late" and "missing" in this data.
- *Why TF-IDF can win here:* the intents are separated by specific words and phrases ("refund", "coupon", "otp"). TF-IDF with character n-grams keys on exactly those, including misspellings; a general-purpose embedding blurs them. This is likely helped by the data being synthetic and template-based, so the finding may not transfer to real chat logs.
- *Paired bootstrap:* resample the 250 rows with replacement 1,000 times; for each resample compute both models' macro-F1; the spread of the difference is the interval. Same resample for both models, so row difficulty cancels.

**Latency details.** Protocol from CLAUDE.md: OMP/MKL threads = 4, `torch.set_num_threads(4)`, one message per call, 20 warm-up, 500 timed dev messages, model already loaded (loading is not in the span). Hardware: 12th Gen Intel Core i5-1235U (10 cores / 12 threads: 2 performance + 8 efficiency), 15.7 GB RAM, Windows, no GPU. `lscpu` does not exist on Windows; CPU model read via Windows WMI instead. This is a laptop with 4 threads allowed, NOT a dedicated 4-core cloud box; I have not measured the target box. My unmeasured estimate for a typical 4-vCPU cloud core is 1.5-2x slower: TF-IDF about 3-5 ms, BGE-small about 30-40 ms. Memory per process after load is recorded per model in `results/latency.csv` notes.

**Limits.**
- 200 of the 250 checked rows are consensus (easy) rows, so checked macro-F1 (0.90) is higher than silver dev (0.845) and likely optimistic about hard messages; the comparison between models is still paired on the same rows.
- All models are trained on silver train labels and inherit its errors; Phase 6 (optional) tests whether cleaning labels helps more than a bigger model.
- No hyper-parameter search. A higher or lower C, class weights, or fine-tuning might change the ranking; I did not try them. Class weighting is the obvious cheap test for `account_help` / `promo_code_issue`.
- Hub connection resets appeared during benchmark start-up ("Could not reach the Hub"): the models loaded from the local cache; metadata checks failed only.

**Where the AI was wrong.** Nothing wrong caught in this phase yet.

## Phase 5 — The out_of_scope decision
**Question.** "Unknown" can be produced two ways: (a) train out_of_scope as one of the 12 classes (the model has seen small talk, gibberish, other services) or (b) train only the 11 real intents and send a message to out_of_scope when the model is not confident. The starter did (b) with a fixed threshold, and it sent 54% of dev to out_of_scope.

**What was compared** (`src/oos.py`; all on the Phase 4 TF-IDF+LR model, C=10 fixed):
- V0: 12-class, plain argmax (trained out_of_scope class).
- V1: 11-class model trained WITHOUT out_of_scope rows; out_of_scope if max probability < t.
- V2: 12-class model; out_of_scope if max probability < t (low-confidence reject).
- V3: 12-class model; out_of_scope if P(out_of_scope) >= t.
The threshold t is chosen ONLY on train out-of-fold predictions (5-fold, seed 42, maximise macro-F1 there). Silver dev, the checked set and a 60-message probe set are report-only. Outputs: `results/oos_threshold.csv` (full sweep), `results/oos_summary.csv`, `results/oos_pr_curve.csv`, `data/oos_probes.csv`.

**Results at the train-OOF-chosen t** (`results/oos_summary.csv`):
| variant | t | silver dev macro-F1 | checked macro-F1 | silver dev OOS P / R | probe: novel OOS recall |
|---|---|---|---|---|---|
| V0 trained class, argmax | - | 0.845 | 0.894 | 0.882 / 0.926 | 0.733 |
| V1 11-class + confidence threshold | 0.50 | 0.759 | 0.795 | 0.292 / 0.254 | 0.433 |
| V2 12-class + confidence threshold | 0.15 | 0.845 | 0.894 | 0.882 / 0.926 | 0.733 |
| V3 12-class, P(oos) >= t | 0.30 | 0.848 | 0.894 | 0.895 / 0.950 | 0.733 |
- **Confidence is a bad out_of_scope detector here.** Average precision for finding out_of_scope on silver dev: P(out_of_scope) 0.898 (V3), 1 - max-probability from the 11-class model 0.218 (V1), from the 12-class model 0.085 (V2). Base rate is about 0.095, so V2's confidence carries no out_of_scope signal at all. V1 at its best threshold (0.5) has out_of_scope precision 0.29 and recall 0.25; to reach recall 0.80 it must send 49% of messages to out_of_scope (t = 0.9).
- **Why:** a linear model is confident when it sees familiar words. "swiggy delivered the wrong order to me" gets 0.99 confidence for wrong_item because of "wrong order"; confidence measures how sure the model is among the 11 classes, not whether the message belongs to any of them.
- **The trained class works** because the train data contains thousands of examples of what out_of_scope looks like (3,011 rows; includes the README's gift-card, loyalty, tax-invoice, tipping and feedback examples), so it learns that vocabulary directly.
- **V2 is V0.** At t = 0.15 the low-confidence reject almost never fires, so results are identical. V3 (t = 0.3) gains +0.003 macro-F1 on silver dev and 0 on checked and probes: inside the noise.
- **Probe set (60 messages I wrote, `data/oos_probes.csv`, NOT from the data and NOT human-checked):** 30 novel out_of_scope messages (gibberish, other companies, weather, taxes, Hinglish) and 30 novel in-domain ones. The trained class recalls 22 of 30 novel out_of_scope messages (0.733) and sends none of the in-domain ones to out_of_scope. All 8 misses: other companies' versions of in-domain complaints ("swiggy delivered the wrong order" -> wrong_item, "zomato has not refunded my money" -> refund_request, "order a pizza from dominos" -> refund_request) and a few short unfamiliar messages ("sing me a song" -> talk_to_agent, "kal ka mausam kaisa rahega" -> order_status). Lowering V3's threshold trades this against dev: t = 0.10 gives probe recall 0.833 and silver-dev macro-F1 0.835 (-0.013 vs 0.848); t = 0.05 gives 0.867 and 0.809.

**Decision: V0, the trained out_of_scope class with plain argmax, no threshold.** It wins or ties on every set, has no tuned parameter, and the principled "confidence threshold" alternative is clearly worse (D14). The remaining lever, a lower P(out_of_scope) threshold, is a policy choice (catch more unseen out_of_scope at a small cost in dev macro-F1); I did not pick it from the probes because they are my own 60 messages.

**Concepts in plain words.**
- *Confidence vs out-of-distribution:* a classifier's top probability says "how much better is the best class than the others", not "is this even my kind of message". That is why a threshold on it fails here.
- *Average precision:* area under the precision-recall curve for one yes/no task (here "is this out_of_scope"); 1.0 is perfect, the base rate (about 0.095) is chance.
- *Choosing the threshold on a different set than you report on:* t was fixed on train out-of-fold predictions, so dev and checked numbers are not tuned.

**Limits.** Probes are few (60), written by me, with my own labels, and the out_of_scope train rows come from the same generator as dev/test, so real unseen traffic is likely harder than dev suggests. The other-company messages are inherently hard for a bag-of-words model; a small list of competitor names as a rule is a possible next step (not tried).

**Where the AI was wrong.** Nothing wrong caught in this phase yet.

## Phase 7 — Predict and benchmark (Phase 6, the optional label-cleaning step, was skipped on Adi's call)
**What was done.**
- `src/final_model.py`: the final system = word+char TF-IDF + logistic regression, 12 classes with `out_of_scope` as a trained class (D11, D12, D14). Trained on `data/train.csv` only (silver labels); saved as `models/final_tfidf_lr.joblib` (3.5 MB; not meant for git, re-created by `python -m src.final_model`).
- `src/predict.py --name adithya` -> `submissions/adithya_predictions.csv` (3,000 rows). Test was opened here only to predict; no labels exist, none were inferred, nothing was tuned on it. Before predicting it re-scores the saved model on held-out dev: silver dev accuracy 0.8532, macro-F1 0.8448; checked accuracy 0.9160, macro-F1 0.9013 (identical to the Phase 4 row, so the submitted model is the one reported).
- `src/check_submission.py`: format check (columns exactly `id,intent`; 3,000 rows; ids = `test.csv` ids in the same order; no duplicates or blanks; all intents among the 12 taxonomy names). The submission passes. I also fed it two broken files (1,999 rows; wrong column name): it fails both with a clear message and a non-zero exit code. `data/test.csv` is unchanged against git.

**Latency of the final system** (`results/latency.csv`, model `final_system_saved`; protocol from CLAUDE.md: 4 threads, one message per call, 20 warm-up, 500 timed dev messages, loaded from disk, timed span = cleaning + vectorising + classifier):
- Three runs: p50 2.27-2.68 ms, p95 3.03-3.59 ms (the csv holds the last run: 2.27 / 3.17). The 15% run-to-run spread is laptop noise (efficiency cores, background load).
- Cold load of the model file 0.10 s; 3.5 MB on disk. Process memory in the benchmark was 300 MB, but that includes PyTorch which `latency.py` imports only to set the thread count; the production process needs scikit-learn and numpy only, so real memory should be lower (not measured separately).
- Hardware: 12th Gen Intel Core i5-1235U (10 cores / 12 threads), 15.7 GB RAM, Windows, no GPU; 4 threads allowed via OMP/MKL env vars. This is not a dedicated 4-core box. Unmeasured estimate for a typical 4-vCPU cloud machine: 3-5 ms per message (assuming 1.5-2x slower cores). Memory is far below the 15 GB ceiling (hundreds of MB at most).

**What runs where.**
- Experiments only: sentence-embedding models (CPU, this laptop, offline; MiniLM, BGE-small, E5-small, multilingual MiniLM), and two hosted open-weights LLMs on Groq for labelling (`openai/gpt-oss-120b`, `qwen/qwen3.8-27b`; about 500 calls on 250 repo messages, free tier, rate-limited to roughly 8 rows per minute per model, raw outputs in `results/llm_raw/`).
- Production: the single 3.5 MB TF-IDF+LR file on CPU. No GPU, no network call, no LLM. If an LLM were used at inference it would add a network round trip (hundreds of ms to seconds), a per-call cost, rate limits, and an outside dependency; none of that is needed for this accuracy.

**What the predicted test distribution says (descriptive; no labels used).** Predicted counts: order_status 417, delivery_delay 403, refund_request 331, out_of_scope 305, cancel_order 301, payment_failed 250, missing_item 225, change_address 185, talk_to_agent 185, wrong_item 145, promo_code_issue 134, account_help 119. The two smallest predicted classes are the two the model is weakest on in dev (account_help recall 0.65), so I expect the same under-prediction on test. I cannot verify this without the hidden labels.

**Reproduce everything (fixed seeds).** `python -m baseline.eval` (starter) -> `python -m src.evaluate` -> `python -m src.cv_flags` -> `python -m src.sample_dev` -> `python -m src.llm_label --model ...` (two models; needs `GROQ_API_KEY` in `.env`) -> `python -m src.label_noise --stage review`, fill `my_label` in `data/dev_review.csv`, `--stage final` -> `python -m src.build_taxonomy_v2` -> `python -m src.taxonomy_eval` -> `python -m src.embed_cache` -> `python -m src.run_models` -> `python -m src.latency --model <name>` -> `python -m src.oos` -> `python -m src.final_model` -> `python -m src.predict --name adithya` -> `python -m src.check_submission submissions/adithya_predictions.csv`.

**Limits.** The model is trained on silver labels and inherits their ~12% noise; label cleaning (Phase 6) was not done, so "cleaner labels vs a bigger model" stays untested. Test accuracy is unknown; my estimates come from silver dev (0.845 macro-F1) and the checked set (0.901, optimistic because 200 of its 250 rows are consensus rows). The submission name `adithya` is my guess from the git user; rename the file if the task wants another name.

**Where the AI was wrong.** The format checker crashed with a traceback on a file with a wrong column name instead of reporting FORMAT FAILED; caught by my own negative test, fixed before submitting.

## Concepts
(Add as they come up: macro vs weighted F1, Wilson interval, calibration, cross-validated disagreement, char n-grams, ...)

## Where the AI was wrong
(Record every caught mistake: what it said, how it was caught, what was right.)
