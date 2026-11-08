"""Fixed evaluation for the QuickCart intent task (Phase 1).

Fixes vs baseline/eval.py:
  * every sklearn call gets labels=intents (taxonomy order) -> per-class rows are attributed to the right class
  * macro-F1 is average="macro" (the starter used "weighted")
  * confusion matrix is saved with row/column names
  * top-3 / MRR are computed AFTER the out_of_scope threshold decision
  * slices: length, Hinglish, multi-request, contains-taxonomy-example

Reusable API: evaluate(y_true, y_pred, utterances, intents) -> dict of DataFrames.
Run:  python -m src.evaluate        (re-scores the unmodified starter, writes results/)
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from baseline.baseline import DATA, THRESHOLD, TaxonomyMatcher, clean

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OOS = "out_of_scope"

# ---- slice heuristics (word lists / regex; imperfect, limits documented in results/eval_bugs.md) ----
# Only words that are unambiguous Hindi/Hinglish (not also common English words like "me", "do", "order", "ho").
HINGLISH = {
    "mera", "meri", "mere", "mujhe", "nahi", "nahin", "nhi", "kahan", "kab", "kya", "kyun", "kyu", "hai", "hain",
    "hua", "hui", "gaya", "gayi", "gaye", "abhi", "bhai", "yaar", "paisa", "paise", "wapas", "karo", "karna",
    "chahiye", "chahie", "aur", "mein", "bhi", "toh", "kaise", "kitna", "jaldi", "bahut", "bohot", "wala", "wali",
    "bhej", "bhejo", "dikha", "dikhao", "batao", "bata", "karwao", "lagega", "ghar", "khana", "hota", "raha",
    "rahi", "tha", "thi", "pura", "nahi",
}
MULTI = re.compile(
    r"\b(also|and|but|plus|then|aur|before that|after that|as well|too|actually|never ?mind|another thing|"
    r"one more|btw|bhi)\b")


def _tokens(t):
    return re.findall(r"[a-z]+", clean(t))


def length_bucket(t):
    n = len(clean(t).split())
    return "short(<=6w)" if n <= 6 else ("medium(7-14w)" if n <= 14 else "long(>=15w)")


def is_hinglish(t):
    return any(w in HINGLISH for w in _tokens(t))


def is_multi(t):
    return bool(MULTI.search(clean(t)))


def contains_example(t, examples_clean):
    c = clean(t)
    return any(e in c for e in examples_clean)


def taxonomy_examples(path=None):
    tax = json.load(open(path or DATA / "taxonomy.json", encoding="utf-8"))
    return [clean(e) for i in tax["intents"] for e in i["examples"]]


# ---- core metrics ----
def prf(y_true, y_pred, intents):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=intents, zero_division=0)
    return pd.DataFrame({"intent": intents, "precision": p, "recall": r, "f1": f, "support": s})


def macro_f1(y_true, y_pred, intents):
    return float(f1_score(y_true, y_pred, labels=intents, average="macro", zero_division=0))


def ranking(scores, intents, y_true, threshold=None, k=3):
    """Top-k / MRR. With a threshold, out_of_scope is ranked first when best score < threshold
    (what the deployed system outputs); otherwise ranks follow raw scores (what the starter reported)."""
    idx = {n: i for i, n in enumerate(intents)}
    oi = idx[OOS]
    ranks = []
    for s, y in zip(scores, y_true):
        order = list(np.argsort(-s))
        if threshold is not None and s.max() < threshold:
            order.remove(oi)
            order = [oi] + order
        ranks.append(order.index(idx[y]) + 1)
    ranks = np.array(ranks)
    return float((ranks <= k).mean()), float((1.0 / ranks).mean())


def evaluate(y_true, y_pred, utterances, intents, example_path=None):
    """Return dict(overall, per_class, confusion, slices). Same format for every later experiment."""
    y_true, y_pred = list(y_true), list(y_pred)
    per_class = prf(y_true, y_pred, intents)
    overall = {"n": len(y_true), "accuracy": accuracy_score(y_true, y_pred), "macro_f1": macro_f1(y_true, y_pred, intents)}
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=intents), index=[f"true:{i}" for i in intents],
                      columns=[f"pred:{i}" for i in intents])
    ex = taxonomy_examples(example_path)
    utt = pd.Series(list(utterances))
    slices = {
        "length": utt.map(length_bucket),
        "hinglish": utt.map(lambda t: "hinglish" if is_hinglish(t) else "english_only"),
        "multi_request": utt.map(lambda t: "multi" if is_multi(t) else "single"),
        "contains_taxonomy_example": utt.map(lambda t: "contains_example" if contains_example(t, ex) else "no_example"),
    }
    yt, yp = np.array(y_true), np.array(y_pred)
    rows = []
    for sname, labels in slices.items():
        for val in sorted(labels.unique()):
            m = (labels == val).to_numpy()
            rows.append({"slice": sname, "value": val, "n": int(m.sum()), "share": m.mean(),
                         "accuracy": accuracy_score(yt[m], yp[m]), "macro_f1": macro_f1(yt[m], yp[m], intents)})
    return {"overall": overall, "per_class": per_class, "confusion": cm, "slices": pd.DataFrame(rows)}


def main():
    dev = pd.read_csv(DATA / "dev.csv")
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    m = TaxonomyMatcher()
    intents = m.intents
    pred, scores = m.predict(dev.utterance.tolist())
    y = dev.intent.tolist()
    res = evaluate(y, pred, dev.utterance, intents)
    ov = res["overall"]
    rep = json.load(open(RESULTS / "baseline_dev.json"))          # the buggy numbers, as shipped
    alpha = sorted(intents)                                        # order sklearn actually used in the starter report
    t3_raw, mrr_raw = ranking(scores, intents, y)
    t3_th, mrr_th = ranking(scores, intents, y, threshold=THRESHOLD)

    # ---- results/baseline_fixed.csv ----
    rows = [
        {"level": "overall", "name": "accuracy", "n": ov["n"], "fixed": ov["accuracy"], "starter_reported": rep["top1_accuracy"]},
        {"level": "overall", "name": "macro_f1", "n": ov["n"], "fixed": ov["macro_f1"], "starter_reported": rep["macro_f1"]},
        {"level": "overall", "name": "top3_after_threshold", "n": ov["n"], "fixed": t3_th, "starter_reported": rep["top3_accuracy"]},
        {"level": "overall", "name": "mrr_after_threshold", "n": ov["n"], "fixed": mrr_th, "starter_reported": rep["mrr"]},
    ]
    for _, r in res["per_class"].iterrows():
        shown = rep["per_class"][r.intent]            # what the starter printed under this class's name
        rows.append({"level": "class", "name": r.intent, "n": int(r.support), "precision": r.precision,
                     "recall": r.recall, "f1": r.f1,
                     "starter_reported_precision": shown["precision"], "starter_reported_recall": shown["recall"],
                     "starter_reported_f1": shown["f1-score"], "starter_reported_support": int(shown["support"]),
                     "starter_row_really_belongs_to": alpha[intents.index(r.intent)]})
    for _, r in res["slices"].iterrows():
        rows.append({"level": f"slice:{r.slice}", "name": r.value, "n": int(r.n), "share": r.share,
                     "fixed": r.accuracy, "f1": r.macro_f1,
                     "note": "fixed=accuracy, f1=macro-F1 on the slice"})
    pd.DataFrame(rows).round(4).to_csv(RESULTS / "baseline_fixed.csv", index=False)
    res["confusion"].to_csv(RESULTS / "baseline_confusion.csv")

    # ---- numbers for results/eval_bugs.md ----
    wf1 = float(f1_score(y, pred, average="weighted", zero_division=0))
    n_oos_pred = sum(p == OOS for p in pred)
    pc = res["per_class"].set_index("intent")
    ex = taxonomy_examples()
    overlap = {nm: sum(contains_example(t, ex) for t in df.utterance) for nm, df in [("train", train), ("dev", dev), ("test", test)]}
    # bug 8: refit the vectoriser on train + taxonomy only (no dev), same hyper-parameters
    m2 = TaxonomyMatcher()
    corpus = [clean(t) for t in train.utterance] + ex
    m2.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(corpus)
    m2.centroids = m2._build_centroids()
    pred2, _ = m2.predict(dev.utterance.tolist())
    acc2, f12 = accuracy_score(y, pred2), macro_f1(y, pred2, intents)
    dsh, tsh = dev.intent.value_counts(normalize=True), train.intent.value_counts(normalize=True)
    tax = json.load(open(DATA / "taxonomy.json", encoding="utf-8"))
    seen, dups = {}, []
    for i in tax["intents"]:
        for e in i["examples"]:
            k = clean(e)
            if k in seen and seen[k] != i["name"]:
                dups.append(f"'{e}' in {seen[k]} and {i['name']}")
            seen[k] = i["name"]
    zero = [i for i in intents if pc.loc[i, "recall"] == 0]

    md = f"""# Evaluation bugs in the starter (generated by `python -m src.evaluate`)

All numbers come from `src/evaluate.py` on dev (n={len(dev)}, silver labels) with the unmodified starter
(`baseline/baseline.py`, threshold {THRESHOLD}). Tables: `results/baseline_fixed.csv`; labelled confusion matrix: `results/baseline_confusion.csv`.

| # | Bug | Evidence | Effect |
|---|---|---|---|
| 1 | Per-class rows attributed to the wrong class | `baseline/eval.py` passes `target_names` in taxonomy order, but sklearn sorts labels alphabetically. The row shown as `order_status` (support {int(rep['per_class']['order_status']['support'])}) really belongs to `{alpha[0]}`. Column `starter_row_really_belongs_to` in baseline_fixed.csv gives the real class of every shown row. | Every per-class number in `baseline_dev.json` describes a different class than its name |
| 2 | "macro_f1" is weighted F1 | `f1_score(average="weighted")`, `baseline/eval.py:36` | Reported {rep['macro_f1']:.3f}; weighted recomputed {wf1:.3f}; true macro-F1 **{ov['macro_f1']:.3f}** |
| 3 | Confusion matrix saved without `labels=` | `confusion_matrix(y, pred)`: sorted-label order, no names | Unreadable as saved; now `results/baseline_confusion.csv` with names |
| 4 | Two intents can never be predicted | `promo_code_issue`, `account_help` have no taxonomy examples -> zero centroids. Classes with recall exactly 0 in the fixed report: {', '.join(zero) or 'none'} | 419 dev rows (10.5%) can never be correct |
| 4b | The starter's zero-recall rows were the wrong classes | Starter printed 0.00 recall for the rows named `order_status` and `change_address`, which really belong to `{alpha[0]}` and `{alpha[8]}` | Looked like two healthy-class failures while the real ones were invisible |
| 5 | Threshold {THRESHOLD} floods `out_of_scope` | {n_oos_pred} of {len(dev)} dev rows predicted `out_of_scope` ({n_oos_pred/len(dev):.1%}); true `out_of_scope` rows: {int((dev.intent==OOS).sum())} | `out_of_scope` precision {pc.loc[OOS,'precision']:.2f}, recall {pc.loc[OOS,'recall']:.2f} |
| 7 | Data contains taxonomy examples verbatim | rows containing a normalised taxonomy example: train {overlap['train']}/{len(train)}, dev {overlap['dev']}/{len(dev)}, test {overlap['test']}/{len(test)} (counts only, no test labels used) | Matching scores on those rows overstate generalisation; see slice rows in baseline_fixed.csv |
| 8 | Vectoriser vocabulary/IDF fitted on train **and dev** | `baseline/baseline.py:36`. Refit on train + taxonomy only: accuracy {ov['accuracy']:.4f} -> {acc2:.4f}, macro-F1 {ov['macro_f1']:.4f} -> {f12:.4f} | Size of the dev leak on the starter (unsupervised, so expected to be small) |
| 9 | Top-3 / MRR ignore the threshold | Starter ranks raw scores. With out_of_scope ranked first when best < {THRESHOLD}: top-3 {t3_raw:.4f} -> {t3_th:.4f}, MRR {mrr_raw:.4f} -> {mrr_th:.4f} | Ranking metrics now describe the deployed system |
| 10 | Train/dev class shift | `promo_code_issue` train {tsh['promo_code_issue']:.1%} vs dev {dsh['promo_code_issue']:.1%}; `account_help` train {tsh['account_help']:.1%} vs dev {dsh['account_help']:.1%} | Matters from Phase 4 (training) |

Note on bug 4: both intents have all-zero centroids, so their cosine is always 0 and they never win the argmax against a positive score. Fixed recall is exactly 0 for them (rows in baseline_fixed.csv). The starter's report hid this: it showed 0.00 under the names `order_status` and `change_address` instead.

Bug 6 (taxonomy contradictions) and 11 (silver labels breaking the rules) are handled in Phases 2-3. Same example text under two different intents in `taxonomy.json`: {dups or 'none'}.

## Reproduced vs shipped
Shipped `results/baseline_dev.json` has top-3 0.7192 / MRR 0.6652; the raw-score recompute here gives {t3_raw:.4f} / {mrr_raw:.4f}. The gap (<0.001) is unexplained; probably tie-breaking among tied scores across platforms (not investigated).

## Slice heuristics and their limits
- length: words after `clean()`; short <=6, medium 7-14, long >=15.
- Hinglish: message contains any token from a hand-written list of unambiguous Hindi words (`HINGLISH` in `src/evaluate.py`). Misses Hinglish made only of rare words; not validated against human labels.
- multi-request: regex for connectors ("also", "and", "but", "aur", "then", "actually"...). "and" also joins non-requests, so this over-counts.
- contains-taxonomy-example: normalised taxonomy example is a substring of the normalised message.
"""
    (RESULTS / "eval_bugs.md").write_text(md, encoding="utf-8")

    print(f"accuracy={ov['accuracy']:.4f} macro_f1={ov['macro_f1']:.4f} weighted_f1={wf1:.4f} "
          f"top3 raw/th={t3_raw:.4f}/{t3_th:.4f} mrr raw/th={mrr_raw:.4f}/{mrr_th:.4f}")
    print(res["per_class"].round(3).to_string(index=False))
    print(res["slices"].round(3).to_string(index=False))
    print("overlap", overlap, "| bug8 refit acc/f1", round(acc2, 4), round(f12, 4), "| oos predicted", n_oos_pred, "| dups", dups)


if __name__ == "__main__":
    main()
