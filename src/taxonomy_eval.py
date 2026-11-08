"""Phase 3b: taxonomy v1 vs v2 with the (unchanged-method) matcher, plus intent-distance vs confusion.

Writes:
  results/taxonomy_before_after.csv   one row per (experiment, eval set): accuracy, macro-F1, per-class F1, zero-recall classes, share predicted out_of_scope
  results/intent_distance.csv         one row per intent pair: taxonomy-centroid cosine (v1, v2), train-centroid cosine,
                                      matcher confusion rate (v1, v2), CV confusion rate on train
  results/intent_distance_correlation.csv   Spearman correlations between similarity and confusion

Eval sets: silver dev (4,000) and the checked dev sample (250, data/dev_checked.csv). Nothing here is tuned on either.
Run: python -m src.taxonomy_eval
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix

from baseline.baseline import DATA, clean
from src.evaluate import evaluate
from src.matcher import OOS, Matcher

RESULTS = Path(__file__).resolve().parent.parent / "results"
V1, V2 = DATA / "taxonomy.json", DATA / "taxonomy_v2.json"

def _ablations():
    """In-memory variants separating the two kinds of edit (both use the train-only vectoriser, thr 0.10)."""
    import copy
    import json
    from src.build_taxonomy_v2 import ADD
    v1, v2 = json.load(open(V1, encoding="utf-8")), json.load(open(V2, encoding="utf-8"))
    fill_only = copy.deepcopy(v1)                       # v1 + examples for the two empty intents, nothing removed/moved
    for i in fill_only["intents"]:
        if i["name"] in ADD:
            i["examples"] = list(ADD[i["name"]])
    clean_only = copy.deepcopy(v2)                      # v2 with the two filled intents emptied again
    for i in clean_only["intents"]:
        if i["name"] in ADD:
            i["examples"] = []
    return fill_only, clean_only


FILL_ONLY, CLEAN_ONLY = _ablations()
EXPERIMENTS = [  # name, taxonomy, fit_on_dev, threshold
    ("A v1 as shipped (vectoriser on train+dev, thr 0.10)", V1, True, 0.10),
    ("B v1, vectoriser on train only, thr 0.10", V1, False, 0.10),
    ("C v2, vectoriser on train only, thr 0.10", V2, False, 0.10),
    ("D v1, train-only, no threshold (argmax)", V1, False, None),
    ("E v2, train-only, no threshold (argmax)", V2, False, None),
    ("F ablation: v1 + fill the 2 empty intents only, thr 0.10", FILL_ONLY, False, 0.10),
    ("G ablation: v2 cleanups only (2 intents left empty), thr 0.10", CLEAN_ONLY, False, 0.10),
]


def main():
    dev = pd.read_csv(DATA / "dev.csv")
    chk = pd.read_csv(DATA / "dev_checked.csv")
    rows, preds, mats = [], {}, {}
    for name, tax, fit_dev, thr in EXPERIMENTS:
        m = Matcher(tax, threshold=thr, fit_on_dev=fit_dev)
        mats[name] = m
        for set_name, df, ycol in [("silver_dev_4000", dev, "intent"), ("checked_dev_250", chk, "checked_intent")]:
            pred, _ = m.predict(df.utterance.tolist())
            if set_name == "silver_dev_4000":
                preds[name] = pred
            r = evaluate(df[ycol].tolist(), pred, df.utterance, m.intents)
            pc = r["per_class"].set_index("intent")
            row = {"experiment": name, "eval_set": set_name, "n": len(df), "accuracy": round(r["overall"]["accuracy"], 4),
                   "macro_f1": round(r["overall"]["macro_f1"], 4),
                   "zero_recall_classes": ";".join(pc.index[pc.recall == 0]) or "none",
                   "share_predicted_oos": round(float(np.mean([p == OOS for p in pred])), 4)}
            for i in m.intents:
                row[f"f1_{i}"] = round(float(pc.loc[i, "f1"]), 3)
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "taxonomy_before_after.csv", index=False)
    pd.set_option("display.width", 250)
    print(out[["experiment", "eval_set", "accuracy", "macro_f1", "zero_recall_classes", "share_predicted_oos"]].to_string(index=False))

    # ---- intent distance vs confusion ----
    intents = mats[EXPERIMENTS[0][0]].intents
    sim1 = mats[EXPERIMENTS[1][0]].centroid_similarity()      # v1, train-only vectoriser (same vectoriser for v1/v2 comparison)
    sim2 = mats[EXPERIMENTS[2][0]].centroid_similarity()
    # train-data centroids with the same vectoriser: how close are the intents by what customers actually write
    mv = mats[EXPERIMENTS[1][0]]
    train = pd.read_csv(DATA / "train.csv")
    X = mv.vec.transform([clean(t) for t in train.utterance])
    from sklearn.preprocessing import normalize
    tc = normalize(np.vstack([np.asarray(X[(train.intent == i).to_numpy()].mean(axis=0)).ravel() for i in intents]))
    simt = pd.DataFrame(tc @ tc.T, index=intents, columns=intents)

    def conf_rate(pred, y):
        C = pd.DataFrame(confusion_matrix(y, pred, labels=intents), index=intents, columns=intents)
        n = C.sum(1)
        return lambda a, b: (C.loc[a, b] + C.loc[b, a]) / (n[a] + n[b])

    y = dev.intent.tolist()
    c1, c2 = conf_rate(preds[EXPERIMENTS[3][0]], y), conf_rate(preds[EXPERIMENTS[4][0]], y)   # no-threshold, v1 / v2
    flags = pd.read_csv(RESULTS / "cv_flags_train.csv")
    ntr = train.intent.value_counts()
    cvc = flags.groupby(["silver", "oof_pred"]).size()
    cvrate = lambda a, b: (cvc.get((a, b), 0) + cvc.get((b, a), 0)) / (ntr[a] + ntr[b])
    pairs = []
    for a, b in combinations(intents, 2):
        pairs.append({"intent_a": a, "intent_b": b, "sim_taxonomy_v1": round(sim1.loc[a, b], 4),
                      "sim_taxonomy_v2": round(sim2.loc[a, b], 4), "sim_train_centroids": round(simt.loc[a, b], 4),
                      "matcher_confusion_v1": round(c1(a, b), 4), "matcher_confusion_v2": round(c2(a, b), 4),
                      "cv_confusion_train": round(cvrate(a, b), 4)})
    P = pd.DataFrame(pairs)
    P.to_csv(RESULTS / "intent_distance.csv", index=False)
    empty = {"promo_code_issue", "account_help"}
    P1 = P[~(P.intent_a.isin(empty) | P.intent_b.isin(empty))]      # v1 has zero centroids for the two empty intents
    corr = []
    for nm, d, x, yv in [("taxonomy_v1 sim vs matcher_v1 confusion (45 pairs without the 2 empty intents)", P1, "sim_taxonomy_v1", "matcher_confusion_v1"),
                         ("taxonomy_v2 sim vs matcher_v2 confusion (66 pairs)", P, "sim_taxonomy_v2", "matcher_confusion_v2"),
                         ("train-centroid sim vs CV confusion on train (66 pairs)", P, "sim_train_centroids", "cv_confusion_train"),
                         ("train-centroid sim vs matcher_v2 confusion (66 pairs)", P, "sim_train_centroids", "matcher_confusion_v2")]:
        rho, p = spearmanr(d[x], d[yv])
        corr.append({"comparison": nm, "pairs": len(d), "spearman_rho": round(rho, 3), "p_value": round(p, 4)})
    pd.DataFrame(corr).to_csv(RESULTS / "intent_distance_correlation.csv", index=False)
    print(pd.DataFrame(corr).to_string(index=False))
    print("\ntop pairs by train-centroid similarity:")
    print(P.sort_values("sim_train_centroids", ascending=False).head(8).to_string(index=False))
    print("\ntop pairs by CV confusion:")
    print(P.sort_values("cv_confusion_train", ascending=False).head(8).to_string(index=False))


if __name__ == "__main__":
    main()
