"""Phase 5: how should out_of_scope be produced?  Trained class vs confidence threshold.

Variants (all on the word+char TF-IDF + LR model chosen in Phase 4, C=10 fixed):
  V0  12-class model, plain argmax (out_of_scope is a trained class)
  V1  11-class model trained WITHOUT out_of_scope rows; predict out_of_scope if max prob < t
  V2  12-class model; predict out_of_scope if max prob < t (low-confidence reject), else argmax
  V3  12-class model; predict out_of_scope if P(out_of_scope) >= t, else best other class

Threshold t is chosen ONLY on train out-of-fold predictions (5-fold, seed 42; maximises macro-F1 there).
Dev silver, the checked set and the 60 hand-written probes (data/oos_probes.csv) are report-only.

Writes results/oos_threshold.csv (full sweep, long format), results/oos_summary.csv (chosen operating points),
results/oos_pr_curve.csv (precision-recall points for detecting out_of_scope).
Run: python -m src.oos
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold

from baseline.baseline import DATA
from src.evaluate import evaluate
from src.models import SEED, TfidfLR

RESULTS = DATA.parent / "results"
OOS = "out_of_scope"
GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


def fit_pair(texts, y, intents):
    """12-class model and an 11-class model (no OOS rows) sharing one TF-IDF fit."""
    m12 = TfidfLR().fit(texts, y)
    X = m12._x(texts)
    keep = np.array(y) != OOS
    clf11 = LogisticRegression(C=10, max_iter=300, random_state=SEED).fit(X[keep], np.array(y)[keep])
    return m12, clf11


def probs(m12, clf11, texts, intents):
    X = m12._x(texts)
    p12 = np.zeros((len(texts), len(intents)))
    p12[:, [intents.index(c) for c in m12.classes_]] = m12.clf.predict_proba(X)
    p11 = np.zeros_like(p12)
    p11[:, [intents.index(c) for c in clf11.classes_]] = clf11.predict_proba(X)
    return p12, p11


def decide(variant, t, p12, p11, intents):
    oi = intents.index(OOS)
    A = np.array(intents)
    if variant == "V0":
        return list(A[p12.argmax(1)])
    if variant == "V1":
        return [OOS if p.max() < t else A[p.argmax()] for p in p11]
    if variant == "V2":
        return [OOS if p.max() < t else A[p.argmax()] for p in p12]
    if variant == "V3":
        q = p12.copy()
        q[:, oi] = -1
        return [OOS if p[oi] >= t else A[qq.argmax()] for p, qq in zip(p12, q)]
    raise ValueError(variant)


def score_oos(variant, p12, p11, intents):
    """Continuous 'is out_of_scope' score for the PR curve."""
    oi = intents.index(OOS)
    return {"V0": p12[:, oi], "V1": 1 - p11.max(1), "V2": 1 - p12.max(1), "V3": p12[:, oi]}[variant]


def metrics(y, pred, utt, intents):
    r = evaluate(y, pred, utt, intents)
    pc = r["per_class"].set_index("intent").loc[OOS]
    return {"accuracy": r["overall"]["accuracy"], "macro_f1": r["overall"]["macro_f1"], "oos_precision": pc.precision,
            "oos_recall": pc.recall, "oos_f1": pc.f1, "share_pred_oos": float(np.mean([p == OOS for p in pred]))}


def main():
    intents = [i["name"] for i in json.load(open(DATA / "taxonomy_v2.json", encoding="utf-8"))["intents"]]
    train, dev = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "dev.csv")
    chk, probe = pd.read_csv(DATA / "dev_checked.csv"), pd.read_csv(DATA / "oos_probes.csv")
    tt, ty = train.utterance.tolist(), train.intent.to_numpy()

    # ---- train out-of-fold probabilities (threshold selection data) ----
    P12, P11 = np.zeros((len(train), 12)), np.zeros((len(train), 12))
    for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(tt, ty)):
        m12, c11 = fit_pair([tt[i] for i in a], ty[a].tolist(), intents)
        P12[b], P11[b] = probs(m12, c11, [tt[i] for i in b], intents)
        print("oof fold", k, flush=True)

    # ---- final models on all train, probabilities on report-only sets ----
    m12, c11 = fit_pair(tt, ty.tolist(), intents)
    sets = {"train_oof(silver)": (train.utterance, train.intent, P12, P11)}
    for nm, df, ycol in [("silver_dev", dev, "intent"), ("checked_dev", chk, "checked_intent")]:
        p12, p11 = probs(m12, c11, df.utterance.tolist(), intents)
        sets[nm] = (df.utterance, df[ycol], p12, p11)
    p12, p11 = probs(m12, c11, probe.utterance.tolist(), intents)
    sets["probes_60"] = (probe.utterance, probe.intent, p12, p11)

    rows = []
    for v in ["V0", "V1", "V2", "V3"]:
        for t in ([None] if v == "V0" else GRID):
            for nm, (u, y, p12, p11) in sets.items():
                pred = decide(v, t, p12, p11, intents)
                rows.append({"variant": v, "threshold": t, "eval_set": nm, "n": len(y), **metrics(list(y), pred, list(u), intents)})
    sweep = pd.DataFrame(rows)
    sweep.round(4).to_csv(RESULTS / "oos_threshold.csv", index=False)

    # ---- choose t on train OOF only, then read the report-only sets ----
    names = {"V0": "12-class argmax (trained out_of_scope class)", "V1": "11-class + reject if max prob < t",
             "V2": "12-class + reject if max prob < t", "V3": "12-class, out_of_scope if P(oos) >= t"}
    summ = []
    for v in ["V0", "V1", "V2", "V3"]:
        s = sweep[(sweep.variant == v) & (sweep.eval_set == "train_oof(silver)")]
        best = s.loc[s.macro_f1.idxmax()]
        t = None if v == "V0" else float(best.threshold)
        row = {"variant": v, "description": names[v], "threshold_chosen_on_train_oof": t}
        for nm in sets:
            r = sweep[(sweep.variant == v) & (sweep.eval_set == nm) & ((sweep.threshold == t) if t is not None else sweep.threshold.isna())].iloc[0]
            tag = nm.split("(")[0]
            row.update({f"{tag}_macro_f1": r.macro_f1, f"{tag}_accuracy": r.accuracy, f"{tag}_oos_precision": r.oos_precision,
                        f"{tag}_oos_recall": r.oos_recall})
        # probes: out_of_scope recall on novel OOS, and in-domain messages wrongly sent to out_of_scope
        u, y, p12, p11 = sets["probes_60"]
        pred = np.array(decide(v, t, p12, p11, intents))
        yy = np.array(list(y))
        row["probe_novel_oos_recall"] = float((pred[yy == OOS] == OOS).mean())
        row["probe_in_domain_sent_to_oos"] = float((pred[yy != OOS] == OOS).mean())
        row["probe_in_domain_accuracy"] = float((pred[yy != OOS] == yy[yy != OOS]).mean())
        summ.append(row)
    S = pd.DataFrame(summ)
    S.round(4).to_csv(RESULTS / "oos_summary.csv", index=False)

    # ---- precision-recall curves for detecting out_of_scope (score-based variants) ----
    pr = []
    for v in ["V1", "V2", "V3"]:        # V0 has no score to sweep; V3 uses the same P(oos) score
        for nm in ["train_oof(silver)", "silver_dev", "checked_dev", "probes_60"]:
            u, y, p12, p11 = sets[nm]
            yt = (np.array(list(y)) == OOS).astype(int)
            sc = score_oos(v, p12, p11, intents)
            prec, rec, thr = precision_recall_curve(yt, sc)
            ap = average_precision_score(yt, sc)
            for pp, rr in list(zip(prec, rec))[::max(1, len(prec) // 60)]:
                pr.append({"variant": v, "eval_set": nm, "precision": round(pp, 4), "recall": round(rr, 4), "average_precision": round(ap, 4)})
    pd.DataFrame(pr).to_csv(RESULTS / "oos_pr_curve.csv", index=False)
    pd.set_option("display.width", 250)
    print(S.round(3).T.to_string())
    ap = pd.DataFrame(pr).groupby(["variant", "eval_set"]).average_precision.first().unstack().round(3)
    print(ap.to_string())


if __name__ == "__main__":
    main()
