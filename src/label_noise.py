"""Phase 2d: silver-vs-checked agreement on the 250-row dev sample.

Stage `review`  : merge silver label + two LLM labels -> results/llm_vs_silver.csv (agreement tables)
                  and data/dev_review.csv (rows Adi must adjudicate: any disagreement among silver, model A, model B).
Stage `final`   : after Adi fills `my_label` (and `error_type`) in data/dev_review.csv ->
                  data/dev_checked.csv and results/label_noise.csv (agreement + 95% Wilson interval, per class, error types).

Checked label of a row = Adi's label if the row was in the review file; else the label on which silver and both models agree.
Rows where all three agreed are NOT human-checked (stated in the write-up).

Run: python -m src.label_noise --stage review | final
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from baseline.baseline import DATA, load_taxonomy

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RAW = RESULTS / "llm_raw"
MODEL_A, MODEL_B = "openai/gpt-oss-120b", "qwen/qwen3.8-27b"


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def load_llm(model):
    f = RAW / f"{model.replace('/', '_')}__dev_sample.jsonl"
    return {r["dev_row"]: r for r in map(json.loads, open(f, encoding="utf-8"))}


def weighted_rate(hit, w):
    """Weighted agreement rate and Wilson interval using Kish effective sample size."""
    hit, w = np.asarray(hit, float), np.asarray(w, float)
    p = float((hit * w).sum() / w.sum())
    n_eff = float(w.sum() ** 2 / (w ** 2).sum())
    lo, hi = wilson(p * n_eff, n_eff)
    return p, lo, hi


def agreement_row(name, a, b, df):
    hit = (df[a] == df[b]).to_numpy()
    lo, hi = wilson(hit.sum(), len(hit))
    wp, wlo, whi = weighted_rate(hit, df.weight)
    return {"comparison": name, "n": len(df), "agree": int(hit.sum()), "rate_unweighted": round(hit.mean(), 4),
            "ci95_low": round(lo, 4), "ci95_high": round(hi, 4), "rate_dev_weighted": round(wp, 4),
            "wtd_ci95_low": round(wlo, 4), "wtd_ci95_high": round(whi, 4)}


def merged():
    s = pd.read_csv(DATA / "dev_sample.csv")
    A, B = load_llm(MODEL_A), load_llm(MODEL_B)
    s["model_a"] = s.dev_row.map(lambda r: (A.get(r) or {}).get("label"))
    s["model_b"] = s.dev_row.map(lambda r: (B.get(r) or {}).get("label"))
    s["reason_a"] = s.dev_row.map(lambda r: (json.loads(A[r]["raw_text"]).get("reason", "") if A.get(r) and A[r]["label"] else ""))
    s["reason_b"] = s.dev_row.map(lambda r: (json.loads(B[r]["raw_text"]).get("reason", "") if B.get(r) and B[r]["label"] else ""))
    return s


def stage_review():
    s = merged()
    bad = s.model_a.isna() | s.model_b.isna()
    print("rows with a missing/invalid model label:", int(bad.sum()))
    ok = s[~bad]
    rows = [agreement_row("silver vs model_a (gpt-oss-120b)", "silver_intent", "model_a", ok),
            agreement_row("silver vs model_b (qwen3.8-27b)", "silver_intent", "model_b", ok),
            agreement_row("model_a vs model_b", "model_a", "model_b", ok)]
    allthree = ok[(ok.silver_intent == ok.model_a) & (ok.model_a == ok.model_b)]
    rows.append({"comparison": "silver = model_a = model_b", "n": len(ok), "agree": len(allthree),
                 "rate_unweighted": round(len(allthree) / len(ok), 4)})
    pd.DataFrame(rows).to_csv(RESULTS / "llm_vs_silver.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    # per-class agreement of silver with each model
    pc = []
    for c, g in ok.groupby("silver_intent"):
        pc.append({"silver_class": c, "n": len(g), "silver_eq_model_a": int((g.silver_intent == g.model_a).sum()),
                   "silver_eq_model_b": int((g.silver_intent == g.model_b).sum()),
                   "models_agree": int((g.model_a == g.model_b).sum())})
    pd.DataFrame(pc).to_csv(RESULTS / "llm_vs_silver_by_class.csv", index=False)
    rev = ok[~((ok.silver_intent == ok.model_a) & (ok.model_a == ok.model_b))].copy()
    rev["my_label"] = ""
    rev["error_type"] = ""
    rev = rev[["dev_row", "utterance", "silver_intent", "model_a", "model_b", "reason_a", "reason_b", "my_label", "error_type"]]
    rev.to_csv(DATA / "dev_review.csv", index=False, encoding="utf-8-sig")
    print(f"review file: {len(rev)} rows -> data/dev_review.csv")


def stage_final():
    intents, _ = load_taxonomy()
    s = merged()
    rev = pd.read_csv(DATA / "dev_review.csv")
    rev["my_label"] = rev.my_label.fillna("").str.strip()
    missing = rev[rev.my_label == ""]
    if len(missing):
        raise SystemExit(f"{len(missing)} review rows still have no my_label")
    bad = set(rev.my_label) - set(intents)
    if bad:
        raise SystemExit(f"invalid labels in my_label: {bad}")
    m = s.merge(rev[["dev_row", "my_label", "error_type"]], on="dev_row", how="left")
    m["human_checked"] = m.my_label.notna()
    m["checked_intent"] = np.where(m.human_checked, m.my_label, m.silver_intent)
    m[["dev_row", "utterance", "silver_intent", "checked_intent", "human_checked", "model_a", "model_b", "error_type", "weight"]] \
        .to_csv(DATA / "dev_checked.csv", index=False, encoding="utf-8-sig")
    m["agree"] = m.silver_intent == m.checked_intent
    rows = []
    p, lo, hi = weighted_rate(m.agree, m.weight)
    k = int(m.agree.sum())
    ulo, uhi = wilson(k, len(m))
    rows.append({"level": "overall", "name": "silver == checked", "n": len(m), "agree": k, "rate": round(k / len(m), 4),
                 "ci95_low": round(ulo, 4), "ci95_high": round(uhi, 4), "rate_dev_weighted": round(p, 4),
                 "wtd_ci95_low": round(lo, 4), "wtd_ci95_high": round(hi, 4)})
    for c in intents:
        g = m[m.silver_intent == c]
        k = int(g.agree.sum())
        lo, hi = wilson(k, len(g))
        rows.append({"level": "class(by silver label)", "name": c, "n": len(g), "agree": k, "rate": round(k / len(g), 4),
                     "ci95_low": round(lo, 4), "ci95_high": round(hi, 4)})
    m["error_type"] = m.error_type.fillna("").str.strip()
    wrong = m[~m.agree]
    for et, g in wrong.groupby("error_type"):
        rows.append({"level": "error_type (your tag)", "name": et, "n": len(g), "agree": 0, "rate": round(len(g) / len(m), 4)})
    # which silver->true confusions the silver process makes (more informative than the vague tag)
    for (a, b), g in wrong.groupby(["silver_intent", "checked_intent"]):
        rows.append({"level": "silver_to_checked", "name": f"{a} -> {b}", "n": len(g), "agree": 0, "rate": round(len(g) / len(m), 4)})
    # how good is each labeller against the checked label (all 250 rows; unreviewed rows = silver = both models)
    for nm, col in [("silver", "silver_intent"), ("model_a gpt-oss-120b", "model_a"), ("model_b qwen3.8-27b", "model_b")]:
        k = int((m[col] == m.checked_intent).sum())
        lo, hi = wilson(k, len(m))
        rows.append({"level": "labeller_vs_checked", "name": nm, "n": len(m), "agree": k, "rate": round(k / len(m), 4),
                     "ci95_low": round(lo, 4), "ci95_high": round(hi, 4)})
    # the same restricted to the 50 hand-reviewed (hard) rows
    h = m[m.human_checked]
    for nm, col in [("silver", "silver_intent"), ("model_a gpt-oss-120b", "model_a"), ("model_b qwen3.8-27b", "model_b")]:
        k = int((h[col] == h.checked_intent).sum())
        lo, hi = wilson(k, len(h))
        rows.append({"level": "labeller_vs_checked_on_reviewed_rows", "name": nm, "n": len(h), "agree": k, "rate": round(k / len(h), 4),
                     "ci95_low": round(lo, 4), "ci95_high": round(hi, 4)})
    pd.DataFrame(rows).to_csv(RESULTS / "label_noise.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["review", "final"], required=True)
    {"review": stage_review, "final": stage_final}[ap.parse_args().stage]()
