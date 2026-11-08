"""Phase 4b: compare models on identical splits and metrics.

Train: data/train.csv (silver labels), the only training data for every learned model.
Eval : silver dev (4,000) and the checked dev sample (data/dev_checked.csv, 250). Nothing is tuned on either:
       all hyper-parameters are fixed in src/models.py.
Writes results/experiments.csv, results/per_class_<model>.csv, results/experiments_slices.csv.
Run: python -m src.run_models        (needs src.embed_cache first)
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from baseline.baseline import DATA
from src.evaluate import evaluate, macro_f1, prf
from src.matcher import Matcher
from src.models import EMBED_MODELS, EmbedLR, TfidfLR, load_cached

RESULTS = DATA.parent / "results"
SEED, B = 42, 1000


def boot_ci(y, preds, intents, base=None):
    """Bootstrap (resample the 250 rows, same resample for every model) macro-F1 CI; paired difference vs `base` preds."""
    rng = np.random.default_rng(SEED)
    y = np.array(y)
    n = len(y)
    idx = rng.integers(0, n, (B, n))
    f = np.array([macro_f1(y[i], np.array(preds)[i], intents) for i in idx])
    out = {"lo": np.percentile(f, 2.5), "hi": np.percentile(f, 97.5)}
    if base is not None:
        fb = np.array([macro_f1(y[i], np.array(base)[i], intents) for i in idx])
        d = f - fb
        out.update(d_lo=np.percentile(d, 2.5), d_hi=np.percentile(d, 97.5))
    return out


def main():
    train = pd.read_csv(DATA / "train.csv")
    dev = pd.read_csv(DATA / "dev.csv")
    chk = pd.read_csv(DATA / "dev_checked.csv")
    ci = chk.dev_row.to_numpy()
    intents = [i["name"] for i in __import__("json").load(open(DATA / "taxonomy_v2.json", encoding="utf-8"))["intents"]]
    preds = {}      # model -> (dev_pred, chk_pred)
    info = {}

    # --- matcher on taxonomy v2 (no training data) ---
    for nm, thr in [("matcher_v2_thr0.10", 0.10), ("matcher_v2_argmax", None)]:
        m = Matcher(DATA / "taxonomy_v2.json", threshold=thr)
        preds[nm] = (m.predict(dev.utterance.tolist())[0], m.predict(chk.utterance.tolist())[0])
        info[nm] = {"family": "taxonomy matcher", "fit_s": 0.0, "notes": "no training data; taxonomy v2 examples only"}

    # --- TF-IDF word+char + LR ---
    t0 = time.time()
    t = TfidfLR().fit(train.utterance.tolist(), train.intent.tolist())
    info["tfidf_word_char_lr"] = {"family": "sparse linear", "fit_s": time.time() - t0, "notes": "word 1-2gram + char_wb 2-5gram, C=10"}
    preds["tfidf_word_char_lr"] = (t.predict(dev.utterance.tolist()), t.predict(chk.utterance.tolist()))

    # --- frozen sentence embeddings + LR (vectors cached by embed_cache.py) ---
    ytr = train.intent.tolist()
    for key in EMBED_MODELS:
        m = EmbedLR(key)
        t0 = time.time()
        m.fit_from_vectors(load_cached(key, "train"), ytr)
        Xd = load_cached(key, "dev")
        info[m.name] = {"family": "embedding + linear", "fit_s": time.time() - t0, "notes": f"{EMBED_MODELS[key][0]}, frozen, C=10"}
        preds[m.name] = (list(m.clf.predict(Xd)), list(m.clf.predict(Xd[ci])))

    rows, slices = [], []
    base = preds["tfidf_word_char_lr"][1]
    for name, (pd_dev, pd_chk) in preds.items():
        rd = evaluate(dev.intent, pd_dev, dev.utterance, intents)
        rc = evaluate(chk.checked_intent, pd_chk, chk.utterance, intents)
        c = boot_ci(chk.checked_intent.tolist(), pd_chk, intents, base)
        rows.append({"model": name, "family": info[name]["family"], "train_rows": 0 if "matcher" in name else len(train),
                     "silver_dev_accuracy": round(rd["overall"]["accuracy"], 4), "silver_dev_macro_f1": round(rd["overall"]["macro_f1"], 4),
                     "checked_accuracy": round(rc["overall"]["accuracy"], 4), "checked_macro_f1": round(rc["overall"]["macro_f1"], 4),
                     "checked_macro_f1_ci95_low": round(c["lo"], 4), "checked_macro_f1_ci95_high": round(c["hi"], 4),
                     "checked_delta_vs_tfidf": round(rc["overall"]["macro_f1"] - macro_f1(chk.checked_intent, base, intents), 4),
                     "delta_ci95_low": round(c["d_lo"], 4), "delta_ci95_high": round(c["d_hi"], 4),
                     "fit_seconds": round(info[name]["fit_s"], 1), "notes": info[name]["notes"]})
        pcd = prf(dev.intent.tolist(), pd_dev, intents).set_index("intent")
        pcc = prf(chk.checked_intent.tolist(), pd_chk, intents).set_index("intent")
        pc = pd.DataFrame({"silver_dev_precision": pcd.precision, "silver_dev_recall": pcd.recall, "silver_dev_f1": pcd.f1,
                           "silver_dev_support": pcd.support.astype(int), "checked_precision": pcc.precision,
                           "checked_recall": pcc.recall, "checked_f1": pcc.f1, "checked_support": pcc.support.astype(int)}).round(4)
        pc.index.name = "intent"
        pc.to_csv(RESULTS / f"per_class_{name}.csv")
        for _, r in rd["slices"].iterrows():
            slices.append({"model": name, "slice": r.slice, "value": r.value, "n": int(r.n), "accuracy": round(r.accuracy, 4),
                           "macro_f1": round(r.macro_f1, 4)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "experiments.csv", index=False)
    pd.DataFrame(slices).to_csv(RESULTS / "experiments_slices.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.drop(columns=["family", "notes", "train_rows"]).to_string(index=False))


if __name__ == "__main__":
    main()
