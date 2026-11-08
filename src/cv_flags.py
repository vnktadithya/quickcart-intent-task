"""Phase 2a: 5-fold cross-validated label disagreement on train.

Train word+char TF-IDF + logistic regression on 4 folds, predict the held-out fold, repeat.
Every train row therefore gets a prediction from a model that never saw that row.
A row is FLAGGED when that out-of-fold prediction differs from the silver label.
A flag means "label OR model is wrong"; it is a review queue, not a noise rate.

Run: python -m src.cv_flags      (seed 42; writes results/cv_flags_train.csv, results/cv_disagreement_by_class.csv)
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from baseline.baseline import DATA, clean, load_taxonomy

RESULTS = Path(__file__).resolve().parent.parent / "results"
SEED = 42


def fit_predict_proba(tr_text, tr_y, te_text):
    w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)
    Xtr = hstack([w.fit_transform(tr_text), c.fit_transform(tr_text)]).tocsr()
    Xte = hstack([w.transform(te_text), c.transform(te_text)]).tocsr()
    clf = LogisticRegression(C=10, max_iter=300, random_state=SEED).fit(Xtr, tr_y)
    return clf.classes_, clf.predict_proba(Xte)


def main():
    intents, _ = load_taxonomy()
    train = pd.read_csv(DATA / "train.csv")
    text = np.array([clean(t) for t in train.utterance])
    y = train.intent.to_numpy()
    oof = np.zeros((len(train), len(intents)))
    t0 = time.time()
    for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(text, y)):
        classes, p = fit_predict_proba(text[a], y[a], text[b])
        order = [list(classes).index(i) for i in intents]
        oof[b] = p[:, order]
        print(f"fold {k}: {time.time()-t0:.0f}s", flush=True)
    pred = np.array(intents)[oof.argmax(1)]
    p_silver = oof[np.arange(len(train)), [intents.index(v) for v in y]]
    out = pd.DataFrame({"train_row": np.arange(len(train)), "utterance": train.utterance, "silver": y,
                        "oof_pred": pred, "p_silver": p_silver.round(4), "p_pred": oof.max(1).round(4)})
    out["flagged"] = out.silver != out.oof_pred
    out[out.flagged].sort_values("p_silver").to_csv(RESULTS / "cv_flags_train.csv", index=False)

    rows = []
    for i in intents:
        m = out.silver == i
        pr = out[m & out.flagged].oof_pred.value_counts()
        rows.append({"silver_class": i, "n_train": int(m.sum()), "n_flagged": int((m & out.flagged).sum()),
                     "flag_rate": round(float((m & out.flagged).sum() / m.sum()), 4),
                     "top_disagreement_1": f"{pr.index[0]} ({pr.iloc[0]})" if len(pr) else "",
                     "top_disagreement_2": f"{pr.index[1]} ({pr.iloc[1]})" if len(pr) > 1 else ""})
    rows.append({"silver_class": "ALL", "n_train": len(out), "n_flagged": int(out.flagged.sum()),
                 "flag_rate": round(float(out.flagged.mean()), 4)})
    pd.DataFrame(rows).to_csv(RESULTS / "cv_disagreement_by_class.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
