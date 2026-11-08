"""The final production model: word+char TF-IDF + logistic regression, 12 classes (out_of_scope is a trained class).

Choices and the evidence for them: DECISIONS.md D11 (model family), D12 (fixed hyper-parameters), D14 (out_of_scope as a class).
Trained on data/train.csv ONLY (silver labels). Dev and checked sets stay held out so the reported numbers describe this exact model.
Saved to models/final_tfidf_lr.joblib (git-ignored; re-create with `python -m src.final_model`).
"""
import time
from pathlib import Path

import joblib
import pandas as pd

from baseline.baseline import DATA
from src.models import TfidfLR

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "final_tfidf_lr.joblib"


def train_and_save():
    train = pd.read_csv(DATA / "train.csv")
    t0 = time.time()
    m = TfidfLR().fit(train.utterance.tolist(), train.intent.tolist())
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(m, MODEL_PATH, compress=3)
    print(f"trained on {len(train)} rows in {time.time()-t0:.0f}s; saved {MODEL_PATH} ({MODEL_PATH.stat().st_size/2**20:.1f} MB)")
    return m


def load():
    return joblib.load(MODEL_PATH)


if __name__ == "__main__":
    train_and_save()
