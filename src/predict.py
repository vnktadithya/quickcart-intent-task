"""Phase 7a: write test predictions with the final model.

Test is read ONLY here, ONLY to predict (no labels exist, none are inferred, nothing is tuned on it).
Before predicting it re-checks the saved model on held-out dev (silver + checked) so the submitted model's reported numbers
are those of this exact file.

Run: python -m src.predict --name adithya          -> submissions/adithya_predictions.csv
"""
import argparse

import pandas as pd

from baseline.baseline import DATA
from src.evaluate import evaluate
from src.final_model import MODEL_PATH, load, train_and_save

ROOT = DATA.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="file becomes submissions/<name>_predictions.csv")
    a = ap.parse_args()
    m = load() if MODEL_PATH.exists() else train_and_save()
    intents = [i["name"] for i in __import__("json").load(open(DATA / "taxonomy.json", encoding="utf-8"))["intents"]]

    dev, chk = pd.read_csv(DATA / "dev.csv"), pd.read_csv(DATA / "dev_checked.csv")
    rd = evaluate(dev.intent, m.predict(dev.utterance.tolist()), dev.utterance, intents)
    rc = evaluate(chk.checked_intent, m.predict(chk.utterance.tolist()), chk.utterance, intents)
    print(f"held-out check of the saved model: silver dev acc {rd['overall']['accuracy']:.4f} macro-F1 {rd['overall']['macro_f1']:.4f} | "
          f"checked acc {rc['overall']['accuracy']:.4f} macro-F1 {rc['overall']['macro_f1']:.4f}")

    test = pd.read_csv(DATA / "test.csv")
    pred = m.predict(test.utterance.tolist())
    out = ROOT / "submissions" / f"{a.name}_predictions.csv"
    pd.DataFrame({"id": test["id"], "intent": pred}).to_csv(out, index=False)
    print("wrote", out, len(pred), "rows")


if __name__ == "__main__":
    main()
