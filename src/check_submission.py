"""Phase 7b: format check for a predictions file (submissions/README.md rules).

Checks: exactly the columns id,intent; one row per id in data/test.csv, same ids, same order, no duplicates, no blanks;
every intent is one of the 12 names in data/taxonomy.json. Also prints the predicted class distribution (descriptive only).
Exit code 1 if any check fails.

Run: python -m src.check_submission submissions/adithya_predictions.csv
"""
import json
import sys
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"


def check(path):
    problems = []
    sub = pd.read_csv(path, dtype=str, keep_default_na=False)
    test = pd.read_csv(DATA / "test.csv", dtype=str)
    names = [i["name"] for i in json.load(open(DATA / "taxonomy.json", encoding="utf-8"))["intents"]]
    if list(sub.columns) != ["id", "intent"]:
        problems.append(f"columns are {list(sub.columns)}, expected ['id', 'intent']")
        return problems, sub
    if len(sub) != len(test):
        problems.append(f"{len(sub)} rows, expected {len(test)}")
    if sub.id.duplicated().any():
        problems.append(f"{int(sub.id.duplicated().sum())} duplicate ids")
    if set(sub.id) != set(test.id):
        problems.append(f"id sets differ: {len(set(sub.id) - set(test.id))} unknown, {len(set(test.id) - set(sub.id))} missing")
    elif list(sub.id) != list(test.id):
        problems.append("ids are the same set but not in test.csv order (allowed by the README, flagged for information)")
    bad = sorted(set(sub.intent) - set(names))
    if bad:
        problems.append(f"invalid intent names: {bad}")
    if (sub.intent.str.strip() == "").any():
        problems.append("blank intents")
    return problems, sub


if __name__ == "__main__":
    p = sys.argv[1]
    problems, sub = check(p)
    print(f"{p}: {len(sub)} rows")
    if "intent" in sub.columns:
        print(sub.intent.value_counts().to_string())
    hard = [x for x in problems if "not in test.csv order" not in x]
    for x in problems:
        print("PROBLEM:", x)
    print("FORMAT OK" if not hard else "FORMAT FAILED")
    sys.exit(1 if hard else 0)
