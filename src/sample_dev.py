"""Phase 2b: draw the 250-row stratified dev sample that the LLM will pre-label.

Allocation: proportional to class share, but at least MIN_PER_CLASS per class so that the four small classes
(promo_code_issue, wrong_item, account_help, change_address; proportional share would be 13-14 rows) have enough
rows to say anything per class. Extra rows come out of the larger classes proportionally.
Each row carries a weight = (dev class size) / (sampled class size) so overall agreement can be re-weighted to dev.

Run: python -m src.sample_dev     (seed 42; writes data/dev_sample.csv)
"""
import numpy as np
import pandas as pd

from baseline.baseline import DATA, load_taxonomy

SEED, N, MIN_PER_CLASS = 42, 250, 15


def allocate(counts, n=N, floor=MIN_PER_CLASS):
    share = counts / counts.sum() * n
    alloc = np.maximum(np.floor(share), floor).astype(int)
    # fix the total: add/remove one row at a time, largest-remainder style, never going below the floor
    while alloc.sum() != n:
        d = alloc.sum() - n
        resid = (share - alloc)
        if d < 0:
            alloc[resid.argmax()] += 1
        else:
            cand = np.where(alloc > floor, resid, np.inf)
            alloc[cand.argmin()] -= 1
    return alloc


def main():
    intents, _ = load_taxonomy()
    dev = pd.read_csv(DATA / "dev.csv")
    dev["dev_row"] = np.arange(len(dev))
    counts = dev.intent.value_counts().reindex(intents)
    alloc = pd.Series(allocate(counts.to_numpy(float)), index=intents)
    parts = [dev[dev.intent == i].sample(int(alloc[i]), random_state=SEED) for i in intents]
    s = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    s["weight"] = s.intent.map(counts / alloc).round(4)
    s = s.rename(columns={"intent": "silver_intent"})[["dev_row", "utterance", "silver_intent", "weight"]]
    s.to_csv(DATA / "dev_sample.csv", index=False)
    print(pd.DataFrame({"dev_n": counts, "proportional": (counts / counts.sum() * N).round(1), "sampled": alloc}).to_string())
    print("total", len(s))


if __name__ == "__main__":
    main()
