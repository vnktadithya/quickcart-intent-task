"""Phase 4c: per-message latency at 4 threads (CLAUDE.md protocol).

OMP_NUM_THREADS=4, MKL_NUM_THREADS=4, torch.set_num_threads(4); ONE message per call; 20 warm-up calls;
then 500 timed calls on dev messages (seed 42). Timed span = the full predict() for a single message
(cleaning + vectorising / embedding + classifier). Model loading and training are NOT in the span (done once at start-up).
Writes/updates a row per model in results/latency.csv: model, p50_ms, p95_ms, hardware, threads, notes.

Run each model in a fresh process so thread settings apply:  python -m src.latency --model tfidf_word_char_lr
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "4"          # must be set before numpy / torch are imported

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from baseline.baseline import DATA
from src.models import EMBED_MODELS, EmbedLR, TfidfLR, load_cached

RESULTS = Path(__file__).resolve().parent.parent / "results"
SEED, WARM, TIMED = 42, 20, 500


def hardware():
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "$p=Get-CimInstance Win32_Processor|Select-Object -First 1; "
                              "\"$($p.Name.Trim()) ($($p.NumberOfCores) cores/$($p.NumberOfLogicalProcessors) threads)\""],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def build(model):
    train = pd.read_csv(DATA / "train.csv")
    if model == "tfidf_word_char_lr":
        return TfidfLR().fit(train.utterance.tolist(), train.intent.tolist()), "word+char TF-IDF"
    if model.startswith("matcher_v2"):
        from src.matcher import Matcher
        m = Matcher(DATA / "taxonomy_v2.json", threshold=0.10)
        m.predict = (lambda f: (lambda texts: f(texts)[0]))(m.predict)
        return m, "taxonomy v2 matcher"
    key = model[:-3]                       # strip "_lr"
    m = EmbedLR(key)
    m.fit_from_vectors(load_cached(key, "train"), train.intent.tolist())
    m.load_encoder()
    return m, EMBED_MODELS[key][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(4)
    import psutil
    model, desc = build(a.model)
    rss_mb = psutil.Process().memory_info().rss / 2**20
    msgs = pd.read_csv(DATA / "dev.csv").utterance.sample(WARM + TIMED, random_state=SEED).tolist()
    for t in msgs[:WARM]:
        model.predict([t])
    lat = []
    for t in msgs[WARM:]:
        t0 = time.perf_counter()
        model.predict([t])
        lat.append((time.perf_counter() - t0) * 1000)
    row = {"model": a.model, "p50_ms": round(float(np.percentile(lat, 50)), 2), "p95_ms": round(float(np.percentile(lat, 95)), 2),
           "hardware": hardware() + ", Windows, no GPU", "threads": 4,
           "notes": f"{desc}; 1 msg/call, {WARM} warm-up + {TIMED} timed; process RSS {rss_mb:.0f} MB after load; "
                    f"laptop with 4 threads allowed, not a dedicated 4-core box"}
    f = RESULTS / "latency.csv"
    df = pd.read_csv(f) if f.exists() else pd.DataFrame()
    df = pd.concat([df[df.model != a.model] if len(df) else df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(f, index=False)
    print(row)


if __name__ == "__main__":
    main()
