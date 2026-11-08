"""Phase 4a: embed train and dev once per sentence-embedding model and cache the vectors (offline step).

test.csv is NOT embedded here (test is used once, in Phase 7).
Also writes results/embedding_models.csv: Hugging Face id, licence (from the model card metadata), dimension, parameters.

Run: python -m src.embed_cache        (uses all CPU threads; this is experiment-time compute, not the 4-core inference budget)
"""
import time

import numpy as np
import pandas as pd

from baseline.baseline import DATA
from src.models import CACHE, EMBED_MODELS, EmbedLR, ROOT


def main():
    CACHE.mkdir(exist_ok=True)
    train = pd.read_csv(DATA / "train.csv").utterance.tolist()
    dev = pd.read_csv(DATA / "dev.csv").utterance.tolist()
    rows = []
    for key, (hf_id, _) in EMBED_MODELS.items():
        m = EmbedLR(key)
        lic = ""
        try:
            from huggingface_hub import model_info
            lic = model_info(hf_id).card_data.license
        except Exception as e:  # network / metadata problem: leave blank, never guess
            lic = f"unknown ({type(e).__name__})"
        st = m.load_encoder()
        n_params = sum(p.numel() for p in st.parameters())
        t0 = time.time()
        for split, texts in [("dev", dev), ("train", train)]:
            f = CACHE / f"{key}_{split}.npy"
            if f.exists():
                continue
            np.save(f, m.encode(texts).astype(np.float32))
            print(f"{key} {split}: {len(texts)} rows, {time.time()-t0:.0f}s elapsed", flush=True)
        rows.append({"model": key, "hf_id": hf_id, "license": lic, "embedding_dim": st.get_sentence_embedding_dimension(),
                     "parameters_M": round(n_params / 1e6, 1), "multilingual": "multilingual" in hf_id})
        pd.DataFrame(rows).to_csv(ROOT / "results" / "embedding_models.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
