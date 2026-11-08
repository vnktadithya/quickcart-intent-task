"""Shared model definitions for Phase 4 (used by run_models.py and latency.py).

Every model exposes: fit(train_texts, train_labels) and predict(list_of_texts) -> list of labels,
and predict_proba(list_of_texts) -> (n, 12) array in `classes_` order. Hyper-parameters are FIXED (not tuned):
tuning on the 250-row checked set would overfit the selection, and silver dev alone must not choose.
"""
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from baseline.baseline import clean

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache_embeddings"
SEED = 42

EMBED_MODELS = {   # key -> (huggingface id, text prefix the model card asks for)
    "minilm_l6": ("sentence-transformers/all-MiniLM-L6-v2", ""),
    "bge_small": ("BAAI/bge-small-en-v1.5", ""),
    "e5_small": ("intfloat/e5-small-v2", "query: "),
    "multilingual_minilm": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", ""),
}


def norm_ws(t):
    return " ".join(str(t).split())


class TfidfLR:
    name = "tfidf_word_char_lr"

    def fit(self, texts, y):
        t = [clean(x) for x in texts]
        self.w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        self.c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)
        X = hstack([self.w.fit_transform(t), self.c.fit_transform(t)]).tocsr()
        self.clf = LogisticRegression(C=10, max_iter=300, random_state=SEED).fit(X, y)
        self.classes_ = self.clf.classes_
        return self

    def _x(self, texts):
        t = [clean(x) for x in texts]
        return hstack([self.w.transform(t), self.c.transform(t)]).tocsr()

    def predict_proba(self, texts):
        return self.clf.predict_proba(self._x(texts))

    def predict(self, texts):
        return list(self.clf.predict(self._x(texts)))


class EmbedLR:
    """Frozen sentence-embedding model + logistic regression on top. Embedding vectors for train/dev are cached on disk
    (offline experiments); at inference `encode` runs the transformer on the live message."""

    def __init__(self, key):
        self.key = key
        self.name = f"{key}_lr"
        self.hf_id, self.prefix = EMBED_MODELS[key]
        self.st = None

    def load_encoder(self):
        from sentence_transformers import SentenceTransformer
        if self.st is None:
            self.st = SentenceTransformer(self.hf_id, device="cpu")
        return self.st

    def encode(self, texts, batch_size=128):
        st = self.load_encoder()
        return st.encode([self.prefix + norm_ws(t) for t in texts], batch_size=batch_size,
                         normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)

    def fit_from_vectors(self, X, y):
        self.clf = LogisticRegression(C=10, max_iter=500, random_state=SEED).fit(X, y)
        self.classes_ = self.clf.classes_
        return self

    def predict_proba_vectors(self, X):
        return self.clf.predict_proba(X)

    def predict_proba(self, texts):
        return self.clf.predict_proba(self.encode(texts))

    def predict(self, texts):
        return list(self.clf.predict(self.encode(texts)))


def load_cached(key, split):
    return np.load(CACHE / f"{key}_{split}.npy")
