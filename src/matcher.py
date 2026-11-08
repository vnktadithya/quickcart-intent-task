"""Taxonomy matcher with the same method as baseline/baseline.py, but parameterised.

Differences from the starter (starter file is untouched):
  * taxonomy file is an argument (v1 or v2)
  * vectoriser is fitted on train + taxonomy examples only (no dev text; eval_bugs.md #8), unless fit_on_dev=True
  * threshold is an argument; None = pure argmax (no out_of_scope-by-threshold)
Method is unchanged: TF-IDF (1-2 grams, min_df=2, sublinear_tf), one centroid per intent, cosine, best intent wins.
"""
import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from baseline.baseline import DATA, clean

OOS = "out_of_scope"


class Matcher:
    def __init__(self, taxonomy_path, threshold=0.10, fit_on_dev=False):
        tax = taxonomy_path if isinstance(taxonomy_path, dict) else json.load(open(taxonomy_path, encoding="utf-8"))
        self.intents = [i["name"] for i in tax["intents"]]
        self.examples = {i["name"]: i["examples"] for i in tax["intents"]}
        self.threshold = threshold
        texts = [pd.read_csv(DATA / "train.csv").utterance]
        if fit_on_dev:
            texts.append(pd.read_csv(DATA / "dev.csv").utterance)
        corpus = [clean(t) for t in pd.concat(texts)] + [clean(e) for ex in self.examples.values() for e in ex]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(corpus)
        rows = []
        for name in self.intents:
            ex = self.examples[name]
            if not ex:
                rows.append(np.zeros(len(self.vec.vocabulary_)))
                continue
            rows.append(np.asarray(self.vec.transform([clean(e) for e in ex]).mean(axis=0)).ravel())
        self.centroids = normalize(np.vstack(rows))

    def scores(self, utterances):
        return np.asarray(normalize(self.vec.transform([clean(u) for u in utterances])) @ self.centroids.T)

    def predict(self, utterances):
        s = self.scores(utterances)
        pred = [self.intents[i] for i in s.argmax(1)]
        if self.threshold is not None:
            pred = [OOS if b < self.threshold else p for p, b in zip(pred, s.max(1))]
        return pred, s

    def centroid_similarity(self):
        """Intent-to-intent cosine similarity of the taxonomy centroids (diagonal = 1, empty intents = 0)."""
        return pd.DataFrame(self.centroids @ self.centroids.T, index=self.intents, columns=self.intents)
