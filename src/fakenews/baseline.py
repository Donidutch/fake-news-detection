"""TF-IDF + logistic regression, the strongest classical model from the 2019 project."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fakenews.preprocess import clean_text


def build(max_features: int = 200_000) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.9,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            ("clf", LogisticRegression(C=4.0, max_iter=2000)),
        ]
    )


def predict_proba(model: Pipeline, texts) -> np.ndarray:
    return model.predict_proba(list(texts))[:, 1]


def top_features(model: Pipeline, k: int = 20) -> dict[str, list[str]]:
    """The n-grams that push hardest towards each class."""
    names = model.named_steps["tfidf"].get_feature_names_out()
    coef = model.named_steps["clf"].coef_[0]
    order = np.argsort(coef)
    return {"FAKE": names[order[::-1][:k]].tolist(), "REAL": names[order[:k]].tolist()}


def save(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load(path: Path) -> Pipeline:
    return joblib.load(path)
