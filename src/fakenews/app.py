"""Gradio demo: paste an article, compare the baseline and the transformer.

Run with ``uv run --extra demo python -m fakenews.app`` after training the models.
"""

import gradio as gr
import numpy as np

from fakenews import baseline, config

BASELINE_PATH = config.MODELS_DIR / "baseline_domain.joblib"
TRANSFORMER_DIR = config.MODELS_DIR / "transformer_domain"

_baseline = baseline.load(BASELINE_PATH) if BASELINE_PATH.exists() else None
_transformer = None
if TRANSFORMER_DIR.exists():
    from fakenews import transformer

    _transformer = transformer.load(TRANSFORMER_DIR)


def _explain(text: str, k: int = 10) -> list[tuple[str, float]]:
    """Per-term contribution (tf-idf weight x coefficient) for the baseline."""
    tfidf = _baseline.named_steps["tfidf"]
    coef = _baseline.named_steps["clf"].coef_[0]
    row = tfidf.transform([text])
    contrib = row.multiply(coef).tocsr()
    idx, vals = contrib.indices, contrib.data
    order = np.argsort(-np.abs(vals))[:k]
    names = tfidf.get_feature_names_out()
    return [(names[idx[i]], round(float(vals[i]), 3)) for i in order]


def classify(text: str):
    scores, terms = {}, []
    if _baseline is not None:
        scores["TF-IDF + LogReg"] = float(baseline.predict_proba(_baseline, [text])[0])
        terms = [[t, w, "FAKE" if w > 0 else "REAL"] for t, w in _explain(text)]
    if _transformer is not None:
        model, tok = _transformer
        scores["RoBERTa"] = float(transformer.predict_proba(model, tok, [text])[0])
    table = [[name, f"{p:.1%}", "FAKE" if p >= 0.5 else "REAL"] for name, p in scores.items()]
    return table, terms


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(lines=12, label="Article text"),
    outputs=[
        gr.Dataframe(headers=["Model", "P(fake)", "Prediction"], label="Predictions"),
        gr.Dataframe(headers=["Term", "Contribution", "Pushes towards"], label="Baseline terms"),
    ],
    title="Fake news detection",
    description=(
        "Models trained on FakeNewsCorpus with a domain-disjoint split. Labels come from the "
        "publishing site's reputation, so the score says how much the text resembles "
        "unreliable outlets, not whether a specific claim is true."
    ),
)

if __name__ == "__main__":
    demo.launch()
