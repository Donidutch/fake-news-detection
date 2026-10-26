"""Metrics shared by every model."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def metrics(y_true: np.ndarray, p_fake: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Accuracy, macro F1 and ROC AUC from the predicted probability of FAKE."""
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(p_fake) >= threshold).astype(int)
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    out["roc_auc"] = (
        float(roc_auc_score(y_true, p_fake)) if len(np.unique(y_true)) == 2 else float("nan")
    )
    return out
