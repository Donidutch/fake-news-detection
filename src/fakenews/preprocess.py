"""Text normalisation, ported from the 2019 project and made vectorised."""

import re

import pandas as pd

_TOKEN = re.compile(
    r"""(?x)
      (?:[A-Za-z]\.)+               # abbreviations, e.g. u.s.a.
    | \w+(?:-\w+)*                  # words with optional internal hyphens
    | \$?\d+(?:\.\d+)?%?            # currency and percentages
    | \.\.\.                        # ellipsis
    | [][.,;!"'?(){}:_`<>=*+/-]     # punctuation as separate tokens
    """
)
_URL = re.compile(r"https?://\S+")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
_WS = re.compile(r"\s+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text)


def clean_text(text: str) -> str:
    """Lowercase, mask URLs and e-mails, and re-join tokens with single spaces."""
    if not isinstance(text, str):
        return ""
    text = _URL.sub(" urltoken ", text)
    text = _EMAIL.sub(" emailtoken ", text)
    text = _WS.sub(" ", text).strip().lower()
    return " ".join(tokenize(text))


def clean_series(texts: pd.Series) -> pd.Series:
    return texts.fillna("").map(clean_text)
