"""Paths and label mappings shared across the pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

FNC_CSV = RAW_DIR / "fakenewscorpus" / "news_cleaned_2018_02_13.csv"
LIAR_DIR = RAW_DIR / "liar"
CORPUS_PARQUET = PROCESSED_DIR / "corpus.parquet"

FAKE, REAL = 1, 0
LABEL_NAMES = {REAL: "REAL", FAKE: "FAKE"}

# FakeNewsCorpus labels come from OpenSources and describe the *domain*, not the article.
# Only the unambiguous classes are kept. The 2019 version of this project also counted
# bias, clickbait, rumor and unknown as REAL, which mixes noise into the positive class.
FNC_LABELS = {
    "fake": FAKE,
    "conspiracy": FAKE,
    "junksci": FAKE,
    "hate": FAKE,
    "unreliable": FAKE,
    "reliable": REAL,
    "political": REAL,
}

# LIAR has six truthfulness levels for short political statements.
LIAR_LABELS = {
    "pants-fire": FAKE,
    "false": FAKE,
    "barely-true": FAKE,
    "half-true": REAL,
    "mostly-true": REAL,
    "true": REAL,
}
