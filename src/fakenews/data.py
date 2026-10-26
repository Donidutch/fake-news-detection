"""Loading FakeNewsCorpus and LIAR, and building leakage-aware splits.

FakeNewsCorpus is ~27 GB of CSV, so it is streamed in chunks. Its labels belong to the
publishing domain, which makes a random article-level split leak: the model can learn
each site's boilerplate instead of anything about the content. The ``domain`` split keeps
every domain on one side of the train/test boundary to measure that gap.
"""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from fakenews.config import FNC_LABELS, LIAR_LABELS

SPLITS = ("random", "domain")
MAX_CHARS = 5000


def _fnc_chunks(path: Path, chunksize: int) -> Iterator[pd.DataFrame]:
    yield from pd.read_csv(
        path,
        usecols=["domain", "type", "title", "content"],
        dtype=str,
        chunksize=chunksize,
        on_bad_lines="skip",
        engine="c",
    )


def sample_fakenewscorpus(
    path: Path,
    per_domain: int = 1000,
    per_class: int = 100_000,
    min_chars: int = 200,
    chunksize: int = 200_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Stream the corpus and return a deduplicated, domain-capped, class-balanced sample.

    Each domain keeps at most ``per_domain`` articles, chosen by the smallest content hash
    (a bottom-k sketch), so the sample is uniform within a domain without loading the file.
    The cap stops a handful of very large sites from dominating the data.
    """
    kept: pd.DataFrame | None = None
    for chunk in _fnc_chunks(path, chunksize):
        chunk = chunk[chunk["type"].isin(FNC_LABELS.keys())].dropna(subset=["content", "domain"])
        chunk = chunk[chunk["content"].str.len() >= min_chars]
        if chunk.empty:
            continue
        chunk = chunk.assign(content=chunk["content"].str.slice(0, MAX_CHARS))
        chunk["h"] = pd.util.hash_pandas_object(chunk["content"], index=False).to_numpy()
        merged = chunk if kept is None else pd.concat([kept, chunk], ignore_index=True)
        merged = merged.drop_duplicates("h")
        kept = merged.sort_values("h").groupby("domain", sort=False).head(per_domain)

    if kept is None:
        raise ValueError(f"no usable rows in {path}")
    kept = kept.assign(label=kept["type"].map(FNC_LABELS).astype(int)).drop(columns="h")
    n = min(per_class, int(kept["label"].value_counts().min()))
    balanced = pd.concat(g.sample(n, random_state=seed) for _, g in kept.groupby("label"))
    return balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_liar(liar_dir: Path, part: str = "test") -> pd.DataFrame:
    """Load one LIAR split (train/valid/test) as ``content`` and binary ``label``."""
    df = pd.read_csv(
        liar_dir / f"{part}.tsv",
        sep="\t",
        header=None,
        usecols=[1, 2],
        names=["raw_label", "content"],
        quoting=3,
        dtype=str,
    )
    df = df[df["raw_label"].isin(LIAR_LABELS.keys())]
    return df.assign(label=df["raw_label"].map(LIAR_LABELS).astype(int)).reset_index(drop=True)


def split(
    df: pd.DataFrame, kind: str, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train/val/test (80/10/10) as a random or domain-disjoint split."""
    if kind == "random":
        train, rest = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=seed)
        val, test = train_test_split(rest, test_size=0.5, stratify=rest["label"], random_state=seed)
        return train, val, test
    if kind == "domain":
        folds = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
        fold_of = np.empty(len(df), dtype=int)
        for i, (_, idx) in enumerate(folds.split(df, df["label"], groups=df["domain"])):
            fold_of[idx] = i
        return df[fold_of >= 2], df[fold_of == 1], df[fold_of == 0]
    raise ValueError(f"unknown split {kind!r}, expected one of {SPLITS}")
