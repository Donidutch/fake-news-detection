import numpy as np
import pandas as pd
import pytest

from fakenews import baseline, data
from fakenews.cli import render_report
from fakenews.config import FAKE, REAL
from fakenews.evaluate import metrics
from fakenews.preprocess import clean_text


def _corpus_csv(tmp_path, n_domains=20, per_domain=30):
    rows = []
    for d in range(n_domains):
        kind = "fake" if d % 2 else "reliable"
        word = "shocking secret" if kind == "fake" else "officials said"
        for i in range(per_domain):
            rows.append(
                {
                    "id": f"{d}-{i}",
                    "domain": f"site{d}.com",
                    "type": kind,
                    "url": "",
                    "content": f"article {d} {i} {word} " + "filler text " * 30,
                    "title": f"title {i}",
                }
            )
    rows.append({"id": "x", "domain": "satire.com", "type": "satire", "content": "a " * 200})
    rows.append({"id": "y", "domain": "short.com", "type": "fake", "content": "too short"})
    path = tmp_path / "news.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_clean_text_masks_urls_and_lowercases():
    out = clean_text("BREAKING:  See https://example.com/x?y=1 or mail a.b@c.org NOW!")
    assert out == "breaking : see urltoken or mail emailtoken now !"
    assert clean_text(None) == ""


def test_sample_drops_unmapped_short_and_caps_domains(tmp_path):
    df = data.sample_fakenewscorpus(_corpus_csv(tmp_path), per_domain=10, chunksize=50)
    assert set(df["type"]) == {"fake", "reliable"}
    assert df.groupby("domain").size().max() <= 10
    assert df["label"].value_counts()[FAKE] == df["label"].value_counts()[REAL]
    assert not df["content"].duplicated().any()


def test_domain_split_is_disjoint(tmp_path):
    df = data.sample_fakenewscorpus(_corpus_csv(tmp_path), per_domain=30)
    train, val, test = data.split(df, "domain")
    assert len(train) + len(val) + len(test) == len(df)
    domains = [set(part["domain"]) for part in (train, val, test)]
    assert not (domains[0] & domains[1] or domains[0] & domains[2] or domains[1] & domains[2])


def test_random_split_is_stratified(tmp_path):
    df = data.sample_fakenewscorpus(_corpus_csv(tmp_path), per_domain=30)
    train, val, test = data.split(df, "random")
    assert abs(test["label"].mean() - 0.5) < 0.05


def test_unknown_split_raises(tmp_path):
    with pytest.raises(ValueError):
        data.split(pd.DataFrame({"label": [0, 1]}), "nope")


def test_load_liar_maps_labels(tmp_path):
    rows = [
        ["1.json", "pants-fire", "The moon is cheese."],
        ["2.json", "true", "Water is wet."],
        ["3.json", "half-true", 'He said "maybe" twice.'],
    ]
    (tmp_path / "test.tsv").write_text("".join("\t".join(r) + "\n" for r in rows))
    df = data.load_liar(tmp_path)
    assert df["label"].tolist() == [FAKE, REAL, REAL]
    assert df["content"][2] == 'He said "maybe" twice.'


def test_baseline_learns_toy_signal(tmp_path):
    df = data.sample_fakenewscorpus(_corpus_csv(tmp_path), per_domain=30)
    train, _, test = data.split(df, "domain")
    model = baseline.build(max_features=1000)
    model.fit(train["content"].tolist(), train["label"])
    m = metrics(test["label"], baseline.predict_proba(model, test["content"]))
    assert m["accuracy"] > 0.9
    assert "shocking secret" in baseline.top_features(model, k=5)["FAKE"]


def test_metrics_single_class_has_nan_auc():
    m = metrics(np.array([1, 1]), np.array([0.9, 0.2]))
    assert m["accuracy"] == 0.5 and np.isnan(m["roc_auc"])


def test_report_renders_rows(tmp_path):
    block = {"accuracy": 0.9, "macro_f1": 0.8, "roc_auc": 0.95, "n": 10}
    (tmp_path / "baseline_domain.json").write_text(
        pd.Series(
            {"model": "m", "split": "domain", "fakenewscorpus_test": block, "liar_test": block}
        ).to_json()
    )
    table = render_report(tmp_path)
    assert "| m | domain | 0.900 | 0.800 | 0.950 |" in table
