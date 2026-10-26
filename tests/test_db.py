"""PostgreSQL tests. They run against ``$FAKENEWS_TEST_DSN`` (the CI service container)
or, locally, an embedded server from ``pgserver``; with neither they are skipped."""

import os

import pandas as pd
import pytest

psycopg = pytest.importorskip("psycopg")

from fakenews import db  # noqa: E402
from fakenews.cli import main  # noqa: E402


@pytest.fixture(scope="session")
def dsn(tmp_path_factory):
    if url := os.environ.get("FAKENEWS_TEST_DSN"):
        return url
    pgserver = pytest.importorskip("pgserver")
    server = pgserver.get_server(tmp_path_factory.mktemp("pg"), cleanup_mode="stop")
    return server.get_uri()


@pytest.fixture
def conn(dsn):
    with psycopg.connect(dsn) as c:
        c.execute("DROP TABLE IF EXISTS article, domain, article_type CASCADE")
        c.commit()
        yield c


def _articles():
    return pd.DataFrame(
        {
            "domain": ["a.com", "a.com", "b.com", "c.com", "c.com"],
            "type": ["fake", "fake", "reliable", "political", "fake"],
            "title": ["t1", None, "t3", "t4", "t5"],
            "content": ["one two three", "four five", "six", "seven eight", "nine"],
        }
    )


def test_load_and_read_round_trip(conn):
    df = _articles()
    assert db.load(conn, df) == 5
    out = db.read_corpus(conn)
    pd.testing.assert_frame_equal(
        out[["domain", "type", "title", "content"]], df, check_dtype=False
    )
    assert out["label"].tolist() == [1, 1, 0, 0, 1]


def test_loading_twice_adds_nothing_and_duplicates_are_skipped(conn):
    df = _articles()
    db.load(conn, df)
    assert db.load(conn, df) == 0
    dup = pd.concat([df.iloc[[0]], df.iloc[[0]].assign(domain="new.com")])
    assert db.load(conn, dup) == 0  # same content, stored once
    assert db.stats(conn)["rows"] == 5


def test_schema_is_normalised(conn):
    db.load(conn, _articles())
    assert conn.execute("SELECT count(*) FROM domain").fetchone()[0] == 3
    labels = dict(conn.execute("SELECT name, label FROM article_type").fetchall())
    assert labels["fake"] == 1 and labels["reliable"] == 0
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        conn.execute(
            "INSERT INTO article (domain_id, type_id, content, content_hash) "
            "VALUES (999, 1, 'x', 'x')"
        )
    conn.rollback()


def test_unknown_type_is_rejected_before_writing(conn):
    bad = _articles().assign(type=["fake", "satire", "fake", "fake", "fake"])
    with pytest.raises(ValueError, match="satire"):
        db.load(conn, bad)
    exists = conn.execute("SELECT to_regclass('article')").fetchone()[0]
    assert exists is None  # nothing was created or written


def test_stats_match_pandas(conn):
    df = _articles()
    db.load(conn, df)
    s = db.stats(conn, top=2)
    assert s["rows"] == len(df)
    assert s["domains"] == df["domain"].nunique()
    assert s["by_type"] == df["type"].value_counts().to_dict()
    assert s["by_label"] == {"FAKE": 3, "REAL": 2}
    assert s["top_domains"] == {"a.com": 2, "c.com": 2}
    assert s["median_chars"] == df["content"].str.len().median()


def test_cli_load_db_and_db_stats(conn, dsn, tmp_path, monkeypatch, capsys):
    from fakenews import config

    parquet = tmp_path / "corpus.parquet"
    _articles().to_parquet(parquet, index=False)
    monkeypatch.setattr(config, "CORPUS_PARQUET", parquet)
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setenv(db.DSN_ENV, dsn)

    main(["load-db"])
    assert "inserted 5 of 5 articles" in capsys.readouterr().out
    assert (tmp_path / "results" / "db_stats.json").exists()
    main(["db-stats"])
    assert '"rows": 5' in capsys.readouterr().out
