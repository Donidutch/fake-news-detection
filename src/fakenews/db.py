"""Storing the corpus in PostgreSQL and querying it with SQL.

The schema is normalised: a domain and an article type are stored once and referenced
by id, so the type -> label mapping lives in one row instead of on every article.
Articles are deduplicated on a SHA-256 hash of their content, so loading the same data
twice adds nothing. Bulk loads go through COPY into a staging table, which is far faster
than row-by-row INSERTs, and run in one transaction: a failed load leaves no partial data.
"""

import hashlib
import os

import pandas as pd
import psycopg

from fakenews.config import FNC_LABELS, LABEL_NAMES

DSN_ENV = "FAKENEWS_DSN"

SCHEMA = """
CREATE TABLE IF NOT EXISTS domain (
    id   serial PRIMARY KEY,
    name text NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS article_type (
    id    smallserial PRIMARY KEY,
    name  text NOT NULL UNIQUE,
    label smallint NOT NULL CHECK (label IN (0, 1))
);
CREATE TABLE IF NOT EXISTS article (
    id           bigserial PRIMARY KEY,
    domain_id    integer  NOT NULL REFERENCES domain (id),
    type_id      smallint NOT NULL REFERENCES article_type (id),
    title        text,
    content      text NOT NULL,
    content_hash bytea NOT NULL UNIQUE
);
CREATE INDEX IF NOT EXISTS article_domain_idx ON article (domain_id);
CREATE INDEX IF NOT EXISTS article_type_idx ON article (type_id);
"""


def dsn_from_env(dsn: str | None = None) -> str:
    dsn = dsn or os.environ.get(DSN_ENV)
    if not dsn:
        raise SystemExit(f"no database: pass --dsn or set {DSN_ENV}")
    return dsn


def create_schema(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)


def _hash(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8")).digest()


def load(conn: psycopg.Connection, df: pd.DataFrame) -> int:
    """Insert articles (columns ``domain``, ``type``, ``title``, ``content``).

    Returns the number of new articles; duplicates of stored content are skipped.
    """
    unknown = set(df["type"]) - FNC_LABELS.keys()
    if unknown:
        raise ValueError(f"article types without a label: {sorted(unknown)}")

    with conn.transaction():
        create_schema(conn)
        conn.cursor().executemany(
            "INSERT INTO article_type (name, label) VALUES (%s, %s) ON CONFLICT (name) DO NOTHING",
            list(FNC_LABELS.items()),
        )
        conn.execute(
            "INSERT INTO domain (name) SELECT unnest(%s::text[]) ON CONFLICT (name) DO NOTHING",
            (sorted(set(df["domain"])),),
        )
        conn.execute(
            "CREATE TEMP TABLE staging (pos bigint, domain text, type text, title text, "
            "content text, content_hash bytea) ON COMMIT DROP"
        )
        with conn.cursor().copy(
            "COPY staging (pos, domain, type, title, content, content_hash) FROM STDIN"
        ) as copy:
            rows = df[["domain", "type", "title", "content"]].itertuples(index=False)
            for pos, row in enumerate(rows):
                title = None if pd.isna(row.title) else row.title
                copy.write_row((pos, row.domain, row.type, title, row.content, _hash(row.content)))
        inserted = conn.execute(
            """
            INSERT INTO article (domain_id, type_id, title, content, content_hash)
            SELECT d.id, t.id, s.title, s.content, s.content_hash
            FROM (
                -- the first occurrence of each content, so ids follow the input order
                SELECT DISTINCT ON (content_hash) *
                FROM staging ORDER BY content_hash, pos
            ) s
            JOIN domain d ON d.name = s.domain
            JOIN article_type t ON t.name = s.type
            ORDER BY s.pos
            ON CONFLICT (content_hash) DO NOTHING
            """
        ).rowcount
    return inserted


def read_corpus(conn: psycopg.Connection) -> pd.DataFrame:
    """The stored corpus in the shape the training code expects, in insertion order."""
    rows = conn.execute(
        """
        SELECT d.name, t.name, a.title, a.content, t.label
        FROM article a
        JOIN domain d ON d.id = a.domain_id
        JOIN article_type t ON t.id = a.type_id
        ORDER BY a.id
        """
    ).fetchall()
    df = pd.DataFrame(rows, columns=["domain", "type", "title", "content", "label"])
    return df.astype({"label": int})


def stats(conn: psycopg.Connection, top: int = 10) -> dict:
    """Summary statistics, each computed by a SQL aggregate."""
    rows, domains, median_chars = conn.execute(
        """
        SELECT count(*), count(DISTINCT domain_id),
               percentile_cont(0.5) WITHIN GROUP (ORDER BY length(content))
        FROM article
        """
    ).fetchone()
    by_type = conn.execute(
        """
        SELECT t.name, count(*) FROM article a JOIN article_type t ON t.id = a.type_id
        GROUP BY t.name ORDER BY count(*) DESC, t.name
        """
    ).fetchall()
    by_label = conn.execute(
        """
        SELECT t.label, count(*) FROM article a JOIN article_type t ON t.id = a.type_id
        GROUP BY t.label ORDER BY t.label
        """
    ).fetchall()
    top_domains = conn.execute(
        """
        SELECT d.name, count(*) FROM article a JOIN domain d ON d.id = a.domain_id
        GROUP BY d.name ORDER BY count(*) DESC, d.name LIMIT %s
        """,
        (top,),
    ).fetchall()
    return {
        "rows": rows,
        "domains": domains,
        "median_chars": float(median_chars) if median_chars is not None else None,
        "by_type": dict(by_type),
        "by_label": {LABEL_NAMES[label]: n for label, n in by_label},
        "top_domains": dict(top_domains),
    }
