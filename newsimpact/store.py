"""SQLite storage with full-text search (FTS5 when available, LIKE fallback otherwise)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id TEXT PRIMARY KEY, title TEXT, url TEXT, source TEXT, published_utc TEXT, published_ist TEXT,
    time_precise INTEGER, trading_day TEXT, query_symbol TEXT, sentiment TEXT, score REAL, confidence REAL,
    backend TEXT, primary_event TEXT, events_json TEXT, sectors TEXT, market_wide INTEGER, listicle INTEGER,
    summary TEXT, cluster_id TEXT, cluster_size INTEGER, cluster_sources INTEGER, ingested_at TEXT
);
CREATE TABLE IF NOT EXISTS article_entities (
    article_id TEXT, symbol TEXT, sector TEXT, sentiment TEXT, score REAL, sentiment_confidence REAL,
    relevance REAL, link_confidence REAL, matched TEXT, event TEXT, event_direction INTEGER,
    signal REAL, signal_label TEXT, PRIMARY KEY (article_id, symbol)
);
CREATE INDEX IF NOT EXISTS ix_ent_symbol ON article_entities(symbol);
CREATE INDEX IF NOT EXISTS ix_art_day ON articles(trading_day);
CREATE TABLE IF NOT EXISTS alerts (
    alert_id TEXT PRIMARY KEY, article_id TEXT, cluster_id TEXT, symbol TEXT, name TEXT, sector TEXT, title TEXT,
    source TEXT, published_ist TEXT, trading_day TEXT, event TEXT, sentiment TEXT, signal REAL,
    relevance REAL, confidence REAL, expected_move_pct REAL, expected_abs_move_pct REAL, hist_n INTEGER,
    hist_hit_rate REAL, priority REAL, severity TEXT, reason TEXT, realized_car_pct REAL
);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""


class Store:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or settings.db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.path, check_same_thread=False)
        self.con.executescript(SCHEMA)
        self.fts = self._init_fts()

    def _init_fts(self) -> bool:
        try:
            self.con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts USING fts5(id UNINDEXED, title, summary, source)")
            return True
        except sqlite3.OperationalError:
            return False

    # ------------------------------------------------------------------ writes
    def reset(self) -> None:
        for t in ("articles", "article_entities", "alerts"):
            self.con.execute(f"DELETE FROM {t}")
        if self.fts:
            self.con.execute("DELETE FROM articles_fts")
        self.con.commit()

    def upsert_articles(self, arts: pd.DataFrame, ents: pd.DataFrame) -> None:
        cols = [c[1] for c in self.con.execute("PRAGMA table_info(articles)")]
        arts = arts.reindex(columns=cols)
        ids = list(arts["id"])
        self.con.executemany(f"INSERT OR REPLACE INTO articles VALUES ({','.join('?' * len(cols))})",
                             arts.astype(object).where(arts.notna(), None).values.tolist())
        q = ",".join("?" * len(ids))
        if ids:
            for chunk in range(0, len(ids), 900):
                part = ids[chunk:chunk + 900]
                self.con.execute(f"DELETE FROM article_entities WHERE article_id IN ({','.join('?' * len(part))})", part)
                if self.fts:
                    self.con.execute(f"DELETE FROM articles_fts WHERE id IN ({','.join('?' * len(part))})", part)
        ecols = [c[1] for c in self.con.execute("PRAGMA table_info(article_entities)")]
        if len(ents):
            ents = ents.reindex(columns=ecols)
            self.con.executemany(f"INSERT OR REPLACE INTO article_entities VALUES ({','.join('?' * len(ecols))})",
                                 ents.astype(object).where(ents.notna(), None).values.tolist())
        if self.fts:
            self.con.executemany("INSERT INTO articles_fts VALUES (?,?,?,?)",
                                 arts[["id", "title", "summary", "source"]].fillna("").values.tolist())
        self.con.commit()
        del q

    def replace_alerts(self, alerts: pd.DataFrame) -> None:
        self.con.execute("DELETE FROM alerts")
        if len(alerts):
            cols = [c[1] for c in self.con.execute("PRAGMA table_info(alerts)")]
            a = alerts.reindex(columns=cols)
            self.con.executemany(f"INSERT OR REPLACE INTO alerts VALUES ({','.join('?' * len(cols))})",
                                 a.astype(object).where(a.notna(), None).values.tolist())
        self.con.commit()

    def set_meta(self, k: str, v) -> None:
        self.con.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (k, json.dumps(v, default=str)))
        self.con.commit()

    def get_meta(self, k: str, default=None):
        r = self.con.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return json.loads(r[0]) if r else default

    # ------------------------------------------------------------------ reads
    def articles(self) -> pd.DataFrame:
        return pd.read_sql("SELECT * FROM articles", self.con)

    def entities(self) -> pd.DataFrame:
        return pd.read_sql("SELECT * FROM article_entities", self.con)

    def alerts(self) -> pd.DataFrame:
        return pd.read_sql("SELECT * FROM alerts ORDER BY priority DESC", self.con)

    def search(self, query: str = "", symbol: str | None = None, sector: str | None = None,
               event: str | None = None, sentiment: str | None = None, source: str | None = None,
               date_from: str | None = None, date_to: str | None = None, limit: int = 200) -> pd.DataFrame:
        """Full-text + faceted search. Returns one row per (article, entity) when filtering by entity fields."""
        where, params = [], []
        base = ("SELECT a.id, a.published_ist, a.trading_day, a.source, a.title, a.sentiment, a.score, a.primary_event, "
                "a.sectors, a.cluster_size, a.backend, a.url, e.symbol, e.signal, e.signal_label, e.relevance, e.event "
                "FROM articles a LEFT JOIN article_entities e ON e.article_id = a.id")
        if query.strip():
            if self.fts:
                base += " JOIN articles_fts f ON f.id = a.id"
                where.append("articles_fts MATCH ?")
                params.append(_fts_query(query))
            else:
                where.append("a.title LIKE ?")
                params.append(f"%{query}%")
        for col, val in (("e.symbol", symbol), ("e.sector", sector), ("e.event", event), ("e.signal_label", sentiment), ("a.source", source)):
            if val:
                where.append(f"{col} = ?")
                params.append(val)
        if date_from:
            where.append("a.published_ist >= ?")
            params.append(str(date_from))
        if date_to:
            where.append("a.published_ist <= ?")
            params.append(str(date_to) + " 23:59:59")
        sql = base + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY a.published_ist DESC LIMIT ?"
        params.append(int(limit))
        try:
            return pd.read_sql(sql, self.con, params=params)
        except Exception:
            if query.strip() and self.fts:  # malformed FTS syntax -> plain LIKE
                self.fts = False
                try:
                    return self.search(query, symbol, sector, event, sentiment, source, date_from, date_to, limit)
                finally:
                    self.fts = True
            raise


def _fts_query(q: str) -> str:
    """Make free text safe for FTS5: quote each token, prefix-match the last one."""
    toks = [t.replace('"', "") for t in q.split() if t.strip('"')]
    if not toks:
        return '""'
    parts = [f'"{t}"' for t in toks[:-1]] + [f'"{toks[-1]}"*']
    return " ".join(parts)
