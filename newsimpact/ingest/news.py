"""News ingestion: live RSS / NSE announcements, or the bundled offline snapshot.

Live sources (no API key needed):
  * Google News RSS search per covered company (India edition, last N days)
  * Market RSS feeds: Economic Times Markets, Moneycontrol, Livemint, Business Standard
  * NSE corporate announcements (best effort; NSE requires a browser-like session)
Every source is fetched independently with timeouts; a failing source is logged and skipped.
"""
from __future__ import annotations

import hashlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from urllib.parse import quote_plus

import pandas as pd
import requests

from ..config import settings
from ..nlp.entities import get_universe

log = logging.getLogger(__name__)
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"}

MARKET_FEEDS = {
    "The Economic Times": "https://economictimes.indiatimes.com/markets/stocks/news/rssfeeds/2146842.cms",
    "The Economic Times ": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "Moneycontrol.com": "https://www.moneycontrol.com/rss/business.xml",
    "Moneycontrol.com ": "https://www.moneycontrol.com/rss/marketreports.xml",
    "Livemint": "https://www.livemint.com/rss/markets",
    "Business Standard": "https://www.business-standard.com/rss/markets-106.rss",
}

COLUMNS = ["published", "source", "title", "query_symbol", "url"]


def article_id(title: str) -> str:
    key = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def _parse_feed(content: bytes, default_source: str, query_symbol: str | None) -> list[dict]:
    import feedparser
    fp = feedparser.parse(content)
    rows = []
    for e in fp.entries:
        title = (e.get("title") or "").strip()
        source = default_source
        # Google News appends " - Publisher" to the title
        if "news.google.com" in (e.get("link") or "") and " - " in title:
            title, source = title.rsplit(" - ", 1)
        src = e.get("source")
        if isinstance(src, dict) and src.get("title"):
            source = src["title"]
        ts = e.get("published_parsed") or e.get("updated_parsed")
        published = datetime(*ts[:6], tzinfo=timezone.utc).isoformat() if ts else datetime.now(timezone.utc).isoformat()
        rows.append({"published": published, "source": source.strip(), "title": title,
                     "query_symbol": query_symbol, "url": e.get("link", "")})
    return rows


def _get(url: str, timeout: int = 15) -> bytes:
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    return r.content


def fetch_google_news(days: int = 7, symbols: list[str] | None = None, workers: int = 8) -> pd.DataFrame:
    u = get_universe()
    symbols = symbols or list(u.companies)
    jobs = {}
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for s in symbols:
            name = u.companies[s].name.split(" (")[0]
            q = quote_plus(f'"{name}" when:{days}d')
            url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
            jobs[ex.submit(_get, url)] = s
        for f in as_completed(jobs):
            s = jobs[f]
            try:
                rows += _parse_feed(f.result(), "Google News", s)
            except Exception as e:
                log.warning("google news failed for %s: %s", s, e)
    return pd.DataFrame(rows, columns=COLUMNS)


def fetch_market_feeds() -> pd.DataFrame:
    rows: list[dict] = []
    for name, url in MARKET_FEEDS.items():
        try:
            rows += _parse_feed(_get(url), name.strip(), None)
        except Exception as e:
            log.warning("feed %s failed: %s", name, e)
    return pd.DataFrame(rows, columns=COLUMNS)


def fetch_nse_announcements(symbols: list[str] | None = None) -> pd.DataFrame:
    """Corporate filings from NSE (best effort - NSE blocks non-browser clients intermittently)."""
    u = get_universe()
    symbols = symbols or list(u.companies)
    s = requests.Session()
    s.headers.update({**UA, "Accept": "application/json", "Referer": "https://www.nseindia.com/"})
    rows = []
    try:
        s.get("https://www.nseindia.com/", timeout=10)
        for sym in symbols:
            try:
                r = s.get("https://www.nseindia.com/api/corporate-announcements",
                          params={"index": "equities", "symbol": sym}, timeout=10)
                for it in (r.json() or [])[:20]:
                    desc = (it.get("desc") or "").strip()
                    txt = (it.get("attchmntText") or "").strip()
                    title = f"{u.companies[sym].name}: {desc}" + (f" - {txt[:200]}" if txt else "")
                    dt = pd.to_datetime(it.get("an_dt") or it.get("sort_date"), errors="coerce", dayfirst=True)
                    if pd.isna(dt):
                        continue
                    published = dt.tz_localize("Asia/Kolkata").tz_convert("UTC").isoformat()
                    rows.append({"published": published, "source": "NSE", "title": title, "query_symbol": sym,
                                 "url": it.get("attchmntFile", "")})
            except Exception as e:
                log.debug("nse %s: %s", sym, e)
    except Exception as e:
        log.warning("NSE announcements unavailable: %s", e)
    return pd.DataFrame(rows, columns=COLUMNS)


def fetch_live(days: int = 7, include_nse: bool = True) -> pd.DataFrame:
    parts = [fetch_google_news(days), fetch_market_feeds()]
    if include_nse:
        parts.append(fetch_nse_announcements())
    df = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else pd.DataFrame(columns=COLUMNS)
    log.info("fetched %d raw items", len(df))
    return df


def load_snapshot() -> pd.DataFrame:
    return pd.read_csv(settings.snapshot_dir / "headlines.csv").reindex(columns=COLUMNS)


def load_file(path: str) -> pd.DataFrame:
    """Load user-provided news (CSV/JSON with at least `title` and `published`)."""
    df = pd.read_json(path) if str(path).endswith(".json") else pd.read_csv(path)
    if "title" not in df.columns:
        for c in ("headline", "text", "Title", "Headline"):
            if c in df.columns:
                df = df.rename(columns={c: "title"})
                break
    df = df.reindex(columns=COLUMNS)
    df["published"] = df["published"].fillna(datetime.now(timezone.utc).isoformat())
    df["source"] = df["source"].fillna("user")
    return df
