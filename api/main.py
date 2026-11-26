"""REST API.  Run:  uvicorn api.main:app --reload      (docs at http://127.0.0.1:8000/docs)"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from newsimpact.config import settings  # noqa: E402
from newsimpact.nlp.entities import get_universe  # noqa: E402
from newsimpact.service import analyze_text  # noqa: E402
from newsimpact.store import Store  # noqa: E402

app = FastAPI(title="News Sentiment & Stock Impact Analyzer", version="1.0",
              description="NLP over Indian market news: companies, sectors, events, sentiment, historical impact and alerts.")
_store: Store | None = None


def store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store


def records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.replace({np.nan: None}).to_json(orient="records"))


class AnalyzeIn(BaseModel):
    text: str = Field(..., min_length=3, max_length=20000, examples=["Infosys beats estimates, raises FY27 guidance"])
    source: Optional[str] = None
    use_llm: Optional[bool] = None


@app.get("/health")
def health():
    s = store()
    return {"status": "ok", "articles": int(s.con.execute("SELECT COUNT(*) FROM articles").fetchone()[0]),
            "alerts": int(s.con.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]),
            "llm": settings.resolved_llm()[:2] if settings.resolved_llm() else None, "fts5": s.fts,
            "last_run": s.get_meta("last_run", {}).get("source")}


@app.post("/analyze")
def analyze(body: AnalyzeIn):
    return analyze_text(body.text, body.source, use_llm=body.use_llm)


@app.get("/news")
def news(q: str = "", symbol: Optional[str] = None, sector: Optional[str] = None, event: Optional[str] = None,
         sentiment: Optional[str] = Query(None, pattern="^(positive|neutral|negative)$"), source: Optional[str] = None,
         date_from: Optional[str] = None, date_to: Optional[str] = None, limit: int = Query(100, le=1000)):
    return records(store().search(q, symbol and symbol.upper(), sector, event, sentiment, source, date_from, date_to, limit))


@app.get("/alerts")
def alerts(symbol: Optional[str] = None, severity: Optional[str] = Query(None, pattern="^(HIGH|MEDIUM|LOW)$"),
           event: Optional[str] = None, since: Optional[str] = None, limit: int = Query(100, le=2000)):
    df = store().alerts()
    if symbol:
        df = df[df.symbol == symbol.upper()]
    if severity:
        df = df[df.severity == severity]
    if event:
        df = df[df.event == event]
    if since:
        df = df[df.published_ist >= since]
    return records(df.sort_values("published_ist", ascending=False).head(limit))


@app.get("/stocks/{symbol}")
def stock(symbol: str, days: int = 30):
    u = get_universe()
    symbol = symbol.upper()
    if symbol not in u.companies:
        raise HTTPException(404, f"unknown symbol {symbol}; covered: {sorted(u.companies)}")
    s = store()
    df = s.search(symbol=symbol, limit=5000)
    df["published_ist"] = pd.to_datetime(df.published_ist)
    recent = df[df.published_ist >= df.published_ist.max() - pd.Timedelta(days=days)] if len(df) else df
    c = u.companies[symbol]
    return {"symbol": symbol, "name": c.name, "sector": c.sector, "articles": int(len(df)),
            "recent_articles": int(len(recent)), "recent_avg_tone": round(float(recent.signal.mean()), 3) if len(recent) else None,
            "recent_tone_counts": recent.signal_label.value_counts().to_dict() if len(recent) else {},
            "alerts": records(s.alerts().query("symbol == @symbol").head(20)),
            "latest_news": records(df.head(20).assign(published_ist=lambda x: x.published_ist.astype(str)))}


@app.get("/impact")
def impact(min_n: int = 10):
    p = settings.reports_dir / "impact_by_event_tone.csv"
    if not p.exists():
        raise HTTPException(404, "run the pipeline first")
    df = pd.read_csv(p)
    return {"by_event_tone": records(df[df.n >= min_n]),
            "by_tone": records(pd.read_csv(settings.reports_dir / "impact_by_tone.csv")),
            "summary": json.loads((settings.reports_dir / "run_summary.json").read_text())}


@app.get("/universe")
def universe():
    return [c.__dict__ for c in get_universe().companies.values()]
