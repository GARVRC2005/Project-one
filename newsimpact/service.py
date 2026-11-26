"""Shared helpers for the API, dashboard and CLI (single-text analysis with impact estimate)."""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from .config import settings
from .impact import event_study as es
from .impact.alerts import EVENT_LABEL, MATERIAL_EVENTS
from .nlp.analyzer import analyze
from .nlp.entities import get_universe


@lru_cache(maxsize=1)
def _lookup_cached(mtime: float):
    rd = settings.reports_dir
    try:
        et = pd.read_csv(rd / "impact_by_event_tone.csv")
        tone = pd.read_csv(rd / "impact_by_tone.csv")
    except FileNotFoundError:
        et, tone = pd.DataFrame(columns=["event", "signal_label"]), pd.DataFrame()
    return es.impact_lookup(et, tone)


def get_lookup():
    p = settings.reports_dir / "impact_by_event_tone.csv"
    return _lookup_cached(p.stat().st_mtime if p.exists() else 0.0)


def analyze_text(text: str, source: str | None = None, use_llm: bool | None = None) -> dict:
    a = analyze(text, source, use_llm=use_llm)
    u = get_universe()
    lookup = get_lookup()
    out = a.to_dict()
    impacts = []
    for e in a.entities:
        imp = lookup(e.event or "other_news", e.signal_label)
        material = e.signal_label != "neutral" or (e.event in MATERIAL_EVENTS)
        impacts.append({"symbol": e.symbol, "name": u.companies[e.symbol].name, "event": e.event,
                        "event_label": EVENT_LABEL.get(e.event or "other_news"), "tone": e.signal_label,
                        "signal": e.signal, "relevance": e.relevance, "alert_worthy": bool(material and e.relevance >= 0.25),
                        **imp})
    out["impact"] = impacts
    return out
