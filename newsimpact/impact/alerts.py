"""Stock-wise event alerts with relevance, confidence and expected-impact scores.

For every (news cluster, company) pair:
  relevance  = entity relevance from the analyzer (position in headline, # of companies, round-up
               penalty, link confidence, source tier), boosted when several outlets carry the story
  confidence = how sure we are about *what* the news says:
               0.45 * sentiment confidence + 0.25 * entity-link confidence + 0.20 * event-rule confidence
               + 0.10 * corroboration (distinct sources, capped at 3); +0.05 if LLM and local model agree
  impact     = historical, shrunk expected abnormal move for this (event type, tone) from the event study
  priority   = 100 * relevance * confidence * impact_factor * tone_factor     (0..100)
  severity   = HIGH >= 45, MEDIUM >= 28, else LOW
Only company events are alerted (see `is_alertable`); price commentary / macro stay in the searchable feed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..nlp.entities import get_universe

MATERIAL_EVENTS = {"merger_acquisition", "management_change", "regulatory_legal", "earnings_surprise",
                   "stake_sale", "order_win", "capital_raise", "guidance", "operational_risk", "dividend_buyback"}
EVENT_LABEL = {
    "earnings": "results", "earnings_surprise": "earnings surprise", "merger_acquisition": "M&A",
    "management_change": "management change", "dividend_buyback": "dividend/buyback", "order_win": "order win",
    "regulatory_legal": "regulatory/legal action", "regulatory_approval": "regulatory approval",
    "rating_change": "broker/rating action", "capital_raise": "capital raise", "stake_sale": "stake sale/block deal",
    "guidance": "guidance", "business_update": "business update", "expansion_launch": "expansion/launch",
    "operational_risk": "operational risk", "macro_policy": "macro/policy", "price_action": "price move",
    "other_news": "news",
}


CORPORATE_EVENTS = MATERIAL_EVENTS | {"earnings", "regulatory_approval", "rating_change", "business_update",
                                      "expansion_launch"}


def is_alertable(event: str, tone: str, signal: float, relevance: float) -> bool:
    """Alerts are for *new information about a company*. Price commentary ("shares jump 3%"),
    macro and generic news only show in the feed unless they are strongly toned and squarely about one stock."""
    if event in MATERIAL_EVENTS:
        return True
    if event in CORPORATE_EVENTS:
        return tone != "neutral"
    return tone != "neutral" and abs(signal) >= 0.6 and relevance >= 0.7 and event != "price_action"


def cluster_articles(arts: pd.DataFrame, ents: pd.DataFrame, window_hours: int = 72, thresh: float = 0.5) -> pd.DataFrame:
    """Near-duplicate clustering per lead company (token Jaccard within a time window).
    Returns arts with cluster_id / cluster_size / cluster_sources."""
    import re
    arts = arts.copy()
    lead = ents.sort_values("relevance", ascending=False).drop_duplicates("article_id").set_index("article_id").symbol
    arts["lead_"] = arts.id.map(lead).fillna("_none")
    arts["t_"] = pd.to_datetime(arts.published_utc, utc=True, errors="coerce", format="mixed")
    arts["tok_"] = arts.title.fillna("").str.lower().map(lambda s: frozenset(w for w in re.findall(r"[a-z0-9]+", s) if len(w) > 2))
    cid = {}
    for _, g in arts.sort_values("t_").groupby("lead_"):
        reps: list[tuple[pd.Timestamp, frozenset, str]] = []
        for r in g.itertuples():
            best = None
            for t, tok, c in reversed(reps):
                if pd.notna(r.t_) and pd.notna(t) and (r.t_ - t).total_seconds() > window_hours * 3600:
                    break
                j = len(r.tok_ & tok) / max(1, len(r.tok_ | tok))
                if j >= thresh:
                    best = c
                    break
            if best is None:
                best = r.id
                reps.append((r.t_, r.tok_, best))
            cid[r.id] = best
    arts["cluster_id"] = arts.id.map(cid)
    size = arts.groupby("cluster_id").id.transform("size")
    srcs = arts.groupby("cluster_id").source.transform("nunique")
    arts["cluster_size"], arts["cluster_sources"] = size, srcs
    return arts.drop(columns=["lead_", "t_", "tok_"])


def build_alerts(arts: pd.DataFrame, ents: pd.DataFrame, lookup, car_by_key: dict | None = None,
                 min_relevance: float = 0.25) -> pd.DataFrame:
    u = get_universe()
    df = ents.merge(arts, left_on="article_id", right_on="id", suffixes=("", "_a"))
    df = df[df.relevance >= min_relevance]
    # one alert per (cluster, symbol): keep the most relevant / earliest article of the cluster
    df = df.sort_values(["relevance", "published_utc"], ascending=[False, True]).drop_duplicates(["cluster_id", "symbol"])
    car_by_key = car_by_key or {}
    rows = []
    for r in df.itertuples(index=False):
        event = r.event if isinstance(r.event, str) and r.event else "other_news"
        tone = r.signal_label
        if not is_alertable(event, tone, r.signal, r.relevance):
            continue
        imp = lookup(event, tone)
        n_src = int(r.cluster_sources or 1)
        rel = min(1.0, r.relevance * (1 + 0.1 * (min(n_src, 4) - 1)))
        ev_conf = 0.6
        try:
            import json
            evs = json.loads(r.events_json or "[]")
            ev_conf = next((e["confidence"] for e in evs if e["type"] == event), 0.6)
        except Exception:
            pass
        conf = 0.45 * r.sentiment_confidence + 0.25 * r.link_confidence + 0.20 * ev_conf + 0.10 * min(1.0, n_src / 3)
        if r.backend == "llm":
            conf += 0.05
        conf = float(min(conf, 1.0))
        impact_factor = float(np.clip(imp["expected_abs_move_pct"] / 3.0, 0.2, 1.0))
        tone_factor = 0.5 + 0.5 * min(1.0, abs(r.signal) * 1.25) if tone != "neutral" else 0.6
        prio = round(100 * rel * conf * impact_factor * tone_factor, 1)
        sev = "HIGH" if prio >= 45 else "MEDIUM" if prio >= 28 else "LOW"
        name = u.companies[r.symbol].name if r.symbol in u.companies else r.symbol
        hist = (f"Historically {EVENT_LABEL.get(event, event)} news with {tone} tone moved NIFTY 50 stocks "
                f"{imp['expected_move_pct']:+.2f}% (abnormal, day -1..+1; typical size ±{imp['expected_abs_move_pct']:.2f}%, n={imp['hist_n']}).")
        reason = (f"{name}: {EVENT_LABEL.get(event, event)} reported with {tone} tone "
                  f"(signal {r.signal:+.2f}). Covered by {n_src} source(s). {hist}")
        realized = car_by_key.get((r.symbol, r.trading_day))
        rows.append({
            "alert_id": f"{r.cluster_id}:{r.symbol}", "article_id": r.article_id, "cluster_id": r.cluster_id,
            "symbol": r.symbol, "name": name, "sector": r.sector, "title": r.title, "source": r.source,
            "published_ist": r.published_ist, "trading_day": r.trading_day, "event": event, "sentiment": tone,
            "signal": round(float(r.signal), 3), "relevance": round(rel, 3), "confidence": round(conf, 3),
            "expected_move_pct": imp["expected_move_pct"], "expected_abs_move_pct": imp["expected_abs_move_pct"],
            "hist_n": imp["hist_n"], "hist_hit_rate": imp["hist_hit_rate"], "priority": prio, "severity": sev,
            "reason": reason, "realized_car_pct": round(100 * realized, 3) if realized is not None and pd.notna(realized) else None,
        })
    return pd.DataFrame(rows).sort_values("priority", ascending=False) if rows else pd.DataFrame()
