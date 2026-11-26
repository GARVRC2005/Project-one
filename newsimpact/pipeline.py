"""End-to-end pipeline: ingest -> NLP -> store -> event study -> alerts -> evaluation reports."""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import settings
from .impact import alerts as alerts_mod
from .impact import event_study as es
from .ingest import news as news_mod
from .ingest.prices import close_matrix, load_prices, volume_matrix
from .nlp.analyzer import analyze_many
from .nlp.entities import get_universe
from .store import Store

log = logging.getLogger(__name__)
JUNK = re.compile(r"^(?:News by .+ on TradingView|Stock market today|Live:|LIVE)|^[\W\d]+$", re.IGNORECASE)


def jsonable(o):
    """Recursively replace NaN/inf with None so reports are strict JSON."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df["title"] = df.title.fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[(df.title.str.len() >= 15) & ~df.title.str.contains(JUNK)]
    df["id"] = df.title.map(news_mod.article_id)
    df["published_utc"] = pd.to_datetime(df.published, utc=True, errors="coerce", format="mixed")
    df = df.dropna(subset=["published_utc"]).sort_values("published_utc")
    # exact-duplicate titles (same story syndicated / found by several company queries): keep earliest,
    # but remember every company query that surfaced it
    qs = df.groupby("id").query_symbol.apply(lambda s: ",".join(sorted({x for x in s if isinstance(x, str)})))
    df = df.drop_duplicates("id", keep="first").copy()
    df["query_symbol"] = df.id.map(qs)
    df["published_utc"] = df.published_utc.dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    return df.reset_index(drop=True)


def enrich(df: pd.DataFrame, use_llm: bool | None = None, chunk: int = 2000) -> tuple[pd.DataFrame, pd.DataFrame]:
    art_rows, ent_rows = [], []
    for s in range(0, len(df), chunk):
        part = df.iloc[s:s + chunk]
        res = analyze_many(part.title.tolist(), part.source.tolist(), use_llm=use_llm)
        for (_, r), a in zip(part.iterrows(), res):
            art_rows.append({
                "id": r.id, "title": r.title, "url": r.get("url", ""), "source": r.source, "published_utc": r.published_utc,
                "query_symbol": r.query_symbol, "sentiment": a.sentiment, "score": a.score, "confidence": a.confidence,
                "backend": a.backend, "primary_event": a.primary_event, "events_json": json.dumps(a.events),
                "sectors": ",".join(a.sectors), "market_wide": int(a.market_wide), "listicle": int(a.listicle),
                "summary": a.summary, "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            # NB: articles returned by a company search whose headline does not name the company are NOT
            # force-tagged (inspection showed most are search noise, e.g. a chess story for "Tech Mahindra").
            for e in a.entities:
                ent_rows.append({"article_id": r.id, **{k: v for k, v in e.__dict__.items()}})
        log.info("analysed %d/%d", min(s + chunk, len(df)), len(df))
    return pd.DataFrame(art_rows), pd.DataFrame(ent_rows)


def run(source: str = "snapshot", reset: bool = True, use_llm: bool | None = None, days: int = 7,
        price_source: str = "auto") -> dict:
    t0 = time.time()
    store = Store()
    if source == "snapshot":
        raw = news_mod.load_snapshot()
    elif source == "live":
        raw = news_mod.fetch_live(days=days)
        if raw.empty:
            log.warning("no live news fetched (offline?) - falling back to the bundled snapshot")
            raw, source = news_mod.load_snapshot(), "snapshot (live fetch failed)"
    else:
        raw = news_mod.load_file(source)
    df = clean(raw)
    if not reset:
        known = set(store.articles().id)
        df = df[~df.id.isin(known)]
    log.info("%d articles after cleaning", len(df))

    prices = load_prices(prefer=price_source)
    close, volume = close_matrix(prices), volume_matrix(prices)
    cal = es.trading_calendar(close)

    arts, ents = enrich(df, use_llm=use_llm)
    if len(arts):
        arts = pd.concat([arts, es.map_trading_day(arts.published_utc, cal)], axis=1)
    if reset:
        store.reset()
    else:
        old_a, old_e = store.articles(), store.entities()
        arts = pd.concat([old_a, arts], ignore_index=True)
        ents = pd.concat([old_e, ents], ignore_index=True)
    arts = alerts_mod.cluster_articles(arts, ents)
    store.upsert_articles(arts, ents)

    report = study_and_alert(store, arts, ents, close, volume)
    report.update({"source": source, "articles": int(len(arts)), "entity_mentions": int(len(ents)),
                   "runtime_sec": round(time.time() - t0, 1),
                   "llm_used": bool((arts.backend == "llm").any()) if len(arts) else False,
                   "price_range": [str(close.index.min().date()), str(close.index.max().date())] if len(close) else None})
    report["entity_eval"] = entity_recall(arts, ents)
    store.set_meta("last_run", jsonable({k: v for k, v in report.items() if k not in ("backtest",)}))
    (settings.reports_dir / "run_summary.json").write_text(json.dumps(jsonable(report), indent=2, default=str))
    return report


def study_and_alert(store: Store, arts: pd.DataFrame, ents: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame) -> dict:
    rep: dict = {}
    events = es.build_events(arts, ents)
    car = es.compute_car(events, close, volume)
    rd = settings.reports_dir
    if car.empty:
        log.warning("no events overlap with price history - impact study skipped")
        lookup = es.impact_lookup(pd.DataFrame(columns=["event", "signal_label"]), pd.DataFrame())
        alerts = alerts_mod.build_alerts(arts, ents, lookup)
        store.replace_alerts(alerts)
        return {"events": 0, "alerts": int(len(alerts))}
    car.to_csv(rd / "event_car.csv", index=False)
    s_tone = es.summarize(car, ["signal_label"])
    s_event = es.summarize(car, ["event"])
    s_et = es.summarize(car, ["event", "signal_label"])
    s_tone.to_csv(rd / "impact_by_tone.csv", index=False)
    s_event.to_csv(rd / "impact_by_event.csv", index=False)
    s_et.to_csv(rd / "impact_by_event_tone.csv", index=False)
    es.car_paths(car).to_csv(rd / "car_paths_by_tone.csv", index=False)
    lookup = es.impact_lookup(s_et, s_tone)
    car_key = car.groupby(["symbol", "trading_day"])["car_-1_1"].first().to_dict()
    alerts = alerts_mod.build_alerts(arts, ents, lookup, car_key)
    store.replace_alerts(alerts)
    bt = es.daily_signal_backtest(ents, arts, close)
    corr = es.same_vs_next_day_correlation(ents, arts, close)
    oos = evaluate_alerts_oos(arts, ents, car)
    (rd / "backtest.json").write_text(json.dumps(jsonable(bt), indent=2, default=str))
    (rd / "alert_eval_oos.json").write_text(json.dumps(jsonable(oos), indent=2, default=str))
    rep.update({"events": int(len(car)), "alerts": int(len(alerts)),
                "alerts_by_severity": alerts.severity.value_counts().to_dict() if len(alerts) else {},
                "impact_by_tone": s_tone.to_dict("records"), "signal_return_correlation": corr,
                "backtest": {k: v for k, v in bt.items() if k != "curve"}, "alert_eval_oos": oos})
    return rep


def evaluate_alerts_oos(arts: pd.DataFrame, ents: pd.DataFrame, car: pd.DataFrame, split: float = 0.6) -> dict:
    """Walk-forward check of the alert engine: impact table from the first 60% of days only,
    alerts scored on the last 40%, compared with realised CAR[-1,+1]."""
    days = sorted(car.trading_day.unique())
    if len(days) < 20:
        return {}
    cutoff = days[int(len(days) * split)]
    train = car[car.trading_day < cutoff]
    lookup = es.impact_lookup(es.summarize(train, ["event", "signal_label"]), es.summarize(train, ["signal_label"]))
    car_key = car.groupby(["symbol", "trading_day"])["car_-1_1"].first().to_dict()
    test_arts = arts[arts.trading_day >= cutoff]
    al = alerts_mod.build_alerts(test_arts, ents[ents.article_id.isin(test_arts.id)], lookup, car_key)
    al = al.dropna(subset=["realized_car_pct"])
    if al.empty:
        return {}
    sign = al.sentiment.map({"positive": 1, "negative": -1}).fillna(0)
    al = al.assign(abs_car=al.realized_car_pct.abs(), hit=np.where(sign != 0, np.sign(al.realized_car_pct) == sign, np.nan))
    by = al.groupby("severity").agg(n=("abs_car", "size"), mean_abs_realized_car_pct=("abs_car", "mean"),
                                    median_abs_realized_car_pct=("abs_car", "median"), direction_hit_rate=("hit", "mean"))
    from scipy import stats
    rho, p = stats.spearmanr(al.priority, al.abs_car)
    return {"cutoff_day": cutoff, "n_alerts_scored": int(len(al)),
            "by_severity": by.round(3).reset_index().to_dict("records"),
            "spearman_priority_vs_abs_move": round(float(rho), 3), "p_value": float(f"{p:.3g}"),
            "direction_hit_rate_all": round(float(np.nanmean(al.hit)), 3)}


def entity_recall(arts: pd.DataFrame, ents: pd.DataFrame) -> dict:
    """How often the company a headline was collected for is recovered from the headline text alone."""
    q = arts[arts.query_symbol.fillna("").str.len() > 0][["id", "query_symbol"]]
    if q.empty:
        return {}
    named = ents[ents.matched != "(search query)"].groupby("article_id").symbol.apply(set)
    hit = [bool(set(qs.split(",")) & named.get(i, set())) for i, qs in zip(q.id, q.query_symbol)]
    extra = ents[ents.matched != "(search query)"].groupby("article_id").size()
    return {"articles_with_query_company": int(len(q)), "query_company_found_in_headline": round(float(np.mean(hit)), 4),
            "avg_companies_per_article": round(float(extra.mean()), 3) if len(extra) else 0.0}
