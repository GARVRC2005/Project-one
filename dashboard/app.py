"""Market News Intelligence dashboard.

Run:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from newsimpact.config import settings  # noqa: E402
from newsimpact.impact.alerts import EVENT_LABEL  # noqa: E402
from newsimpact.ingest.prices import load_prices  # noqa: E402
from newsimpact.nlp.entities import get_universe  # noqa: E402
from newsimpact.store import Store  # noqa: E402

st.set_page_config(page_title="News Impact Analyzer", page_icon="📰", layout="wide")
POS, NEU, NEG = "#1a9850", "#9e9e9e", "#d73027"
TONE_COLORS = {"positive": POS, "neutral": NEU, "negative": NEG}
SEV_ORDER = ["HIGH", "MEDIUM", "LOW"]
U = get_universe()


# ----------------------------------------------------------------------------- data
def _db_stamp() -> float:
    p = settings.db_path
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_resource
def store() -> Store:
    return Store()


@st.cache_data(show_spinner=False)
def load_all(stamp: float):
    s = store()
    arts, ents, alerts = s.articles(), s.entities(), s.alerts()
    meta = s.get_meta("last_run", {})
    if len(arts):
        arts["day"] = pd.to_datetime(arts.published_ist.str[:10], errors="coerce")
    j = ents.merge(arts[["id", "day", "title", "source", "published_ist", "trading_day", "listicle", "url"]],
                   left_on="article_id", right_on="id", how="left") if len(ents) else pd.DataFrame()
    return arts, ents, j, alerts, meta


@st.cache_data(show_spinner=False)
def report(name: str, stamp: float):
    p = settings.reports_dir / name
    if not p.exists():
        return None
    return json.loads(p.read_text()) if p.suffix == ".json" else pd.read_csv(p)


@st.cache_data(show_spinner=False)
def prices_for(symbol: str):
    return load_prices([symbol, settings.benchmark_symbol])


stamp = _db_stamp()
if stamp == 0.0:
    # First start (e.g. on a fresh cloud deploy): build the database from the bundled snapshot automatically.
    st.title("📰 News Sentiment & Stock Impact Analyzer")
    from newsimpact import pipeline
    with st.spinner("First start: analysing ~16k headlines and running the event study (about a minute)..."):
        try:
            pipeline.run("snapshot")
        except Exception as e:  # noqa: BLE001
            st.error(f"Build failed: {e}. Run `python -m newsimpact.cli build` and reload.")
            st.stop()
    st.cache_data.clear()
    st.rerun()

arts, ents, joined, alerts, meta = load_all(stamp)

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("📰 News Impact Analyzer")
    st.caption("NIFTY 50 · NLP + event study")
    llm = settings.resolved_llm()
    st.markdown(f"**Data:** {meta.get('source', '?')}  \n**Articles:** {len(arts):,}  \n"
                f"**Prices:** {' → '.join(meta.get('price_range') or ['?'])}  \n"
                f"**NLP backend:** {'LLM: ' + llm[0] + ' / ' + llm[1] if llm else 'local model (no LLM key)'}")
    dmin, dmax = arts.day.min().date(), arts.day.max().date()
    period = st.date_input("Period", (dmin, dmax), min_value=dmin, max_value=dmax)
    d0, d1 = (pd.Timestamp(period[0]), pd.Timestamp(period[1])) if isinstance(period, (tuple, list)) and len(period) == 2 else (pd.Timestamp(dmin), pd.Timestamp(dmax))
    watch = st.multiselect("Watchlist (filters alerts)", sorted(U.companies), [])
    st.divider()
    if st.button("🔄 Fetch latest news (live)", help="Pulls Google News / ET / Moneycontrol / Mint / BS RSS, NSE filings and yfinance prices. Needs internet."):
        from newsimpact import pipeline
        from newsimpact.ingest.prices import download
        with st.spinner("Fetching and analysing..."):
            try:
                download()
            except Exception as e:  # noqa: BLE001
                st.info(f"Price refresh failed ({e}); using cached prices.")
            r = pipeline.run("live", reset=False, days=3)
        st.success(f"Done: {r.get('articles', 0):,} articles in DB")
        st.cache_data.clear()
        st.rerun()

in_period = lambda d: (d >= d0) & (d <= d1 + pd.Timedelta(days=1))  # noqa: E731
A = arts[in_period(arts.day)]
J = joined[in_period(joined.day)] if len(joined) else joined
AL = alerts.copy()
AL["day"] = pd.to_datetime(AL.published_ist.str[:10], errors="coerce")
AL = AL[in_period(AL.day)]
if watch:
    AL = AL[AL.symbol.isin(watch)]

tabs = st.tabs(["Overview", "Alerts", "News search", "Stock view", "Impact study", "Analyze text", "Model quality"])

# ----------------------------------------------------------------------------- overview
with tabs[0]:
    c = st.columns(5)
    c[0].metric("Articles", f"{len(A):,}")
    c[1].metric("Companies covered", J.symbol.nunique() if len(J) else 0)
    c[2].metric("HIGH alerts", int((AL.severity == "HIGH").sum()))
    net = J.signal.mean() if len(J) else 0
    c[3].metric("Avg news tone", f"{net:+.2f}")
    share_neg = (J.signal_label == "negative").mean() if len(J) else 0
    c[4].metric("Negative share", f"{share_neg:.0%}")

    l, r = st.columns(2)
    with l:
        st.subheader("Sector sentiment")
        sec = J[J.listicle == 0].groupby("sector").agg(tone=("signal", "mean"), n=("signal", "size")).reset_index()
        sec = sec[sec.n >= 5].sort_values("tone")
        fig = px.bar(sec, x="tone", y="sector", orientation="h", color="tone", color_continuous_scale=[NEG, "#f7f7f7", POS],
                     range_color=(-0.5, 0.5), hover_data=["n"])
        fig.update_layout(height=520, coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0), yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
    with r:
        st.subheader("Most positive / negative stocks (relevance-weighted)")
        sc = J[J.listicle == 0].assign(w=lambda x: x.signal * x.relevance).groupby("symbol").agg(
            net=("w", "sum"), n=("w", "size")).reset_index()
        top = pd.concat([sc.nlargest(8, "net"), sc.nsmallest(8, "net")]).drop_duplicates().sort_values("net")
        fig = px.bar(top, x="net", y="symbol", orientation="h", color="net", color_continuous_scale=[NEG, "#f7f7f7", POS],
                     hover_data=["n"])
        fig.update_layout(height=520, coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0), yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Daily news tone vs NIFTY 50")
    daily = J.groupby("day").agg(tone=("signal", "mean"), n=("signal", "size")).reset_index()
    px_n = prices_for(settings.benchmark_symbol).get(settings.benchmark_symbol)
    fig = go.Figure()
    fig.add_bar(x=daily.day, y=daily.tone.rolling(3, min_periods=1).mean(), name="news tone (3d avg)",
                marker_color=[POS if v >= 0 else NEG for v in daily.tone.rolling(3, min_periods=1).mean()], yaxis="y")
    if px_n is not None:
        pn = px_n[(px_n.index >= d0) & (px_n.index <= d1)]
        fig.add_scatter(x=pn.index, y=pn.close, name="NIFTY 50", yaxis="y2", line=dict(color="#2c7fb8"))
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), yaxis=dict(title="tone"),
                      yaxis2=dict(title="NIFTY 50", overlaying="y", side="right"), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("What kind of news")
    ev = J.assign(event=J.event.fillna("other_news").map(lambda e: EVENT_LABEL.get(e, e))).groupby(["event", "signal_label"]).size().reset_index(name="n")
    fig = px.bar(ev, x="n", y="event", color="signal_label", orientation="h", color_discrete_map=TONE_COLORS)
    fig.update_layout(height=450, margin=dict(l=0, r=0, t=10, b=0), yaxis={"categoryorder": "total ascending"}, yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------- alerts
with tabs[1]:
    st.subheader("Stock-wise event alerts")
    st.caption("Priority = relevance × confidence × historical impact × tone strength. "
               "Expected move = shrunk average abnormal return (day −1..+1) of past events of the same type and tone.")
    f1, f2, f3, f4 = st.columns(4)
    sev = f1.multiselect("Severity", SEV_ORDER, ["HIGH", "MEDIUM"])
    syms = f2.multiselect("Symbol", sorted(AL.symbol.dropna().unique()))
    evs = f3.multiselect("Event", sorted(AL.event.dropna().unique()), format_func=lambda e: EVENT_LABEL.get(e, e))
    tone = f4.multiselect("Tone", ["positive", "negative", "neutral"])
    X = AL[AL.severity.isin(sev or SEV_ORDER)]
    if syms:
        X = X[X.symbol.isin(syms)]
    if evs:
        X = X[X.event.isin(evs)]
    if tone:
        X = X[X.sentiment.isin(tone)]
    X = X.sort_values(["published_ist", "priority"], ascending=[False, False])
    st.write(f"{len(X):,} alerts")
    show = X[["severity", "priority", "published_ist", "symbol", "event", "sentiment", "relevance", "confidence",
              "expected_move_pct", "hist_n", "realized_car_pct", "title", "source"]].copy()
    show["event"] = show.event.map(lambda e: EVENT_LABEL.get(e, e))
    st.dataframe(show, use_container_width=True, hide_index=True, height=420, column_config={
        "priority": st.column_config.ProgressColumn("priority", min_value=0, max_value=100, format="%.0f"),
        "relevance": st.column_config.NumberColumn(format="%.2f"), "confidence": st.column_config.NumberColumn(format="%.2f"),
        "expected_move_pct": st.column_config.NumberColumn("exp. move %", format="%+.2f"),
        "realized_car_pct": st.column_config.NumberColumn("realised CAR %", format="%+.2f",
                                                          help="Actual abnormal return day −1..+1 (back-test view; empty for recent news)"),
        "hist_n": st.column_config.NumberColumn("hist. n"),
    })
    st.markdown("**Latest high-priority alerts**")
    for _, a in X[X.severity == "HIGH"].head(8).iterrows():
        icon = "🟢" if a.sentiment == "positive" else "🔴" if a.sentiment == "negative" else "⚪"
        with st.expander(f"{icon} {a.symbol} · {EVENT_LABEL.get(a.event, a.event)} · priority {a.priority:.0f} — {a.title[:110]}"):
            st.write(a.reason)
            st.caption(f"{a.source} · {a.published_ist} IST · relevance {a.relevance:.2f} · confidence {a.confidence:.2f}")

# ----------------------------------------------------------------------------- search
with tabs[2]:
    st.subheader("Search market news")
    q1, q2, q3, q4, q5 = st.columns([3, 1, 1, 1, 1])
    q = q1.text_input("Full-text search", placeholder="e.g. block deal, USFDA, guidance, Infosys deal")
    s_sym = q2.selectbox("Symbol", [""] + sorted(U.companies))
    s_sec = q3.selectbox("Sector", [""] + sorted({c.sector for c in U.companies.values()}))
    s_ev = q4.selectbox("Event", [""] + list(EVENT_LABEL), format_func=lambda e: EVENT_LABEL.get(e, "any") if e else "any")
    s_tone = q5.selectbox("Tone", ["", "positive", "neutral", "negative"])
    res = store().search(q, symbol=s_sym or None, sector=s_sec or None, event=s_ev or None, sentiment=s_tone or None,
                         date_from=str(d0.date()), date_to=str(d1.date()), limit=500)
    st.write(f"{len(res):,} results" + (" (showing first 500)" if len(res) == 500 else ""))
    st.dataframe(res[["published_ist", "symbol", "title", "source", "event", "signal_label", "signal", "relevance", "url"]],
                 use_container_width=True, hide_index=True, height=560,
                 column_config={"url": st.column_config.LinkColumn("link", display_text="open"),
                                "signal": st.column_config.NumberColumn(format="%+.2f"),
                                "relevance": st.column_config.NumberColumn(format="%.2f")})

# ----------------------------------------------------------------------------- stock view
with tabs[3]:
    sym = st.selectbox("Stock", sorted(U.companies), index=sorted(U.companies).index("RELIANCE"),
                       format_func=lambda s: f"{s} — {U.companies[s].name}")
    P = prices_for(sym).get(sym)
    SJ = joined[(joined.symbol == sym)]
    c = st.columns(4)
    c[0].metric("Articles", len(SJ))
    c[1].metric("Avg tone", f"{SJ.signal.mean():+.2f}" if len(SJ) else "–")
    c[2].metric("Alerts", int((alerts.symbol == sym).sum()))
    c[3].metric("Sector", U.companies[sym].sector)
    if P is not None:
        fig = go.Figure()
        fig.add_candlestick(x=P.index, open=P.open, high=P.high, low=P.low, close=P.close, name=sym, showlegend=False)
        nd = SJ[SJ.listicle == 0].dropna(subset=["trading_day"]).copy()
        if len(nd):
            nd["td"] = pd.to_datetime(nd.trading_day)
            agg = nd.groupby("td").agg(signal=("signal", "mean"), n=("signal", "size"),
                                       title=("title", lambda s: "<br>".join(t[:90] for t in list(s)[:4]))).reset_index()
            agg = agg[agg.td.isin(P.index)]
            yv = P.loc[agg.td, "high"].values * 1.02
            fig.add_scatter(x=agg.td, y=yv, mode="markers", name="news",
                            marker=dict(size=6 + 2 * agg.n.clip(upper=8), color=agg.signal, colorscale=[[0, NEG], [0.5, "#dddddd"], [1, POS]],
                                        cmin=-1, cmax=1, line=dict(width=0.5, color="#333")),
                            text=agg.title, hovertemplate="%{x|%d %b %Y}<br>%{text}<extra></extra>")
        fig.update_layout(height=480, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Markers = news days (colour = tone, size = number of articles). Hover for headlines.")
    car = report("event_car.csv", stamp)
    l, r = st.columns([3, 2])
    with l:
        st.markdown("**Recent news**")
        st.dataframe(SJ.sort_values("published_ist", ascending=False)[["published_ist", "title", "source", "event", "signal_label", "signal", "relevance"]],
                     hide_index=True, use_container_width=True, height=380)
    with r:
        st.markdown("**How this stock reacted to its news (abnormal return, day −1..+1)**")
        if car is not None:
            cs = car[car.symbol == sym]
            if len(cs):
                t = cs.groupby("signal_label")["car_-1_1"].agg(["count", "mean", "median"]).mul([1, 100, 100]).round(2)
                st.dataframe(t, use_container_width=True)
                fig = px.strip(cs.assign(car=cs["car_-1_1"] * 100), x="signal_label", y="car", color="signal_label",
                               color_discrete_map=TONE_COLORS, hover_data=["trading_day", "event", "example"])
                fig.update_layout(height=300, showlegend=False, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="CAR %")
                st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------- impact study
with tabs[4]:
    st.subheader("Do news events move prices? (event study vs NIFTY 50 market model)")
    paths = report("car_paths_by_tone.csv", stamp)
    by_et = report("impact_by_event_tone.csv", stamp)
    summ = report("run_summary.json", stamp) or {}
    bt = report("backtest.json", stamp) or {}
    oos = report("alert_eval_oos.json", stamp) or {}
    l, r = st.columns(2)
    with l:
        if paths is not None:
            fig = px.line(paths, x="day", y="cum_ar_pct", color="signal_label", color_discrete_map=TONE_COLORS, markers=True,
                          labels={"day": "trading days relative to news", "cum_ar_pct": "cumulative abnormal return %"})
            fig.add_vline(x=0, line_dash="dot")
            fig.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0), title="Average cumulative abnormal return around news")
            st.plotly_chart(fig, use_container_width=True)
    with r:
        corr = summ.get("signal_return_correlation", {})
        if corr:
            cdf = pd.DataFrame(corr).T.reset_index().rename(columns={"index": "return window"})
            fig = px.bar(cdf, x="return window", y="spearman", text="spearman", title="Rank correlation: news tone vs excess return")
            fig.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
    st.info("Reading: prices move **with** the news on the day it becomes public (strong same-session correlation, "
            "and clear drift on days −1..0), but the tone of public headlines has **no reliable predictive power for the next "
            "session**. News is therefore most useful for *explaining and alerting*, not as a stand-alone trading signal.")
    if by_et is not None:
        st.markdown("**Impact by event type and tone** (events with n ≥ 15)")
        t = by_et[by_et.n >= 15].copy()
        t["event"] = t.event.map(lambda e: EVENT_LABEL.get(e, e))
        cols = ["event", "signal_label", "n", "mean_car_0_0_pct", "mean_car_-1_1_pct", "t_-1_1", "mean_car_1_3_pct", "t_1_3",
                "mean_car_-5_-1_pct", "hit_rate_-1_1", "median_abn_volume"]
        st.dataframe(t[cols].sort_values("t_-1_1"), hide_index=True, use_container_width=True, height=420,
                     column_config={"t_-1_1": st.column_config.NumberColumn("t (−1..+1)", help="|t| > 2 ≈ significant at 5%"),
                                    "t_1_3": st.column_config.NumberColumn("t (+1..+3)")})
    l, r = st.columns(2)
    with l:
        st.markdown("**Walk-forward check of the alert engine**")
        if oos:
            st.caption(f"Impact table learned on news before {oos['cutoff_day']}, alerts scored after it.")
            o = pd.DataFrame(oos["by_severity"]).set_index("severity").reindex(SEV_ORDER)
            st.dataframe(o, use_container_width=True)
            st.write(f"Spearman(priority, |realised move|) = **{oos['spearman_priority_vs_abs_move']}** (p = {oos['p_value']}); "
                     f"direction hit rate = **{oos['direction_hit_rate_all']:.1%}**")
    with r:
        st.markdown("**Naïve trading back-test** (long positive / short negative news at the close, hold 1 day)")
        if bt.get("curve"):
            cv = pd.DataFrame(bt["curve"])
            fig = px.line(cv, x="date", y=["equity_gross", "equity_net"])
            fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), legend_title=None, yaxis_title="equity (excess vs NIFTY)")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Mean {bt['mean_daily_excess_bps']} bps/day (t = {bt['t_stat']}), hit rate {bt['hit_rate']:.0%}, "
                       f"Sharpe gross {bt['sharpe_gross']}, net of 10 bps/side {bt['sharpe_net_of_costs']}.")

# ----------------------------------------------------------------------------- analyze
with tabs[5]:
    st.subheader("Analyze any headline, article or announcement")
    txt = st.text_area("Text", "Infosys beats Street estimates in Q2, raises FY27 revenue guidance; HCL Tech misses estimates", height=120)
    use_llm = st.toggle("Use LLM if an API key is configured", value=bool(settings.resolved_llm()), disabled=not settings.resolved_llm())
    if st.button("Analyze", type="primary") and txt.strip():
        from newsimpact.service import analyze_text
        out = analyze_text(txt, use_llm=use_llm)
        c = st.columns(4)
        c[0].metric("Overall tone", out["sentiment"], f"{out['score']:+.2f}")
        c[1].metric("Primary event", EVENT_LABEL.get(out["primary_event"] or "other_news"))
        c[2].metric("Sectors", ", ".join(out["sectors"]) or "–")
        c[3].metric("Backend", out["backend"])
        if out["impact"]:
            st.markdown("**Company-level impact**")
            st.dataframe(pd.DataFrame(out["impact"]), hide_index=True, use_container_width=True)
        else:
            st.info("No NIFTY 50 company recognised in the text." + (" Market-wide news detected." if out["market_wide"] else ""))
        with st.expander("Raw analysis JSON"):
            st.json(out)

# ----------------------------------------------------------------------------- model quality
with tabs[6]:
    ev = report("sentiment_eval.json", stamp)
    st.subheader("Sentiment model evaluation (held-out test sets)")
    if ev:
        rows = [{"test set": k, **{m: v[m] for m in ("n", "accuracy", "macro_f1", "majority_baseline_acc")}} for k, v in ev["doc_level"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        cm = ev["doc_level"]["sentfin_test (Indian headlines)"]["confusion_matrix(labels=neg,neu,pos)"]
        fig = px.imshow(cm, x=["neg", "neu", "pos"], y=["neg", "neu", "pos"], text_auto=True, color_continuous_scale="Blues",
                        labels=dict(x="predicted", y="actual"), title="Confusion matrix — Indian headlines (SEntFiN test)")
        fig.update_layout(height=380, width=420)
        st.plotly_chart(fig)
        el = ev.get("entity_level", {})
        if el:
            st.markdown(f"**Mixed-sentiment headlines** (several companies with different tones, n={el['n_entity_mentions']}): "
                        f"document-level accuracy {el['doc_level_accuracy']:.1%} → target-clause accuracy {el['target_clause_accuracy']:.1%}. "
                        "This is the case where an LLM backend helps most.")
    ee = (report("run_summary.json", stamp) or {}).get("entity_eval", {})
    if ee:
        st.markdown(f"**Entity linking:** the company a headline was collected for is recognised in "
                    f"{ee['query_company_found_in_headline']:.1%} of headlines (the rest mostly do not name it — search noise); "
                    f"{ee['avg_companies_per_article']} companies per tagged headline on average.")
