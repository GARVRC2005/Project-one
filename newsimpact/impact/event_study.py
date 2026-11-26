"""Historical news -> price impact study.

Methodology (standard short-horizon event study, MacKinlay 1997):
  1. Each article is mapped to the first trading session in which it could move the price
     (published after the 15:30 IST close or on a holiday -> next session).
  2. Articles are collapsed into *events* = (stock, trading day, event type, tone) so that 20 articles
     about one result announcement count once.
  3. Market model per event: r_i = a + b * r_NIFTY + e, fitted by OLS on trading days [-120, -11]
     (>= 60 observations, otherwise market-adjusted returns with b = 1).
  4. Abnormal returns AR_t for t in [-5, +5]; cumulative abnormal returns (CAR) over several windows.
     [-1, +1] is the headline window because many RSS timestamps are date-only.
     [+1, +3] / [+1, +5] measure post-news drift, i.e. what was still tradable after the news.
  5. Cross-sectional mean CAR, t-statistic, hit rate (share of events whose CAR sign matches the tone)
     per (event type x tone), plus abnormal volume (volume / median of the previous 20 sessions).
  6. A simple, honest long/short backtest of the daily news signal (enter at the close of the news day).
Returns with |r| > 25% in a day are treated as data errors / corporate actions and dropped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import settings

MAX_ABS_RET = 0.25
PRE, POST = 5, 5


def trading_calendar(close: pd.DataFrame) -> pd.DatetimeIndex:
    bench = settings.benchmark_symbol
    idx = close[bench].dropna().index if bench in close else close.dropna(how="all").index
    return pd.DatetimeIndex(sorted(idx))


def map_trading_day(published_utc: pd.Series, cal: pd.DatetimeIndex) -> pd.DataFrame:
    """-> DataFrame[published_ist, time_precise, trading_day] (trading_day NaT if beyond price history)."""
    ts = pd.to_datetime(published_utc, utc=True, errors="coerce", format="mixed")
    ist = ts.dt.tz_convert("Asia/Kolkata")
    # Google News gives date-only items stamped 07:00:00 UTC; treat those times as unknown.
    precise = ~((ts.dt.hour == 7) & (ts.dt.minute == 0) & (ts.dt.second == 0))
    day = ist.dt.tz_localize(None).dt.normalize()
    h, m = settings.market_close_ist
    after_close = (ist.dt.hour * 60 + ist.dt.minute) >= (h * 60 + m)
    after_close &= precise
    cand = day + pd.to_timedelta(after_close.astype(int), unit="D")
    cal_arr = cal.values
    pos = np.searchsorted(cal_arr, cand.values.astype("datetime64[ns]"), side="left")
    td = pd.Series(pd.NaT, index=published_utc.index, dtype="datetime64[ns]")
    ok = (pos < len(cal_arr)) & cand.notna().values
    td[ok] = cal_arr[pos[ok]]
    return pd.DataFrame({"published_ist": ist.dt.strftime("%Y-%m-%d %H:%M"), "time_precise": precise.astype(int),
                         "trading_day": td.dt.strftime("%Y-%m-%d")})


def log_returns(close: pd.DataFrame) -> pd.DataFrame:
    r = np.log(close / close.shift(1))
    return r.mask(r.abs() > MAX_ABS_RET)


def build_events(arts: pd.DataFrame, ents: pd.DataFrame, min_relevance: float = 0.3) -> pd.DataFrame:
    """Collapse (article, entity) rows into unique stock-day-event-tone events."""
    df = ents.merge(arts[["id", "trading_day", "listicle", "cluster_id", "source", "title"]],
                    left_on="article_id", right_on="id")
    df = df[(df.relevance >= min_relevance) & (df.listicle == 0) & df.trading_day.notna()]
    df["event"] = df["event"].fillna("other_news").replace("", "other_news")
    g = df.groupby(["symbol", "trading_day", "event", "signal_label"])
    ev = g.agg(n_articles=("article_id", "size"), n_sources=("source", "nunique"), signal=("signal", "mean"),
               relevance=("relevance", "max"), example=("title", "first")).reset_index()
    return ev


def compute_car(events: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame | None = None) -> pd.DataFrame:
    rets = log_returns(close)
    bench = settings.benchmark_symbol
    rm = rets[bench]
    dates = rets.index
    pos_of = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(dates)}
    e0, e1 = settings.est_window
    out_rows = []
    beta_cache: dict[tuple[str, int], tuple[float, float, str]] = {}
    for r in events.itertuples(index=False):
        if r.symbol not in rets or r.trading_day not in pos_of:
            continue
        t0 = pos_of[r.trading_day]
        key = (r.symbol, t0)
        if key not in beta_cache:
            lo, hi = max(0, t0 + e0), t0 + e1
            ri, mi = rets[r.symbol].iloc[lo:hi + 1], rm.iloc[lo:hi + 1]
            m = ri.notna() & mi.notna()
            if m.sum() >= settings.est_min_obs:
                b, a = np.polyfit(mi[m].values, ri[m].values, 1)
                beta_cache[key] = (a, b, "market_model")
            else:
                beta_cache[key] = (0.0, 1.0, "market_adjusted")
        a, b, method = beta_cache[key]
        row = {**r._asdict(), "method": method, "beta": b}
        for k in range(-PRE, POST + 1):
            t = t0 + k
            if 0 <= t < len(dates):
                ri, mi = rets[r.symbol].iloc[t], rm.iloc[t]
                row[f"ar_{k}"] = ri - (a + b * mi) if pd.notna(ri) and pd.notna(mi) else np.nan
            else:
                row[f"ar_{k}"] = np.nan
        if volume is not None and r.symbol in volume:
            v = volume[r.symbol]
            base = v.iloc[max(0, t0 - 20):t0].median()
            row["abn_volume"] = float(v.iloc[t0] / base) if base and base > 0 and pd.notna(v.iloc[t0]) else np.nan
        out_rows.append(row)
    car = pd.DataFrame(out_rows)
    if car.empty:
        return car
    for lo, hi in settings.car_windows:
        cols = [f"ar_{k}" for k in range(lo, hi + 1)]
        car[f"car_{lo}_{hi}"] = car[cols].sum(axis=1, min_count=len(cols))
    return car


def _tstat(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 3 or x.std(ddof=1) == 0:
        return np.nan
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def summarize(car: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    if car.empty:
        return pd.DataFrame()
    sign = car.signal_label.map({"positive": 1, "negative": -1, "neutral": 0})
    car = car.assign(_sign=sign)
    rows = []
    for key, g in car.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key))
        row["n"] = len(g)
        for lo, hi in settings.car_windows:
            c = g[f"car_{lo}_{hi}"]
            row[f"mean_car_{lo}_{hi}_pct"] = round(100 * c.mean(), 3)
            row[f"t_{lo}_{hi}"] = round(_tstat(c), 2)
        c = g["car_-1_1"].dropna()
        row["median_car_-1_1_pct"] = round(100 * c.median(), 3) if len(c) else np.nan
        row["mean_abs_car_-1_1_pct"] = round(100 * c.abs().mean(), 3) if len(c) else np.nan
        s = g["_sign"].iloc[0] if "signal_label" in by else 0
        row["hit_rate_-1_1"] = round(float((np.sign(c) == s).mean()), 3) if s != 0 and len(c) else np.nan
        if "abn_volume" in g:
            row["median_abn_volume"] = round(float(g["abn_volume"].median()), 2)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def car_paths(car: pd.DataFrame, by: str = "signal_label") -> pd.DataFrame:
    """Average cumulative AR path from t=-5..+5 (for charts)."""
    cols = [f"ar_{k}" for k in range(-PRE, POST + 1)]
    out = []
    for key, g in car.groupby(by):
        m = g[cols].mean().fillna(0).cumsum() * 100
        for k, v in zip(range(-PRE, POST + 1), m.values):
            out.append({by: key, "day": k, "cum_ar_pct": round(float(v), 4), "n": len(g)})
    return pd.DataFrame(out)


def daily_signal_backtest(ents: pd.DataFrame, arts: pd.DataFrame, close: pd.DataFrame,
                          threshold: float = 0.2, cost_bps: float = 10.0) -> dict:
    """Long stocks with positive net news signal, short negative, enter at the close of the
    news session (all news up to that close is known), hold one session. Excess vs NIFTY 50."""
    df = ents.merge(arts[["id", "trading_day", "listicle", "time_precise"]], left_on="article_id", right_on="id")
    df = df[(df.listicle == 0) & df.trading_day.notna()]
    df["w"] = df.signal * df.relevance
    sig = df.groupby(["trading_day", "symbol"]).w.sum().unstack()
    sig.index = pd.to_datetime(sig.index)
    rets = np.exp(log_returns(close)) - 1
    bench = settings.benchmark_symbol
    excess = rets.sub(rets[bench], axis=0).drop(columns=[bench], errors="ignore")
    nxt = excess.shift(-1)  # return of the session after the news session
    sig = sig.reindex(columns=nxt.columns)
    sig = sig[sig.index.isin(nxt.index)]
    daily, longs, shorts = [], 0, 0
    for d, row in sig.iterrows():
        L = row[row > threshold].index
        S = row[row < -threshold].index
        rl = nxt.loc[d, L].dropna()
        rs = nxt.loc[d, S].dropna()
        if len(rl) == 0 and len(rs) == 0:
            continue
        leg_l = rl.mean() if len(rl) else 0.0
        leg_s = -rs.mean() if len(rs) else 0.0
        n_legs = (len(rl) > 0) + (len(rs) > 0)
        daily.append({"date": d, "ret": (leg_l + leg_s) / n_legs, "n_long": len(rl), "n_short": len(rs),
                      "long_ret": leg_l, "short_ret": leg_s})
        longs += len(rl)
        shorts += len(rs)
    bt = pd.DataFrame(daily)
    if bt.empty:
        return {"n_days": 0}
    bt["ret_net"] = bt.ret - 2 * cost_bps / 1e4
    ann = np.sqrt(252)
    res = {
        "n_days": int(len(bt)), "n_long_positions": int(longs), "n_short_positions": int(shorts),
        "mean_daily_excess_bps": round(1e4 * bt.ret.mean(), 2),
        "t_stat": round(_tstat(bt.ret), 2),
        "hit_rate": round(float((bt.ret > 0).mean()), 3),
        "sharpe_gross": round(float(bt.ret.mean() / bt.ret.std() * ann), 2) if bt.ret.std() > 0 else None,
        "sharpe_net_of_costs": round(float(bt.ret_net.mean() / bt.ret_net.std() * ann), 2) if bt.ret_net.std() > 0 else None,
        "cum_excess_gross_pct": round(100 * float((1 + bt.ret).prod() - 1), 2),
        "cum_excess_net_pct": round(100 * float((1 + bt.ret_net).prod() - 1), 2),
        "long_leg_mean_bps": round(1e4 * bt.long_ret[bt.n_long > 0].mean(), 2),
        "short_leg_mean_bps": round(1e4 * bt.short_ret[bt.n_short > 0].mean(), 2),
        "threshold": threshold, "cost_bps_per_side": cost_bps,
    }
    bt["equity_gross"] = (1 + bt.ret).cumprod()
    bt["equity_net"] = (1 + bt.ret_net).cumprod()
    res["curve"] = bt[["date", "equity_gross", "equity_net"]].assign(date=lambda x: x.date.dt.strftime("%Y-%m-%d")).to_dict("records")
    return res


def same_vs_next_day_correlation(ents: pd.DataFrame, arts: pd.DataFrame, close: pd.DataFrame) -> dict:
    """Correlation of the daily net news signal with same-session vs next-session excess returns."""
    from scipy import stats
    df = ents.merge(arts[["id", "trading_day", "listicle"]], left_on="article_id", right_on="id")
    df = df[(df.listicle == 0) & df.trading_day.notna()]
    sig = df.assign(w=df.signal * df.relevance).groupby(["trading_day", "symbol"]).w.mean()
    rets = np.exp(log_returns(close)) - 1
    ex = rets.sub(rets[settings.benchmark_symbol], axis=0)
    ex.index = ex.index.strftime("%Y-%m-%d")
    same = ex.stack()
    nxt = ex.shift(-1).stack()
    prev = ex.shift(1).stack()
    out = {}
    for name, s in (("previous_session", prev), ("same_session", same), ("next_session", nxt)):
        j = pd.concat([sig.rename("sig"), s.rename("ret")], axis=1, join="inner").dropna()
        if len(j) > 10:
            rho, p = stats.spearmanr(j.sig, j.ret)
            out[name] = {"n": int(len(j)), "spearman": round(float(rho), 4), "p_value": float(f"{p:.3g}")}
    return out


def impact_lookup(summary_event_tone: pd.DataFrame, summary_tone: pd.DataFrame, k: float | None = None):
    """Empirical-Bayes shrunk expected CAR[-1,+1] for (event, tone): small groups are pulled toward
    the tone-level mean, so a 3-event bucket cannot produce an extreme alert."""
    k = settings.shrinkage_k if k is None else k
    tone = summary_tone.set_index("signal_label") if len(summary_tone) else pd.DataFrame()

    def lookup(event: str | None, tone_label: str) -> dict:
        parent_mean = float(tone.loc[tone_label, "mean_car_-1_1_pct"]) if tone_label in tone.index else 0.0
        parent_abs = float(tone.loc[tone_label, "mean_abs_car_-1_1_pct"]) if tone_label in tone.index else 1.5
        parent_hit = tone.loc[tone_label, "hit_rate_-1_1"] if tone_label in tone.index else np.nan
        row = summary_event_tone[(summary_event_tone.event == event) & (summary_event_tone.signal_label == tone_label)]
        if len(row):
            r = row.iloc[0]
            n = int(r.n)
            w = n / (n + k)
            return {"expected_move_pct": round(w * r["mean_car_-1_1_pct"] + (1 - w) * parent_mean, 3),
                    "expected_abs_move_pct": round(w * r["mean_abs_car_-1_1_pct"] + (1 - w) * parent_abs, 3),
                    "hist_n": n, "hist_hit_rate": r.get("hit_rate_-1_1", np.nan), "t": r.get("t_-1_1", np.nan)}
        return {"expected_move_pct": round(parent_mean, 3), "expected_abs_move_pct": round(parent_abs, 3),
                "hist_n": 0, "hist_hit_rate": parent_hit, "t": np.nan}
    return lookup
