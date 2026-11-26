"""Daily prices: yfinance (live) with a local CSV cache, falling back to the bundled snapshot."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..config import settings
from ..nlp.entities import get_universe

log = logging.getLogger(__name__)
CACHE = settings.data_dir / "prices"


def _fname(sym: str) -> str:
    return sym.replace("&", "_and_") + ".csv"


def _read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df.set_index("date").sort_index()


def download(period: str = "2y") -> list[str]:
    """Refresh the cache from Yahoo Finance via yfinance. Returns the symbols that succeeded."""
    import yfinance as yf
    CACHE.mkdir(parents=True, exist_ok=True)
    u = get_universe()
    tickers = {c.yahoo: c.symbol for c in u.companies.values()}
    tickers[settings.benchmark_yahoo] = settings.benchmark_symbol
    ok = []
    data = yf.download(list(tickers), period=period, interval="1d", auto_adjust=True, group_by="ticker",
                       progress=False, threads=True)
    for yt, sym in tickers.items():
        try:
            d = data[yt].dropna(how="all")
            if d.empty:
                continue
            d = d.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
            d.index = pd.to_datetime(d.index).tz_localize(None).normalize()
            d.index.name = "date"
            d.to_csv(CACHE / _fname(sym))
            ok.append(sym)
        except Exception as e:
            log.warning("price download failed for %s: %s", sym, e)
    return ok


def load_prices(symbols: list[str] | None = None, prefer: str = "auto") -> dict[str, pd.DataFrame]:
    """{symbol: OHLCV frame}. Uses the live cache when present, else the snapshot."""
    u = get_universe()
    symbols = symbols or (list(u.companies) + [settings.benchmark_symbol])
    out = {}
    for s in symbols:
        df = None
        if prefer in ("auto", "live"):
            df = _read(CACHE / _fname(s))
        if df is None and prefer in ("auto", "snapshot"):
            df = _read(settings.snapshot_dir / "prices" / _fname(s))
        if df is not None and len(df):
            out[s] = df
    return out


def close_matrix(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame({s: d["close"] for s, d in prices.items()}).sort_index()


def volume_matrix(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.DataFrame({s: d["volume"] for s, d in prices.items()}).sort_index()
