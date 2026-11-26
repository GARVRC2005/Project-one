"""Build the bundled offline snapshot (data/snapshot/) from a recorded public dataset.

The snapshot lets the whole system (pipeline, event study, dashboard) run with no network:
  * headlines.csv   - 17k real NIFTY 50 headlines (Jul-Nov 2026) collected from Google News RSS
  * prices/*.csv    - 1 year of daily OHLCV for the 50 stocks + NIFTY 50 index (Yahoo chart API)
Source: NiftyLens recorded data (MIT, https://github.com/Tejdeep-67/niftylens).

With internet access you do not need this: `python -m newsimpact.cli fetch` pulls live RSS + yfinance.
Usage: python scripts/build_snapshot.py <path-to-niftylens-checkout>
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "snapshot"

def main(src: str) -> None:
    src = Path(src)
    (OUT / "prices").mkdir(parents=True, exist_ok=True)
    h = pd.read_csv(src / "analysis" / "data" / "headlines.csv")
    h = h.rename(columns={"symbol": "query_symbol"})
    h["url"] = ""
    h[["published", "source", "title", "query_symbol", "url"]].to_csv(OUT / "headlines.csv", index=False)
    n = 0
    for f in sorted((src / "public" / "sample" / "charts").glob("*-daily.json")):
        sym = f.name[: -len("-daily.json")]
        sym = {"MandM": "M&M"}.get(sym, sym)
        j = json.loads(f.read_text())
        df = pd.DataFrame(j["c"], columns=["ts", "open", "high", "low", "close", "volume"])
        df["date"] = (pd.to_datetime(df.ts, unit="ms", utc=True).dt.tz_convert("Asia/Kolkata").dt.date)
        df = df.drop(columns="ts").drop_duplicates("date", keep="last")[["date", "open", "high", "low", "close", "volume"]]
        df.to_csv(OUT / "prices" / f"{sym.replace('&', '_and_')}.csv", index=False)
        n += 1
    print(f"headlines: {len(h)}, price files: {n}")

if __name__ == "__main__":
    main(sys.argv[1])
