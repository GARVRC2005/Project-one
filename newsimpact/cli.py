"""Command line entry point.

  python -m newsimpact.cli build                 # offline: bundled snapshot -> DB, impact study, alerts
  python -m newsimpact.cli fetch --days 7        # live: RSS + NSE + yfinance prices (needs internet)
  python -m newsimpact.cli ingest my_news.csv    # your own news file (title, published[, source])
  python -m newsimpact.cli analyze "Infosys beats estimates, raises guidance"
  python -m newsimpact.cli alerts --top 20
  python -m newsimpact.cli train                 # retrain + evaluate the local sentiment model
"""
from __future__ import annotations

import argparse
import json
import logging
import sys


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="newsimpact")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--no-llm", action="store_true")
    f = sub.add_parser("fetch"); f.add_argument("--days", type=int, default=7); f.add_argument("--no-llm", action="store_true")
    f.add_argument("--append", action="store_true", help="add to the existing DB instead of rebuilding")
    i = sub.add_parser("ingest"); i.add_argument("path"); i.add_argument("--no-llm", action="store_true")
    a = sub.add_parser("analyze"); a.add_argument("text"); a.add_argument("--no-llm", action="store_true")
    al = sub.add_parser("alerts"); al.add_argument("--top", type=int, default=20); al.add_argument("--symbol")
    sub.add_parser("train")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    from . import pipeline
    if args.cmd == "build":
        _print_report(pipeline.run("snapshot", use_llm=False if args.no_llm else None))
    elif args.cmd == "fetch":
        from .ingest.prices import download
        try:
            ok = download()
            print(f"prices refreshed for {len(ok)} symbols")
        except Exception as e:
            print(f"price download failed ({e}); using cached/snapshot prices", file=sys.stderr)
        _print_report(pipeline.run("live", reset=not args.append, days=args.days, use_llm=False if args.no_llm else None))
    elif args.cmd == "ingest":
        _print_report(pipeline.run(args.path, reset=False, use_llm=False if args.no_llm else None))
    elif args.cmd == "analyze":
        from .service import analyze_text
        print(json.dumps(analyze_text(args.text, use_llm=False if args.no_llm else None), indent=2, default=str))
    elif args.cmd == "alerts":
        from .store import Store
        df = Store().alerts()
        if args.symbol:
            df = df[df.symbol == args.symbol.upper()]
        cols = ["severity", "priority", "symbol", "event", "sentiment", "relevance", "confidence", "expected_move_pct", "published_ist", "title"]
        import pandas as pd
        with pd.option_context("display.max_colwidth", 80, "display.width", 250):
            print(df[cols].head(args.top).to_string(index=False))
    elif args.cmd == "train":
        from .nlp.train import train_and_save
        train_and_save(evaluate=True)


def _print_report(r: dict) -> None:
    keep = {k: v for k, v in r.items() if k not in ("impact_by_tone",)}
    print(json.dumps(keep, indent=2, default=str))


if __name__ == "__main__":
    main()
