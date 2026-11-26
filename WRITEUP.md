# Write-up: AI-Based News Sentiment & Stock Impact Analyzer

**Problem.** Investors cannot read thousands of daily headlines and filings, or tell which stocks they affect
and by how much. The goal was a proof of concept that reads Indian market news, finds the companies, sectors
and events in it, scores sentiment, learns from history how such news moved prices, and raises ranked,
explainable stock-wise alerts in a searchable dashboard.

## Approach

1. **Ingestion.** Google News RSS (one query per NIFTY 50 company), ET / Moneycontrol / Mint / Business Standard
   market feeds, NSE corporate announcements, or any CSV/JSON. Titles are cleaned, exact duplicates removed,
   and near-duplicates clustered so that one story carried by 10 outlets becomes one alert with higher confidence.
2. **Entity & sector recognition.** A curated alias dictionary for the 50 companies: legal names, short names,
   tickers, brands and key executives, each with a link confidence. A blocklist of look-alike entities stops
   the classic broker-news errors: "HDFC Securities recommends …" is not about HDFC Bank, and "Reliance Power" is
   not Reliance Industries. Sectors come from the company plus keyword rules, and macro news is flagged separately.
3. **Event extraction.** A 17-type financial event taxonomy with ~50 rules, each carrying a directional prior.
   Rules are transparent, fast and testable. The LLM returns the same taxonomy when enabled.
4. **Sentiment.** A layered design with automatic fallback:
   LLM (Claude / GPT / Gemini, entity-level, cached, validated) → FinBERT (optional) → a bundled TF-IDF +
   logistic-regression model trained on 9.4k labelled financial sentences (incl. Indian headlines) → lexicon.
   Sentiment is scored per company on that company's own clause, then blended (70/30) with the event prior
   into a `signal`.
5. **Historical impact.** A standard market-model event study (β vs NIFTY 50 over days −120…−11). It reports
   abnormal returns and CAR over day 0, −1…+1, +1…+3, +1…+5 and the pre-news window −5…−1, plus abnormal volume.
   Results are aggregated by event type × tone with t-stats and hit rates.
6. **Alerts.** priority = relevance × confidence × historical expected move × tone strength. Expected move is
   shrunk toward the tone average when an event bucket is small. Each alert carries a one-line explanation,
   e.g. *"Order win with positive tone; historically +1.3% abnormal over day −1..+1 (n=84)"*.
7. **Delivery.** SQLite with FTS5 search, a FastAPI service and a 7-tab Streamlit dashboard: overview,
   alerts, news search, stock view with news on the price chart, impact study, analyze-any-text, model quality.

## Key findings (15,939 headlines, 50 stocks, Jul–Nov 2026)

- **Sentiment model:** 84.3% accuracy / 0.84 macro-F1 on held-out Indian headlines (baseline 35.9%), and
  89.4% on Financial PhraseBank.
- **Prices react on the day:** positive news +0.32% and negative news −0.62% abnormal return over day −1..+1
  (t = 6.9 and −8.1). Negative news hits about twice as hard. Order wins (+1.3%), earnings beats (+1.1%) and
  weak monthly sales (−2.9%) are the most price-relevant event types.
- **But not the next day:** the correlation of news tone with next-session excess return is −0.01 (p = 0.63),
  and a naive long/short strategy does not beat costs. Public headlines are priced within the session.
  The product's value is speed, coverage and prioritisation, not a trading signal. Saying this plainly is
  part of the result.
- **The alert ranking works out of sample:** with the impact table learned only on the first 60% of days,
  HIGH alerts were followed by an average 3.6% absolute abnormal move against 2.1% for LOW, with a 60% vs 55%
  direction hit rate.

## Robustness choices

Offline-first (bundled data + model, automatic retraining if the model file is incompatible). Every
network source fails independently and falls back to the snapshot. The LLM layer has a cache, batching,
retries, schema validation, a cost cap and per-item fallback. Timestamps are converted to IST and mapped to
the trading session (news after 15:30 goes to the next day). Articles are deduplicated into events before
any statistics. Returns that look like corporate actions are removed. Small samples are shrunk. 26 automated
tests cover entity linking, events, target-aware sentiment, LLM validation and fallback (mocked), the
trading calendar, event-study maths (recovering an injected abnormal return) and search.

## Limitations / next steps

NIFTY 50 universe and headline-level text for now. Next: article bodies and filing PDFs summarised by the LLM,
the full NSE universe, intraday prices for minute-level impact, and per-user watchlist push alerts
integrated into Arihant Plus through the existing API.
