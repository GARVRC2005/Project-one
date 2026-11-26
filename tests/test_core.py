import numpy as np
import pandas as pd
import pytest

from newsimpact.nlp.entities import get_universe
from newsimpact.nlp.events import extract_events, is_listicle
from newsimpact.nlp.analyzer import analyze
from newsimpact.nlp.llm import validate
from newsimpact.nlp.sentiment import target_clause
from newsimpact.impact import event_study as es
from newsimpact.store import Store, _fts_query


@pytest.mark.parametrize("text,expected", [
    ("HDFC Securities recommends buy on Reliance Industries", {"RELIANCE"}),
    ("Reliance Power shares jump 5%", set()),
    ("Kotak Mahindra Bank Q2 profit falls; Tech Mahindra rises", {"KOTAKBANK", "TECHM"}),
    ("M&M Finance up 3%; Mahindra & Mahindra SUV sales jump", {"M&M"}),
    ("SBI Life and SBI Card shares fall, SBI steady", {"SBILIFE", "SBIN"}),
    ("Sensex falls 500 pts on BSE", set()),
    ("BSE shares surge 8% after Q2 results", {"BSE"}),
    ("ITC Hotels demerger: what ITC shareholders get", {"ITC"}),
    ("Indigo Paints rallies 4%", set()),
])
def test_entity_linking(text, expected):
    assert set(get_universe().companies_in(text)) == expected


@pytest.mark.parametrize("text,event,direction", [
    ("Infosys beats Street estimates, raises guidance", "earnings_surprise", 1),
    ("Sun Pharma Q2 net profit falls 36%", "earnings", -1),
    ("L&T bags orders worth Rs 5,000 crore", "order_win", 1),
    ("SEBI imposes penalty on Adani Enterprises", "regulatory_legal", -1),
    ("HDFC Bank CFO resigns", "management_change", -1),
    ("Jefferies upgrades Titan to buy, raises target price", "rating_change", 1),
    ("Promoter sells 2% stake in block deal", "stake_sale", -1),
    ("Shriram Finance raises Rs 2,000 crore via NCDs", "capital_raise", 0),
])
def test_events(text, event, direction):
    hits = extract_events(text)
    assert hits and hits[0].event == event and hits[0].direction == direction


def test_capital_raise_not_triggered_by_target():
    assert "capital_raise" not in [h.event for h in extract_events("Jefferies raises target price to Rs 4,200")]


def test_listicle():
    assert is_listicle("Stocks to watch: HDFC Bank, BSE, HCL Tech and more", 3)


def test_target_clause_and_entity_sentiment():
    t = "Kotak Mahindra Bank Q2 profit falls 10%; Tech Mahindra shares jump 5% on deal win"
    assert target_clause(t, t.index("Tech"), t.index("Tech") + 13).startswith("Tech Mahindra")
    a = analyze(t, use_llm=False)
    tones = {e.symbol: e.signal_label for e in a.entities}
    assert tones == {"KOTAKBANK": "negative", "TECHM": "positive"}


def test_llm_output_validation():
    v = validate({"companies": [{"symbol": "INFY.NS", "sentiment": "POSITIVE", "score": 3, "relevance": "x"},
                                {"symbol": "NOTREAL", "sentiment": "positive"}],
                  "events": [{"type": "earnings", "direction": 5}, {"type": "made_up"}], "overall_sentiment": "great"})
    assert [c["symbol"] for c in v["companies"]] == ["INFY"]
    assert v["companies"][0]["score"] == 1.0 and v["companies"][0]["relevance"] == 0.5
    assert v["events"] == [{"type": "earnings", "direction": 1}] and v["overall_sentiment"] == "neutral"
    assert validate(None) is None


def test_trading_day_mapping():
    cal = pd.DatetimeIndex(["2026-08-13", "2026-08-14", "2026-08-17"])  # Fri -> Mon
    pub = pd.Series(["2026-08-13T05:00:00+00:00",   # 10:30 IST Thu -> same day
                     "2026-08-13T10:30:00+00:00",   # 16:00 IST Thu -> Fri
                     "2026-08-14T11:00:00+00:00",   # 16:30 IST Fri -> Mon
                     "2026-08-15T07:00:00+00:00",   # Sat, date-only -> Mon
                     "2026-08-20T05:00:00+00:00"])  # beyond price history -> NaN
    m = es.map_trading_day(pub, cal)
    assert m.trading_day.tolist()[:4] == ["2026-08-13", "2026-08-14", "2026-08-17", "2026-08-17"]
    assert pd.isna(m.trading_day.iloc[4]) and m.time_precise.tolist()[3] == 0


def test_event_study_recovers_injected_abnormal_return():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2025-01-01", periods=200)
    rm = rng.normal(0, 0.01, 200)
    ri = 0.0002 + 1.2 * rm + rng.normal(0, 0.002, 200)
    ri[150] += 0.05  # +5% abnormal jump on the event day
    close = pd.DataFrame({"NIFTY50": 100 * np.exp(np.cumsum(rm)), "X": 100 * np.exp(np.cumsum(ri))}, index=idx)
    ev = pd.DataFrame([{"symbol": "X", "trading_day": idx[150].strftime("%Y-%m-%d"), "event": "e", "signal_label": "positive",
                        "n_articles": 1, "n_sources": 1, "signal": 0.9, "relevance": 1, "example": ""}])
    car = es.compute_car(ev, close)
    assert car.method.iloc[0] == "market_model"
    assert abs(car.beta.iloc[0] - 1.2) < 0.05
    assert abs(car["ar_0"].iloc[0] - 0.05) < 0.01
    assert abs(car["ar_1"].iloc[0]) < 0.01


def test_store_search(tmp_path):
    s = Store(tmp_path / "t.db")
    arts = pd.DataFrame([{"id": "a1", "title": "Cipla gets USFDA nod for generic", "source": "ET", "published_ist": "2026-08-01 10:00",
                          "summary": ""}])
    ents = pd.DataFrame([{"article_id": "a1", "symbol": "CIPLA", "sector": "Pharma", "signal_label": "positive", "event": "regulatory_approval"}])
    s.upsert_articles(arts, ents)
    assert s.search("usfda").symbol.tolist() == ["CIPLA"]
    assert s.search('weird "( query').empty
    assert len(s.search(symbol="CIPLA", sentiment="positive")) == 1
    assert _fts_query("block deal") == '"block" "deal"*'
