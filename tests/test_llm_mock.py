"""LLM path end-to-end with a mocked provider (no network, no key needed)."""
import json

import newsimpact.nlp.llm as llm
from newsimpact.config import settings
from newsimpact.nlp import analyzer


class _Resp:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_llm_enrichment_merges(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "anthropic_key", "test-key")
    monkeypatch.setattr(settings, "db_path", tmp_path / "x.db")
    monkeypatch.setattr(llm, "_client", None)
    monkeypatch.setattr(llm, "_client_failed", False)
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(url)
        out = {"items": [{"id": 0, "companies": [{"symbol": "TCS", "sentiment": "negative", "score": -0.7, "relevance": 0.9, "confidence": 0.9}],
                          "sectors": ["IT"], "events": [{"type": "guidance", "direction": -1}], "surprise": "miss",
                          "overall_sentiment": "negative", "summary": "TCS cuts outlook"}]}
        return _Resp({"content": [{"type": "text", "text": __import__("json").dumps(out)}]})

    monkeypatch.setattr(llm.requests, "post", fake_post)
    a = analyzer.analyze("TCS trims FY27 outlook as clients delay spending", use_llm=True)
    assert a.backend == "llm" and a.entities[0].symbol == "TCS" and a.entities[0].signal_label == "negative"
    assert a.events[0]["type"] == "earnings_surprise"
    # second call is served from cache
    analyzer.analyze("TCS trims FY27 outlook as clients delay spending", use_llm=True)
    assert len(calls) == 1 and "anthropic" in calls[0]


def test_llm_failure_falls_back(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "openai_key", "k")
    monkeypatch.setattr(settings, "anthropic_key", "")
    monkeypatch.setattr(settings, "db_path", tmp_path / "y.db")
    monkeypatch.setattr(llm, "_client", None)
    monkeypatch.setattr(llm, "_client_failed", False)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)

    def boom(*a, **k):
        raise llm.requests.ConnectionError("offline")

    monkeypatch.setattr(llm.requests, "post", boom)
    a = analyzer.analyze("Infosys beats estimates", use_llm=True)
    assert a.backend == "local" and a.entities[0].symbol == "INFY"
