"""Optional LLM enrichment (Anthropic Claude / OpenAI / Google Gemini) over plain HTTPS.

One call analyses a batch of headlines and returns, per headline, entity-level sentiment,
events and a short rationale in a strict JSON schema. Robustness measures:
  * provider picked from whichever API key is set (or NIA_LLM_PROVIDER),
  * results cached in SQLite keyed by (model, prompt version, text) -> re-runs are free,
  * retries with exponential backoff on 429 / 5xx / timeouts,
  * output validated: unknown symbols / event types / labels are dropped, numbers clamped,
  * any failure returns None for that item so the caller falls back to the local pipeline.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
import time

import requests

from ..config import settings
from .entities import get_universe
from .events import EVENT_TYPES

log = logging.getLogger(__name__)
PROMPT_VERSION = "v3"
_lock = threading.Lock()


def _system_prompt() -> str:
    u = get_universe()
    comp = "; ".join(f"{c.symbol}={c.name}" for c in u.companies.values())
    events = "; ".join(f"{k}: {v}" for k, v in EVENT_TYPES.items())
    return f"""You are a sell-side equity analyst covering Indian equities (NSE).
For every news item, identify which of the covered companies it is materially ABOUT, the sentiment of the
news FOR EACH such company's shareholders, and the financial events it reports.

Covered companies (use these symbols only): {comp}

Event types (use these keys only): {events}

Rules:
- Only list a company if the item is about it (not if it is merely the broker/source, e.g. "HDFC Securities recommends ...").
- sentiment is from that company's shareholders' perspective: positive | neutral | negative; score in [-1, 1].
- relevance in [0, 1]: 1 = the company is the main subject; 0.3 = one of many names in a round-up.
- confidence in [0, 1]: how sure you are of the sentiment label.
- For earnings, set surprise to "beat", "miss", "inline" or null.
- Prefer neutral for previews, schedules and generic price commentary without a cause.
Return ONLY JSON: {{"items": [{{"id": <id>, "companies": [{{"symbol": str, "sentiment": str, "score": float,
"relevance": float, "confidence": float}}], "sectors": [str], "events": [{{"type": str, "direction": -1|0|1}}],
"surprise": str|null, "overall_sentiment": str, "summary": str (<= 20 words)}}]}}"""


class LLMClient:
    def __init__(self):
        cfg = settings.resolved_llm()
        if not cfg:
            raise RuntimeError("no LLM API key configured")
        self.provider, self.model, self.key = cfg
        self.db = sqlite3.connect(settings.db_path.parent / "llm_cache.sqlite", check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT, ts REAL)")
        self.system = _system_prompt()
        self.calls = 0

    # ------------------------------------------------------------------ cache
    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.provider}|{self.model}|{PROMPT_VERSION}|{text}".encode()).hexdigest()

    def _get(self, text: str):
        with _lock:
            r = self.db.execute("SELECT v FROM cache WHERE k=?", (self._key(text),)).fetchone()
        return json.loads(r[0]) if r else None

    def _put(self, text: str, value: dict) -> None:
        with _lock:
            self.db.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?)", (self._key(text), json.dumps(value), time.time()))
            self.db.commit()

    # ------------------------------------------------------------------ transport
    def _post(self, user: str) -> str:
        if self.provider == "anthropic":
            url = "https://api.anthropic.com/v1/messages"
            headers = {"x-api-key": self.key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
            body = {"model": self.model, "max_tokens": 4096, "temperature": 0, "system": self.system,
                    "messages": [{"role": "user", "content": user}]}
            parse = lambda j: "".join(b.get("text", "") for b in j["content"])  # noqa: E731
        elif self.provider == "openai":
            url = "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {self.key}", "content-type": "application/json"}
            body = {"model": self.model, "temperature": 0, "response_format": {"type": "json_object"},
                    "messages": [{"role": "system", "content": self.system}, {"role": "user", "content": user}]}
            parse = lambda j: j["choices"][0]["message"]["content"]  # noqa: E731
        elif self.provider == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            headers = {"x-goog-api-key": self.key, "content-type": "application/json"}
            body = {"systemInstruction": {"parts": [{"text": self.system}]},
                    "contents": [{"role": "user", "parts": [{"text": user}]}],
                    "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}
            parse = lambda j: j["candidates"][0]["content"]["parts"][0]["text"]  # noqa: E731
        else:
            raise ValueError(self.provider)

        delay = 2.0
        for attempt in range(4):
            try:
                r = requests.post(url, headers=headers, json=body, timeout=settings.llm_timeout)
                if r.status_code in (429, 500, 502, 503, 504, 529):
                    raise requests.HTTPError(f"retryable {r.status_code}")
                r.raise_for_status()
                self.calls += 1
                return parse(r.json())
            except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
                if attempt == 3 or (isinstance(e, requests.HTTPError) and "retryable" not in str(e)):
                    raise
                time.sleep(delay)
                delay *= 2
        raise RuntimeError("unreachable")

    # ------------------------------------------------------------------ public
    def analyze(self, texts: list[str]) -> list[dict | None]:
        out: list[dict | None] = [self._get(t) for t in texts]
        todo = [i for i, v in enumerate(out) if v is None]
        bs = max(1, settings.llm_batch_size)
        for s in range(0, len(todo), bs):
            idx = todo[s:s + bs]
            user = "News items:\n" + "\n".join(json.dumps({"id": k, "text": texts[i]}, ensure_ascii=False)
                                               for k, i in enumerate(idx))
            try:
                raw = self._post(user)
                items = _parse_json(raw).get("items", [])
                by_id = {int(it.get("id", -1)): it for it in items if isinstance(it, dict)}
                for k, i in enumerate(idx):
                    v = validate(by_id.get(k))
                    if v is not None:
                        self._put(texts[i], v)
                        out[i] = v
            except Exception as e:
                log.warning("LLM batch failed (%s); falling back to local models for %d items", e, len(idx))
        return out


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    m = re.search(r"\{.*\}", raw, re.S)
    return json.loads(m.group(0) if m else raw)


def _clamp(x, lo, hi, default):
    try:
        return max(lo, min(hi, float(x)))
    except (TypeError, ValueError):
        return default


def validate(item: dict | None) -> dict | None:
    if not isinstance(item, dict):
        return None
    u = get_universe()
    comps = []
    for c in item.get("companies") or []:
        sym = str(c.get("symbol", "")).upper().replace(".NS", "")
        if sym not in u.companies:
            continue
        lab = str(c.get("sentiment", "neutral")).lower()
        if lab not in ("positive", "neutral", "negative"):
            lab = "neutral"
        comps.append({"symbol": sym, "sentiment": lab, "score": _clamp(c.get("score"), -1, 1, 0.0),
                      "relevance": _clamp(c.get("relevance"), 0, 1, 0.5), "confidence": _clamp(c.get("confidence"), 0, 1, 0.6)})
    events = []
    for e in item.get("events") or []:
        t = e.get("type") if isinstance(e, dict) else e
        if t in EVENT_TYPES:
            events.append({"type": t, "direction": int(_clamp(e.get("direction", 0) if isinstance(e, dict) else 0, -1, 1, 0))})
    ov = str(item.get("overall_sentiment", "neutral")).lower()
    surprise = item.get("surprise")
    return {"companies": comps, "events": events,
            "sectors": [s for s in (item.get("sectors") or []) if isinstance(s, str)][:5],
            "surprise": surprise if surprise in ("beat", "miss", "inline") else None,
            "overall_sentiment": ov if ov in ("positive", "neutral", "negative") else "neutral",
            "summary": str(item.get("summary", ""))[:200]}


_client: LLMClient | None = None
_client_failed = False


def get_client() -> LLMClient | None:
    global _client, _client_failed
    if _client is None and not _client_failed:
        try:
            _client = LLMClient()
        except Exception as e:
            log.info("LLM disabled: %s", e)
            _client_failed = True
    return _client
