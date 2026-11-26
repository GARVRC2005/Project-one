"""Sentiment backends with automatic fallback.

    llm      -> external LLM (Anthropic / OpenAI / Gemini) via newsimpact.nlp.llm (entity-level, best quality)
    finbert  -> ProsusAI/finbert through `transformers` (optional dependency)
    local    -> TF-IDF + calibrated logistic regression trained on Indian headlines (SEntFiN)
                + Financial PhraseBank; ships with the repo, CPU-only, ~1 ms / headline
    lexicon  -> tiny finance lexicon, last-resort fallback that can never fail

All backends return the same `SentimentResult`; `score` = P(positive) - P(negative) in [-1, 1].
The local model also supports *target-aware* scoring: for a headline mentioning several
companies, only the clause that mentions the target company is scored
("Kotak Bank profit falls; Tech Mahindra rises" -> Kotak negative, TechM positive).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from ..config import settings

log = logging.getLogger(__name__)
LABELS = ("negative", "neutral", "positive")
MODEL_FILE = "sentiment_tfidf_lr.joblib"


@dataclass
class SentimentResult:
    label: str
    score: float  # p_pos - p_neg
    confidence: float  # probability of the predicted label
    probs: dict = field(default_factory=dict)
    backend: str = "local"

    @classmethod
    def from_probs(cls, p_neg: float, p_neu: float, p_pos: float, backend: str) -> "SentimentResult":
        probs = {"negative": float(p_neg), "neutral": float(p_neu), "positive": float(p_pos)}
        label = max(probs, key=probs.get)
        return cls(label, round(p_pos - p_neg, 4), round(probs[label], 4), probs, backend)


def normalize(text: str) -> str:
    t = text.lower()
    t = re.sub(r"https?://\S+", " ", t)
    t = t.replace("₹", " rs ").replace("%", " pct ")
    t = re.sub(r"\brs\.?\s*", " rs ", t)
    t = re.sub(r"\d+(?:[.,]\d+)*", " num ", t)
    t = re.sub(r"[^a-z&' ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# ---------------------------------------------------------------- local model
class LocalModel:
    name = "local"

    def __init__(self, path: Path | None = None):
        self.path = path or settings.model_dir / MODEL_FILE
        self.pipe = None
        self._load()

    def _load(self) -> None:
        import joblib
        try:
            obj = joblib.load(self.path)
            self.pipe = obj["pipeline"]
            self.classes = list(self.pipe.classes_)
        except Exception as e:  # missing file or scikit-learn version mismatch -> retrain from bundled data
            log.warning("local sentiment model unavailable (%s); retraining from bundled data", e)
            from .train import train_and_save
            train_and_save(self.path, quiet=True)
            obj = joblib.load(self.path)
            self.pipe = obj["pipeline"]
            self.classes = list(self.pipe.classes_)

    def predict(self, texts: list[str]) -> list[SentimentResult]:
        if not texts:
            return []
        P = self.pipe.predict_proba([normalize(t) for t in texts])
        idx = {c: i for i, c in enumerate(self.classes)}
        return [SentimentResult.from_probs(p[idx["negative"]], p[idx["neutral"]], p[idx["positive"]], self.name) for p in P]


# ---------------------------------------------------------------- FinBERT
class FinBertModel:
    name = "finbert"

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        from transformers import pipeline  # optional dependency
        self.clf = pipeline("text-classification", model=model_name, top_k=None, truncation=True)

    def predict(self, texts: list[str]) -> list[SentimentResult]:
        out = []
        for scores in self.clf(list(texts), batch_size=32):
            d = {s["label"].lower(): s["score"] for s in scores}
            out.append(SentimentResult.from_probs(d.get("negative", 0), d.get("neutral", 0), d.get("positive", 0), self.name))
        return out


# ---------------------------------------------------------------- lexicon fallback
_POS = set("""beat beats surge surges surged jump jumps jumped soar soars rally rallies gain gains gained rise rises rose
record high upgrade upgraded outperform buy strong robust growth grows grew profit profits wins win won bags secures
approval approved nod expansion boost boosts boosted positive optimistic bullish higher up improves improved recovery
dividend buyback bonus order orders acquires launch launches top best""".split())
_NEG = set("""miss misses missed fall falls fell drop drops dropped decline declines declined slump slumps plunge plunges
crash crashes tank tanks low weak weaker loss losses downgrade downgraded underperform sell cut cuts penalty fine fined
probe raid fraud resign resigns resigned quits exit concern concerns risk risks pressure lower down bearish negative
default delay delays strike fire outage ban banned warning recall halts slowdown worst slips slip""".split())


class LexiconModel:
    name = "lexicon"

    def predict(self, texts: list[str]) -> list[SentimentResult]:
        out = []
        for t in texts:
            toks = normalize(t).split()
            p = sum(w in _POS for w in toks)
            n = sum(w in _NEG for w in toks)
            if p == n:
                out.append(SentimentResult.from_probs(0.2, 0.6, 0.2, self.name))
            else:
                s = (p - n) / (p + n)
                pp, pn = max(0.0, s) * 0.7 + 0.15, max(0.0, -s) * 0.7 + 0.15
                out.append(SentimentResult.from_probs(pn, max(0.0, 1 - pp - pn), pp, self.name))
        return out


# ---------------------------------------------------------------- target-aware helper
_CLAUSE_SPLIT = re.compile(r"\s*(?:;|\||:(?=\s)|\bwhile\b|\bbut\b|\beven as\b|\bwhereas\b|\bhowever\b|,\s+(?=[A-Z]))\s*")


def target_clause(text: str, start: int, end: int) -> str:
    """Return the clause of `text` containing span [start, end) (falls back to the full text)."""
    pieces, pos = [], 0
    for m in _CLAUSE_SPLIT.finditer(text):
        pieces.append((pos, m.start()))
        pos = m.end()
    pieces.append((pos, len(text)))
    for s, e in pieces:
        if s <= start < max(e, s + 1):
            clause = text[s:e].strip()
            return clause if len(clause.split()) >= 4 else text
    return text


# ---------------------------------------------------------------- factory
@lru_cache(maxsize=4)
def get_model(backend: str | None = None):
    """Return a classifier object with `.predict(list[str])`; falls back down the chain on failure."""
    backend = (backend or settings.sentiment_backend or "auto").lower()
    chain = {"auto": ["local", "lexicon"], "llm": ["local", "lexicon"], "finbert": ["finbert", "local", "lexicon"],
             "local": ["local", "lexicon"], "lexicon": ["lexicon"]}.get(backend, ["local", "lexicon"])
    if backend == "auto":
        try:
            import transformers  # noqa: F401
            if settings.__dict__.get("prefer_finbert") or _env_true("NIA_USE_FINBERT"):
                chain = ["finbert"] + chain
        except Exception:
            pass
    for name in chain:
        try:
            return {"finbert": FinBertModel, "local": LocalModel, "lexicon": LexiconModel}[name]()
        except Exception as e:  # pragma: no cover - depends on environment
            log.warning("sentiment backend %s failed to load: %s", name, e)
    return LexiconModel()


def _env_true(name: str) -> bool:
    import os
    return os.environ.get(name, "").lower() in ("1", "true", "yes")
