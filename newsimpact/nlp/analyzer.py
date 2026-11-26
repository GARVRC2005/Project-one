"""Combine entity linking, event extraction and sentiment into one article analysis."""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

from ..config import settings
from .entities import get_universe
from .events import extract_events, is_listicle
from .sentiment import get_model, target_clause

log = logging.getLogger(__name__)

# Source credibility tiers: primary financial newsrooms > aggregators / social re-posts.
TIER1 = {"the economic times", "economic times", "livemint", "mint", "moneycontrol", "moneycontrol.com",
         "business standard", "cnbc tv18", "cnbctv18", "ndtv profit", "the hindu businessline", "businessline",
         "business today", "the financial express", "financial express", "reuters", "bloomberg", "zee business",
         "nse", "bse", "exchange filing"}
TIER3 = {"facebook", "x", "twitter", "youtube", "instagram", "scanx.trade", "marketscreener"}


def source_weight(source: str | None) -> float:
    s = (source or "").strip().lower()
    if s in TIER1:
        return 1.0
    if s in TIER3:
        return 0.7
    return 0.85


@dataclass
class EntityAnalysis:
    symbol: str
    sector: str
    sentiment: str
    score: float
    sentiment_confidence: float
    relevance: float
    link_confidence: float
    matched: str
    event: str | None = None
    event_direction: int = 0
    signal: float = 0.0          # sentiment score blended with the event's directional prior
    signal_label: str = "neutral"


@dataclass
class Analysis:
    text: str
    entities: list[EntityAnalysis]
    events: list[dict]
    primary_event: str | None
    sectors: list[str]
    market_wide: bool
    listicle: bool
    sentiment: str
    score: float
    confidence: float
    backend: str
    summary: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def blend_signal(score: float, ev: dict | None) -> tuple[float, str]:
    """70% model sentiment + 30% event prior (e.g. 'declared L1 bidder' is positive even if the
    wording is flat; 'SEBI penalty' is negative). Thresholded at +/-0.15 for the label."""
    d = (ev or {}).get("direction", 0) * (ev or {}).get("confidence", 0.0)
    s = round(0.7 * score + 0.3 * d, 4)
    return s, ("positive" if s > 0.15 else "negative" if s < -0.15 else "neutral")


def analyze_many(texts: list[str], sources: list[str | None] | None = None, use_llm: bool | None = None) -> list[Analysis]:
    u = get_universe()
    model = get_model()
    sources = sources or [None] * len(texts)
    texts = [t or "" for t in texts]

    # --- optional LLM enrichment (bounded so a big backfill does not run up a bill) ---
    llm_out: list[dict | None] = [None] * len(texts)
    want_llm = settings.sentiment_backend in ("auto", "llm") if use_llm is None else use_llm
    if want_llm:
        from .llm import get_client
        client = get_client()
        if client is not None:
            n = min(len(texts), settings.llm_max_items)
            llm_out[:n] = client.analyze(texts[:n])

    # --- local pass: doc-level + target-clause sentiment in two vectorised calls ---
    matches = [u.match(t) for t in texts]
    doc_sent = model.predict(texts)
    clause_jobs: list[tuple[int, str, str]] = []
    for i, (t, ms) in enumerate(zip(texts, matches)):
        if len({m.symbol for m in ms}) > 1:
            seen = set()
            for m in ms:
                if m.symbol not in seen:
                    seen.add(m.symbol)
                    clause_jobs.append((i, m.symbol, target_clause(t, m.start, m.end)))
    clause_res = dict(((i, s), r) for (i, s, _), r in zip(clause_jobs, model.predict([c for _, _, c in clause_jobs])))

    results = []
    for i, text in enumerate(texts):
        ms = matches[i]
        ev_hits = extract_events(text)
        events = [{"type": h.event, "direction": h.direction, "confidence": h.confidence, "evidence": h.evidence} for h in ev_hits]
        syms: list[str] = []
        for m in ms:
            if m.symbol not in syms:
                syms.append(m.symbol)
        listicle = is_listicle(text, len(syms))
        sw = source_weight(sources[i])
        ents: list[EntityAnalysis] = []
        for rank, sym in enumerate(syms):
            link = max(m.confidence for m in ms if m.symbol == sym)
            r = clause_res.get((i, sym), doc_sent[i])
            if (i, sym) in clause_res:  # multi-company headline: events of the company's own clause
                first = next(m for m in ms if m.symbol == sym)
                own = extract_events(target_clause(text, first.start, first.end))
                ev = ({"type": own[0].event, "direction": own[0].direction, "confidence": own[0].confidence}
                      if own else (events[0] if events else None))
            else:
                ev = events[0] if events else None
            sig, sig_label = blend_signal(r.score, ev)
            rel = (1.0 if rank == 0 else 0.75) / (1 + 0.25 * (len(syms) - 1))
            rel *= (0.5 if listicle else 1.0) * sw * (0.6 + 0.4 * link)
            ents.append(EntityAnalysis(sym, u.sector_of(sym), r.label, r.score, r.confidence, round(min(rel, 1.0), 3),
                                       link, ", ".join(m.text for m in ms if m.symbol == sym),
                                       ev["type"] if ev else None, ev["direction"] if ev else 0, sig, sig_label))
        sectors = sorted(set(u.sectors_in(text)) | {e.sector for e in ents})
        d = doc_sent[i]
        a = Analysis(text, ents, events, events[0]["type"] if events else None, sectors, u.is_market_wide(text),
                     listicle, d.label, d.score, d.confidence, d.backend)

        llm = llm_out[i]
        if llm:
            _merge_llm(a, llm, u, sw)
        results.append(a)
    return results


def _merge_llm(a: Analysis, llm: dict, u, sw: float) -> None:
    """LLM output wins on entity selection and sentiment; rule-based events are kept and unioned."""
    local = {e.symbol: e for e in a.entities}
    ents = []
    have = {e["type"] for e in a.events}
    llm_ev = llm["events"][0] if llm["events"] else None
    for c in llm["companies"]:
        base = local.get(c["symbol"])
        ev_type = base.event if base and base.event else (llm_ev["type"] if llm_ev else None)
        ev_dir = base.event_direction if base and base.event else (llm_ev["direction"] if llm_ev else 0)
        # the LLM already reads context, so its score is used as the signal directly
        ents.append(EntityAnalysis(
            c["symbol"], u.sector_of(c["symbol"]), c["sentiment"], c["score"], c["confidence"],
            round(min(1.0, c["relevance"] * sw * (0.5 if a.listicle else 1.0)), 3),
            base.link_confidence if base else 0.85, base.matched if base else c["symbol"],
            ev_type, ev_dir, c["score"], c["sentiment"]))
    a.entities = ents
    for e in llm["events"]:
        if e["type"] not in have:
            a.events.append({"type": e["type"], "direction": e["direction"], "confidence": 0.8, "evidence": "llm"})
    if llm.get("surprise") in ("beat", "miss") and "earnings_surprise" not in have:
        a.events.insert(0, {"type": "earnings_surprise", "direction": 1 if llm["surprise"] == "beat" else -1,
                            "confidence": 0.85, "evidence": f"llm: {llm['surprise']}"})
    a.primary_event = a.events[0]["type"] if a.events else None
    a.sectors = sorted(set(a.sectors) | set(llm.get("sectors") or []))
    a.extra["local_sentiment"] = a.sentiment
    a.sentiment = llm["overall_sentiment"]
    a.backend = "llm"
    a.summary = llm.get("summary", "")
    a.extra["llm_agrees_with_local"] = a.extra["local_sentiment"] == a.sentiment


def analyze(text: str, source: str | None = None, use_llm: bool | None = None) -> Analysis:
    return analyze_many([text], [source], use_llm=use_llm)[0]
