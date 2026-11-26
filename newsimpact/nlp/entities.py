"""Company / sector recognition for Indian market news.

Dictionary-based entity linking against the NIFTY 50 universe
(`newsimpact/data_universe.csv`), with:
  * case-sensitive aliases (prefix `=`) for tickers / short names such as ITC, SBI, BEL,
  * regex aliases (prefix `re:`) for ambiguous names such as "BSE" (exchange vs. company),
  * a blocklist of look-alike entities ("HDFC Securities", "Reliance Power", "Axis MF" ...)
    that consume their span so that the alias inside them is NOT tagged,
  * longest-match-first, non-overlapping resolution,
  * a per-alias link confidence (full legal name > ticker > brand / executive name).

The design is deliberately transparent: every tag carries the matched text and confidence,
which feeds the relevance / confidence scores of alerts.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent

# aliases that point to a company only indirectly (brand, subsidiary, executive, very short name)
INDIRECT_ALIASES = {
    "pulsar", "chetak", "sunil bharti mittal", "gopal vittal", "salil parekh", "nandan nilekani",
    "mukesh ambani", "uday kotak", "deepinder goyal", "dilip shanghvi", "sashidhar jagdishan",
    "tanishq", "zudio", "westside", "novelis", "royal enfield", "jaguar land rover", "jlr", "birla opus",
    "jio", "larsen", "mahindra", "reliance", "nestle", "nestlé", "eicher", "kotak", "bajaj allianz",
    "apollo 24/7", "apollo healthco", "blinkit", "ongc videsh", "jio blackrock", "jio payments bank",
    "adani airports", "adani new industries", "reliance retail", "reliance jio", "jio platforms",
    "reliance consumer", "tata motors", "indigo", "interglobe", "icici", "hdfc",
}

SECTOR_KEYWORDS: dict[str, list[str]] = {
    "Banking": [r"\bbank(?:s|ing)?\b", r"\bPSU banks?\b", r"\blenders?\b", r"\bNPAs?\b", r"\bdeposits?\b", r"\bNIM\b"],
    "IT": [r"\bIT (?:stocks?|sector|services|companies|firms|majors?|index)\b", r"\bNifty IT\b", r"\btech (?:stocks|services)\b", r"\bsoftware services\b", r"\bdeal wins?\b"],
    "Pharma": [r"\bpharma\b", r"\bUSFDA\b", r"\bUS FDA\b", r"\bdrugs?\b", r"\bgeneric\b", r"\bANDA\b"],
    "Automobile": [r"\bauto (?:stocks?|sales|sector|makers?|index)\b", r"\bcarmakers?\b", r"\bautomakers?\b", r"\bpassenger vehicles?\b", r"\btwo-wheelers?\b", r"\bEVs?\b", r"\bSUVs?\b", r"\bvehicle sales\b"],
    "FMCG": [r"\bFMCG\b", r"\bconsumer staples\b", r"\brural demand\b", r"\bcigarettes?\b"],
    "Metals & Mining": [r"\bmetal(?:s)? (?:stocks?|index|sector)\b", r"\bsteel\b", r"\baluminium\b", r"\bcopper\b", r"\bcoal\b", r"\biron ore\b", r"\bNifty Metal\b"],
    "Oil & Gas": [r"\bcrude\b", r"\boil (?:prices?|marketing|companies)\b", r"\bOMCs?\b", r"\brefin(?:ery|ing)\b", r"\bnatural gas\b", r"\bpetrol\b", r"\bdiesel\b"],
    "Power": [r"\bpower (?:sector|demand|stocks?|generation|utilities)\b", r"\belectricity\b", r"\btransmission\b", r"\brenewable\b", r"\bsolar\b"],
    "Telecom": [r"\btelecom\b", r"\btariffs? hike\b", r"\b5G\b", r"\bARPU\b", r"\bspectrum\b", r"\bsatcom\b"],
    "Cement": [r"\bcement\b"],
    "Insurance": [r"\binsur(?:ance|er|ers)\b", r"\bIRDAI\b", r"\bVNB\b", r"\bpremiums?\b"],
    "NBFC": [r"\bNBFCs?\b", r"\bnon-banking\b", r"\bmicrofinance\b", r"\bgold loans?\b", r"\bconsumer loans?\b"],
    "Aviation": [r"\bairlines?\b", r"\baviation\b", r"\bDGCA\b", r"\bATF\b", r"\bflights?\b"],
    "Healthcare": [r"\bhospitals?\b", r"\bhealthcare\b"],
    "Defence": [r"\bdefen[cs]e (?:stocks?|orders?|sector|ministry|exports)\b", r"\bmissiles?\b", r"\bMoD\b"],
    "Capital Markets": [r"\bexchanges?\b", r"\bF&O\b", r"\bderivatives?\b", r"\bdemat\b", r"\bbrokers?\b"],
    "Consumer Durables": [r"\bpaints?\b", r"\bjewell?ery\b", r"\bgold demand\b"],
    "Retail": [r"\bretail(?:er|ers)?\b", r"\bapparel\b", r"\bfashion\b"],
    "Infrastructure": [r"\binfra(?:structure)?\b", r"\bports?\b", r"\bEPC\b", r"\bcapex\b"],
    "Consumer Services": [r"\bquick commerce\b", r"\bfood delivery\b", r"\bq-commerce\b"],
}

MARKET_KEYWORDS = [
    r"\bSensex\b", r"\bNifty(?: 50)?\b", r"\bRBI\b", r"\brepo rate\b", r"\bFIIs?\b", r"\bFPIs?\b", r"\bDIIs?\b",
    r"\bUnion Budget\b", r"\binflation\b", r"\bCPI\b", r"\bGDP\b", r"\brupee\b", r"\btariffs?\b", r"\bFed\b",
    r"\bmarket(?:s)? (?:today|outlook|wrap|close|open)\b", r"\bDalal Street\b",
]


@dataclass(frozen=True)
class Company:
    symbol: str
    yahoo: str
    name: str
    sector: str


@dataclass
class EntityMatch:
    symbol: str
    text: str
    start: int
    end: int
    confidence: float


@dataclass
class _Alias:
    pattern: re.Pattern
    symbol: str | None  # None => blocklisted entity
    confidence: float


def _boundary_pattern(alias: str, case_sensitive: bool) -> re.Pattern:
    esc = re.escape(alias).replace(r"\ ", r"\s+")
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(rf"(?<![A-Za-z0-9&]){esc}(?![A-Za-z0-9&])", flags)


class Universe:
    def __init__(self, universe_csv: Path | None = None, blocklist: Path | None = None):
        universe_csv = universe_csv or PKG / "data_universe.csv"
        blocklist = blocklist or PKG / "data_blocklist.txt"
        self.companies: dict[str, Company] = {}
        self.aliases: list[_Alias] = []
        with open(universe_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                c = Company(row["symbol"], row["yahoo"], row["name"], row["sector"])
                self.companies[c.symbol] = c
                for raw in row["aliases"].split(";"):
                    raw = raw.strip()
                    if not raw:
                        continue
                    self.aliases.append(self._compile(raw, c))
        for line in blocklist.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cs = line.startswith("=")
            text = line[1:] if cs else line
            self.aliases.append(_Alias(_boundary_pattern(text, cs), None, 0.0))
        # patterns containing an acronym (2+ capitals, e.g. "IT", "USFDA", "MoD") stay case-sensitive
        self._sector_re = {s: [re.compile(p, 0 if re.search(r"[A-Z][A-Za-z]*[A-Z]", p) else re.IGNORECASE)
                               for p in pats] for s, pats in SECTOR_KEYWORDS.items()}
        self._market_re = [re.compile(p) for p in MARKET_KEYWORDS]

    @staticmethod
    def _compile(raw: str, c: Company) -> _Alias:
        if raw.startswith("re:"):
            return _Alias(re.compile(raw[3:]), c.symbol, 0.9)
        cs = raw.startswith("=")
        text = raw[1:] if cs else raw
        low = text.lower()
        if low in INDIRECT_ALIASES:
            conf = 0.7
        elif low == c.name.lower() or len(text) >= 10:
            conf = 1.0
        else:
            conf = 0.92
        return _Alias(_boundary_pattern(text, cs), c.symbol, conf)

    # ------------------------------------------------------------------
    def sector_of(self, symbol: str) -> str | None:
        c = self.companies.get(symbol)
        return c.sector if c else None

    def match(self, text: str) -> list[EntityMatch]:
        """Return non-overlapping company mentions, in order of appearance."""
        if not text:
            return []
        cands = []
        for a in self.aliases:
            for m in a.pattern.finditer(text):
                cands.append((m.start(), m.end(), a))
        # longest span first; for equal length, blocklist (None) wins, then higher confidence
        cands.sort(key=lambda t: (-(t[1] - t[0]), t[2].symbol is not None, -t[2].confidence))
        taken: list[tuple[int, int]] = []
        out: list[EntityMatch] = []
        for s, e, a in cands:
            if any(s < te and ts < e for ts, te in taken):
                continue
            taken.append((s, e))
            if a.symbol is not None:
                out.append(EntityMatch(a.symbol, text[s:e], s, e, a.confidence))
        out.sort(key=lambda m: m.start)
        return out

    def companies_in(self, text: str) -> dict[str, dict]:
        """Aggregate matches per symbol: {symbol: {confidence, mentions, first_pos, matched}}."""
        agg: dict[str, dict] = {}
        for m in self.match(text):
            d = agg.setdefault(m.symbol, {"confidence": 0.0, "mentions": 0, "first_pos": m.start, "matched": []})
            d["confidence"] = max(d["confidence"], m.confidence)
            d["mentions"] += 1
            d["matched"].append(m.text)
        return agg

    def sectors_in(self, text: str) -> list[str]:
        found = []
        for s, pats in self._sector_re.items():
            if any(p.search(text) for p in pats):
                found.append(s)
        return found

    def is_market_wide(self, text: str) -> bool:
        return any(p.search(text) for p in self._market_re)


@lru_cache(maxsize=1)
def get_universe() -> Universe:
    return Universe()
