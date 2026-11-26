"""Financial event extraction.

A transparent, rule-based taxonomy tuned on Indian market headlines (Moneycontrol / ET /
Mint / Business Standard style). Each rule carries:
  * the event type,
  * a directional prior (+1 / -1 / 0) used to break ties with the sentiment model,
  * a base confidence (how specific the pattern is).

The LLM backend (when an API key is configured) returns the same taxonomy; rules are
always run as well so that the system keeps working (and can be cross-checked) offline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Ordered by specificity: corporate events first, generic price commentary last.
EVENT_TYPES = {
    "earnings": "Quarterly / annual results, profit, revenue, margins",
    "earnings_surprise": "Results explicitly beating or missing estimates",
    "merger_acquisition": "M&A, stake acquisition, demerger, JV",
    "management_change": "CEO / MD / CFO / chairman / board changes",
    "dividend_buyback": "Dividend, buyback, bonus issue, stock split",
    "order_win": "Orders, contracts, project wins, LoA",
    "regulatory_legal": "Regulator / court / tax action, penalties, probes, USFDA observations",
    "regulatory_approval": "Approvals, licences, clearances, USFDA nods",
    "rating_change": "Broker upgrades / downgrades, target changes, credit ratings",
    "capital_raise": "QIP, rights, bonds/NCDs, IPO of a unit, fund raise",
    "stake_sale": "Block / bulk deals, promoter selling, OFS, pledge",
    "guidance": "Outlook / guidance / targets set by management",
    "business_update": "Monthly sales, volumes, deliveries, business updates",
    "expansion_launch": "Capacity expansion, capex, new plant, product launch, partnerships",
    "operational_risk": "Fire, strike, outage, cyber attack, recall, accident",
    "macro_policy": "RBI / budget / tariff / tax-rate / commodity macro news",
    "price_action": "News about the share price move itself (reactive, lagging)",
}


@dataclass
class EventRule:
    event: str
    pattern: str
    direction: int = 0
    confidence: float = 0.8
    regex: re.Pattern = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.regex = re.compile(self.pattern, re.IGNORECASE)


_NUM = r"(?:\d+(?:\.\d+)?\s*(?:%|per\s*cent|pc|bps))"
_UPV = r"(?:rises?|rose|jumps?|jumped|surges?|surged|soars?|soared|climbs?|grows?|grew|up|increases?|increased|gains?|spurts?|zooms?|doubles?|beats?|higher|improves?|rebounds?)"
_DNV = r"(?:falls?|fell|drops?|dropped|declines?|declined|dips?|dipped|slumps?|slumped|plunges?|plunged|sinks?|sank|down|decreases?|decreased|slides?|tumbles?|lower|shrinks?|contracts?|halves?|crashes?|misses?|cuts?)"

RULES: list[EventRule] = [
    # ---- earnings surprise (most specific) ----
    EventRule("earnings_surprise", r"\b(?:beats?|tops?|surpass(?:es)?|exceeds?|ahead of|above|better[- ]than[- ]expected)\b[^.;:]{0,25}\b(?:estimates?|expectations?|street|consensus|forecasts?|views?)\b", +1, 0.92),
    EventRule("earnings_surprise", r"\bbetter[- ]than[- ](?:expected|estimated)\b", +1, 0.9),
    EventRule("earnings_surprise", r"\b(?:miss(?:es|ed)?|below|short of|lags?|trails?|worse[- ]than[- ]expected|disappoints?)\b[^.;:]{0,25}\b(?:estimates?|expectations?|street|consensus|forecasts?|views?)\b", -1, 0.92),
    EventRule("earnings_surprise", r"\b(?:worse|weaker)[- ]than[- ](?:expected|estimated)\b", -1, 0.9),
    # ---- earnings ----
    EventRule("earnings", rf"\b(?:net\s+)?(?:profit|PAT|earnings|EBITDA|revenue|income|margin)s?\b[^.;:]{{0,30}}\b{_UPV}\b", +1, 0.88),
    EventRule("earnings", rf"\b(?:net\s+)?(?:profit|PAT|earnings|EBITDA|revenue|income|margin)s?\b[^.;:]{{0,30}}\b{_DNV}\b", -1, 0.88),
    EventRule("earnings", r"\b(?:net\s+loss|loss widens|swings? to (?:a )?loss|posts? (?:a )?loss)\b", -1, 0.9),
    EventRule("earnings", r"\b(?:swings? to (?:a )?profit|returns? to profit|turns? profitable|loss narrows)\b", +1, 0.9),
    EventRule("earnings", r"\b(?:record|highest[- ]ever)\s+(?:quarterly\s+)?(?:profit|revenue|earnings)\b", +1, 0.9),
    EventRule("earnings", r"\bQ[1-4]\s*(?:FY)?\s*'?\d{0,4}\s*(?:results?|earnings|update|numbers|preview|review|show)\b", 0, 0.85),
    EventRule("earnings", r"\b(?:quarterly|Q[1-4]|H[12]|FY\d{2,4}|annual|nine[- ]month(?:ly)?)\s+(?:results?|earnings|numbers)\b", 0, 0.85),
    EventRule("earnings", r"\b(?:results? (?:date|preview|today|announced|declared)|earnings call|board meeting to consider)\b", 0, 0.75),
    EventRule("earnings", r"\bQ[1-4]\s*(?:FY)?\s*\d{2,4}\b", 0, 0.6),
    # ---- M&A ----
    EventRule("merger_acquisition", r"\b(?:acquires?|acquired|acquisition|acquiring|to acquire|takeover|take over|buyout|merger|merges?|amalgamation|demerger|demerge[sd]?|scheme of arrangement)\b", 0, 0.9),
    EventRule("merger_acquisition", r"\b(?:to buy|buys?|bought|picks? up|purchases?)\b[^.;:]{0,40}\bstake\b", 0, 0.88),
    EventRule("merger_acquisition", r"\b(?:joint venture|JV)\b", 0, 0.75),
    # ---- management ----
    EventRule("management_change", r"\b(?:CEO|MD|CFO|COO|CTO|chairman|chairperson|chief executive|managing director|whole[- ]time director|president|director|board)\b[^.;:]{0,50}\b(?:resigns?|resigned|resignation|quits?|steps? down|exits?|exit|retires?|sacked|ousted)\b", -1, 0.9),
    EventRule("management_change", r"\b(?:resigns?|resigned|quits?|steps? down)\b[^.;:]{0,40}\b(?:CEO|MD|CFO|COO|chairman|chief|director)\b", -1, 0.88),
    EventRule("management_change", r"\b(?:appoints?|appointed|appointment|names?|named|elevates?|elevated|re-?appoint(?:s|ed|ment)?|succession|successor|new (?:CEO|MD|CFO|chairman))\b[^.;:]{0,50}\b(?:CEO|MD|CFO|COO|CTO|chairman|chief|managing director|director|head)\b", 0, 0.88),
    EventRule("management_change", r"\b(?:CEO|MD|CFO|chairman)\b[^.;:]{0,15}\b(?:appointed|named|elevated)\b", 0, 0.85),
    # ---- shareholder payouts ----
    EventRule("dividend_buyback", r"\b(?:dividend|buy-?back|bonus (?:issue|shares?)|stock split|share split|record date|special dividend)\b", +1, 0.9),
    # ---- orders ----
    EventRule("order_win", r"\b(?:bags?|bagged|wins?|won|secures?|secured|receives?|received|bags|lands?|grabs?|clinch(?:es|ed)?)\b[^.;:]{0,50}\b(?:orders?|contracts?|deals?|projects?|LoA|letter of award|mandate|tender)\b", +1, 0.9),
    EventRule("order_win", r"\b(?:orders? worth|order inflows?|order book|declared (?:as )?(?:the )?(?:successful|L1|lowest) bidder|emerges? (?:as )?(?:the )?(?:L1|lowest) bidder)\b", +1, 0.88),
    EventRule("order_win", r"\b(?:deal wins?|mega deal|large deal|TCV)\b", +1, 0.8),
    # ---- regulatory ----
    EventRule("regulatory_legal", r"\b(?:penalty|penalised|penalized|fined|fine of|show[- ]cause|probe|investigation|raid(?:s|ed)?|searches|tax demand|GST (?:notice|demand)|income tax (?:notice|demand)|notice from|SEBI (?:order|bars?|bans?|action|notice)|lawsuit|sued|court|NCLT|tribunal|CCI|antitrust|ban(?:s|ned)?|warning letter|import alert|Form 483|observations?|OAI|recall(?:s|ed)?|flags?|non-compliance|violation|irregularit(?:y|ies)|fraud|whistleblower)\b", -1, 0.85),
    EventRule("regulatory_approval", r"\b(?:approval|approves?|approved|nod|clearance|clears?|cleared|licen[cs]e (?:granted|received)|gets? (?:the )?(?:go-ahead|green signal)|EIR|establishment inspection report|tentative approval|final approval|zero observations?|VAI)\b", +1, 0.82),
    # ---- ratings / brokers ----
    EventRule("rating_change", r"\b(?:upgrades?|upgraded|raises? target|hikes? target|target (?:price )?(?:raised|hiked)|initiates? (?:coverage )?with (?:a )?buy|overweight|outperform|buy rating|rating to buy|top pick)\b", +1, 0.88),
    EventRule("rating_change", r"\b(?:downgrades?|downgraded|cuts? target|lowers? target|target (?:price )?(?:cut|lowered|slashed)|underweight|underperform|sell rating|rating to sell|reduce rating)\b", -1, 0.88),
    EventRule("rating_change", r"\b(?:target price|price target|TP of|brokerages?|Jefferies|Morgan Stanley|Goldman|CLSA|Nomura|Macquarie|JPMorgan|Citi|Bernstein|UBS|HSBC|Motilal Oswal|Kotak Institutional|Nuvama|Emkay|Antique|ICICI Securities|HDFC Securities|Axis Securities|stocks? to buy|buy or sell|should you buy)\b", 0, 0.65),
    EventRule("rating_change", r"\b(?:CRISIL|ICRA|CARE Ratings|India Ratings|Moody'?s|S&P|Fitch)\b[^.;:]{0,40}\b(?:rating|outlook|affirms?|reaffirms?)\b", 0, 0.8),
    # ---- capital raising / stake sales ----
    EventRule("capital_raise", r"\b(?:QIP|rights issue|preferential (?:issue|allotment)|fund ?rais(?:e|ing)|NCDs?|bonds?|debentures?|commercial paper|IPO|DRHP|warrants?)\b", 0, 0.78),
    EventRule("capital_raise", r"\brais(?:es?|ed|ing)\b(?![^.;:]{0,20}\b(?:target|TP|price|guidance|outlook|stake))[^.;:]{0,25}\b(?:Rs|INR|USD|crore|cr|bn|billion|million|funds?|capital|debt|equity)\b|\brais(?:es?|ed|ing)\s+(?:₹|\$)", 0, 0.8),
    EventRule("stake_sale", r"\b(?:block deal|bulk deal|OFS|offer for sale|promoters? (?:sells?|sold|offloads?|pledge)|stake sale|sells? (?:\d+(?:\.\d+)?%?\s*)?stake|offloads?|divests?|divestment|pledged? shares|large trade|equity changes hands)\b", -1, 0.85),
    # ---- guidance ----
    EventRule("guidance", r"\b(?:guidance|outlook|forecast|targets?|expects?|sees?|projects?|aims? (?:for|to)|eyes)\b[^.;:]{0,40}\b(?:growth|revenue|margin|volume|demand|FY\d{2,4}|this year|next year)\b", 0, 0.75),
    EventRule("guidance", r"\b(?:raises?|upgrades?|lifts?) (?:FY\d{2,4} )?(?:revenue |growth |margin )?(?:guidance|outlook|forecast)\b", +1, 0.9),
    EventRule("guidance", r"\b(?:cuts?|lowers?|trims?|withdraws?) (?:FY\d{2,4} )?(?:revenue |growth |margin )?(?:guidance|outlook|forecast)\b", -1, 0.9),
    # ---- business update ----
    EventRule("business_update", rf"\b(?:sales|volumes?|deliveries|dispatches|wholesales?|retail sales|loan book|AUM|deposits|advances|disbursements?|production|output|subscribers?|ARPU|footfalls?)\b[^.;:]{{0,30}}\b{_UPV}\b", +1, 0.85),
    EventRule("business_update", rf"\b(?:sales|volumes?|deliveries|dispatches|wholesales?|loan book|AUM|deposits|advances|disbursements?|production|output|subscribers?)\b[^.;:]{{0,30}}\b{_DNV}\b", -1, 0.85),
    EventRule("business_update", r"\b(?:business update|monthly sales|auto sales|sales numbers|(?:January|February|March|April|May|June|July|August|September|October|November|December) sales|provisional (?:numbers|update))\b", 0, 0.8),
    # ---- expansion / launches ----
    EventRule("expansion_launch", r"\b(?:launch(?:es|ed)?|unveils?|unveiled|introduces?|rolls? out|debuts?|new (?:plant|facility|factory|model|variant|store)|capacity (?:expansion|addition)|expands?|expansion|commissions?|commissioned|capex|invests?|investment of|sets? up|inaugurates?|opens?|partners? with|partnership|ties? up|tie-?up|MoU|collaborat(?:es|ion))\b", +1, 0.72),
    # ---- operational risk ----
    EventRule("operational_risk", r"\b(?:fire|blaze|explosion|strike|lockout|outage|shutdown|shut down|disruption|cyber ?attack|ransomware|data breach|hack(?:ed)?|accident|crash|grounded|cancell?ations?|halts?|suspends? (?:operations|production))\b", -1, 0.8),
    # ---- macro ----
    EventRule("macro_policy", r"\b(?:RBI|repo rate|monetary policy|MPC|Union Budget|budget|GST (?:council|rate|cut|hike)|tariffs?|duty|anti-dumping|safeguard duty|windfall tax|excise|inflation|CPI|WPI|GDP|IIP|PMI|crude oil|oil prices|rupee|Fed|US Fed|trade deal|sanctions?|PLI scheme|government policy)\b", 0, 0.7),
    # ---- price action (reactive) ----
    EventRule("price_action", rf"\b(?:shares?|stocks?|scrip|counter)\b[^.;:]{{0,30}}\b(?:{_UPV}|rall(?:y|ies|ied)|hits? (?:a )?(?:fresh |new )?(?:record|52-week|all-time|lifetime) high|upper circuit|top gainers?)\b", +1, 0.7),
    EventRule("price_action", rf"\b(?:shares?|stocks?|scrip|counter)\b[^.;:]{{0,30}}\b(?:{_DNV}|hits? (?:a )?(?:fresh |new )?52-week low|lower circuit|top losers?|sell-?off|crack(?:s|ed)?|tank(?:s|ed)?)\b", -1, 0.7),
    EventRule("price_action", r"\b(?:share price|stock price|in focus|stocks? to watch|buzzing stocks?|top gainers?|top losers?|market (?:cap|capitalisation|capitalization)|m-?cap)\b", 0, 0.5),
]

PRIORITY = [
    "earnings_surprise", "merger_acquisition", "management_change", "regulatory_legal", "order_win",
    "stake_sale", "dividend_buyback", "capital_raise", "regulatory_approval", "guidance", "earnings",
    "operational_risk", "business_update", "rating_change", "expansion_launch", "macro_policy", "price_action",
]
_PRIO = {e: i for i, e in enumerate(PRIORITY)}


@dataclass
class EventHit:
    event: str
    direction: int
    confidence: float
    evidence: str


def extract_events(text: str) -> list[EventHit]:
    """All distinct event types found in `text`, primary (most specific) first."""
    if not text:
        return []
    best: dict[str, EventHit] = {}
    for r in RULES:
        m = r.regex.search(text)
        if not m:
            continue
        cur = best.get(r.event)
        hit = EventHit(r.event, r.direction, r.confidence, m.group(0))
        if cur is None or (r.confidence, abs(r.direction)) > (cur.confidence, abs(cur.direction)):
            best[r.event] = hit
    # "earnings_surprise" implies "earnings"; keep the surprise only
    if "earnings_surprise" in best:
        best.pop("earnings", None)
    # pure broker commentary ("stocks to buy") is weak; drop it if a concrete event exists
    if "rating_change" in best and best["rating_change"].confidence < 0.7 and len(best) > 1:
        best.pop("rating_change")
    # "price_action" is only primary when nothing else explains the move
    hits = sorted(best.values(), key=lambda h: (_PRIO[h.event], -h.confidence))
    return hits


def primary_event(text: str) -> EventHit | None:
    hits = extract_events(text)
    return hits[0] if hits else None


def is_listicle(text: str, n_companies: int) -> bool:
    """Round-up headlines ("Stocks to watch: A, B, C and more") carry little stock-specific signal."""
    if n_companies >= 4:
        return True
    return bool(re.search(r"\b(?:stocks? to (?:watch|buy)|buzzing stocks|and more|top \d+ stocks|\d+ stocks|market live|live updates|stock market today|closing bell|opening bell)\b", text, re.IGNORECASE))
