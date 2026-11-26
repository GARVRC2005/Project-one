"""Central configuration. Everything can be overridden with environment variables
(or a `.env` file in the project root, loaded automatically if present)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = Path(__file__).resolve().parent


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass
class Settings:
    # storage
    data_dir: Path = field(default_factory=lambda: Path(_env("NIA_DATA_DIR", str(ROOT / "data"))))
    db_path: Path = field(default_factory=lambda: Path(_env("NIA_DB", str(ROOT / "data" / "newsimpact.db"))))
    snapshot_dir: Path = field(default_factory=lambda: ROOT / "data" / "snapshot")
    model_dir: Path = field(default_factory=lambda: ROOT / "models")
    reports_dir: Path = field(default_factory=lambda: ROOT / "reports")

    # sentiment backend: auto | llm | finbert | local | lexicon
    sentiment_backend: str = field(default_factory=lambda: _env("NIA_SENTIMENT_BACKEND", "auto").lower())

    # LLM (optional). Provider is picked from whichever key is present unless NIA_LLM_PROVIDER is set.
    llm_provider: str = field(default_factory=lambda: _env("NIA_LLM_PROVIDER").lower())
    llm_model: str = field(default_factory=lambda: _env("NIA_LLM_MODEL"))
    anthropic_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    openai_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    gemini_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY"))
    llm_batch_size: int = field(default_factory=lambda: int(_env("NIA_LLM_BATCH", "15")))
    llm_max_items: int = field(default_factory=lambda: int(_env("NIA_LLM_MAX_ITEMS", "500")))
    llm_timeout: int = field(default_factory=lambda: int(_env("NIA_LLM_TIMEOUT", "60")))

    # market conventions (NSE)
    market_close_ist: tuple[int, int] = (15, 30)
    benchmark_symbol: str = "NIFTY50"
    benchmark_yahoo: str = "^NSEI"

    # event study
    est_window: tuple[int, int] = (-120, -11)   # trading days, relative to event day
    est_min_obs: int = 60
    car_windows: tuple[tuple[int, int], ...] = ((0, 0), (-1, 1), (0, 1), (1, 3), (1, 5), (-5, -1))
    shrinkage_k: float = 20.0   # empirical-Bayes prior strength for impact estimates

    def __post_init__(self) -> None:
        for d in (self.data_dir, self.model_dir, self.reports_dir):
            Path(d).mkdir(parents=True, exist_ok=True)

    # ---- LLM resolution -------------------------------------------------
    def resolved_llm(self) -> tuple[str, str, str] | None:
        """Return (provider, model, key) or None if no LLM is configured."""
        defaults = {
            "anthropic": ("claude-haiku-5-5", self.anthropic_key),
            "openai": ("gpt-4o-mini", self.openai_key),
            "gemini": ("gemini-2.5-flash", self.gemini_key),
        }
        order = [self.llm_provider] if self.llm_provider else ["anthropic", "openai", "gemini"]
        for p in order:
            if p in defaults and defaults[p][1]:
                return p, (self.llm_model or defaults[p][0]), defaults[p][1]
        return None


settings = Settings()
