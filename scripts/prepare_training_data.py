"""Build data/training/*.csv from the public datasets.

Sources (downloaded with git / https; run once):
  * SEntFiN 1.0 (Sinha et al., 2022, MIT) - 10.7k Indian financial headlines with entity-level sentiment
      https://github.com/pyRis/SEntFiN
  * Financial PhraseBank, AllAgree subset (Malo et al., 2014, CC BY-NC-SA 3.0)

Usage:
  python scripts/prepare_training_data.py --sentfin path/to/SEntFiN.csv --phrasebank path/to/Sentences_AllAgree.txt
"""
import argparse
import ast
import hashlib
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "training"


def split_of(text: str, test_share: float = 0.2) -> str:
    h = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % 1000
    return "test" if h < test_share * 1000 else "train"


_PAIR = re.compile(r"""['"](.+?)['"]\s*:\s*'(positive|negative|neutral)'""")


def parse_decisions(raw: str) -> dict:
    try:
        return ast.literal_eval(raw)
    except Exception:  # a few rows contain unescaped apostrophes, e.g. 'Osian's Art Fund'
        return {m.group(1): m.group(2) for m in _PAIR.finditer(raw)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentfin", required=True)
    ap.add_argument("--phrasebank", required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    sf = pd.read_csv(a.sentfin)
    doc_rows, ent_rows = [], []
    for _, r in sf.iterrows():
        d = parse_decisions(r["Decisions"])
        if not d:
            continue
        labels = set(d.values())
        sp = split_of(r["Title"])
        for ent, lab in d.items():
            ent_rows.append({"text": r["Title"], "entity": ent, "label": lab, "n_entities": len(d), "split": sp})
        if len(labels) == 1:
            doc_rows.append({"text": r["Title"], "label": labels.pop(), "source": "sentfin", "split": sp})

    pb = []
    for line in Path(a.phrasebank).read_text(encoding="latin-1").splitlines():
        if "@" not in line:
            continue
        text, lab = line.rsplit("@", 1)
        pb.append({"text": text.strip(), "label": lab.strip(), "source": "phrasebank", "split": split_of(text)})

    docs = pd.DataFrame(doc_rows + pb).drop_duplicates("text")
    docs.to_csv(OUT / "sentiment_docs.csv", index=False)
    pd.DataFrame(ent_rows).to_csv(OUT / "sentfin_entities.csv", index=False)
    print(docs.groupby(["source", "split", "label"]).size())
    print("entity rows:", len(ent_rows))


if __name__ == "__main__":
    main()
