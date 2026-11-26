"""Train / evaluate the bundled local sentiment model.

    python -m newsimpact.nlp.train            # train, evaluate, save models/ + reports/sentiment_eval.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import settings, ROOT
from .sentiment import LABELS, normalize, target_clause

TRAIN_DIR = ROOT / "data" / "training"


def build_pipeline(C: float = 4.0):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline

    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=40000)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=60000)),
    ])
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    return Pipeline([("features", feats), ("clf", clf)])


def _load_docs() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_DIR / "sentiment_docs.csv")
    df = df[df.label.isin(LABELS)].dropna(subset=["text"])
    df["norm"] = df.text.map(normalize)
    return df


def train_and_save(path: Path | None = None, quiet: bool = False, evaluate: bool = False) -> dict:
    import joblib
    import sklearn
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    path = Path(path or settings.model_dir / "sentiment_tfidf_lr.joblib")
    df = _load_docs()
    train, test = df[df.split == "train"], df[df.split == "test"]
    report: dict = {"trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "sklearn": sklearn.__version__,
                    "n_train": int(len(train)), "n_test": int(len(test))}

    if evaluate:
        # small C search on a validation slice of train (no test leakage)
        rng = np.random.RandomState(0)
        mask = rng.rand(len(train)) < 0.85
        best = None
        for C in (1.0, 2.0, 4.0, 8.0):
            p = build_pipeline(C).fit(train.norm[mask], train.label[mask])
            f1 = f1_score(train.label[~mask], p.predict(train.norm[~mask]), average="macro")
            if not quiet:
                print(f"C={C}: val macro-F1={f1:.3f}")
            if best is None or f1 > best[1]:
                best = (C, f1)
        C = best[0]
        report["chosen_C"] = C
        pipe = build_pipeline(C).fit(train.norm, train.label)
        res = {}
        for name, sub in [("sentfin_test (Indian headlines)", test[test.source == "sentfin"]),
                          ("phrasebank_test", test[test.source == "phrasebank"]),
                          ("all_test", test)]:
            pred = pipe.predict(sub.norm)
            res[name] = {
                "n": int(len(sub)),
                "accuracy": round(accuracy_score(sub.label, pred), 4),
                "macro_f1": round(f1_score(sub.label, pred, average="macro"), 4),
                "majority_baseline_acc": round(sub.label.value_counts(normalize=True).max(), 4),
                "confusion_matrix(labels=neg,neu,pos)": confusion_matrix(sub.label, pred, labels=list(LABELS)).tolist(),
                "per_class": {k: {m: round(v, 3) for m, v in d.items()} for k, d in
                              classification_report(sub.label, pred, output_dict=True).items() if k in LABELS},
            }
        report["doc_level"] = res
        report["entity_level"] = _entity_eval(pipe)
    else:
        pipe = build_pipeline().fit(df[df.split == "train"].norm, df[df.split == "train"].label)

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipe, "meta": report}, path, compress=3)
    if evaluate:
        out = settings.reports_dir / "sentiment_eval.json"
        out.write_text(json.dumps(report, indent=2))
        if not quiet:
            print(json.dumps(report, indent=2))
    return report


def _entity_eval(pipe) -> dict:
    """Entity-level (target-aware) evaluation on SEntFiN headlines that mention >1 entity
    with *different* sentiments - the hard case where document-level sentiment is wrong."""
    import re
    from sklearn.metrics import accuracy_score, f1_score

    ent = pd.read_csv(TRAIN_DIR / "sentfin_entities.csv")
    ent = ent[(ent.split == "test") & (ent.n_entities > 1)]
    mixed = ent.groupby("text").label.nunique()
    ent = ent[ent.text.isin(mixed[mixed > 1].index)]
    doc_pred, tgt_pred = [], []
    for _, r in ent.iterrows():
        doc_pred.append(r.text)
        m = re.search(re.escape(str(r.entity)), r.text)
        tgt_pred.append(target_clause(r.text, m.start(), m.end()) if m else r.text)
    dp = pipe.predict([normalize(t) for t in doc_pred])
    tp = pipe.predict([normalize(t) for t in tgt_pred])
    return {
        "n_entity_mentions": int(len(ent)),
        "doc_level_accuracy": round(accuracy_score(ent.label, dp), 4),
        "target_clause_accuracy": round(accuracy_score(ent.label, tp), 4),
        "doc_level_macro_f1": round(f1_score(ent.label, dp, average="macro"), 4),
        "target_clause_macro_f1": round(f1_score(ent.label, tp, average="macro"), 4),
    }


if __name__ == "__main__":
    train_and_save(evaluate=True)
