"""Cross-marketplace check: a specification model trained on mebel.kz, applied to Amazon (ABO) listings.

Only inputs both stores have: orientation-free sizes and materials (no price: ABO has none; no weight:
mebel.kz has none; no colour: the two stores name colours differently). Prototype of final.ipynb section 6.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from furniture.abo import MATERIALS, SIZE, load_abo, size_features, spec_prep  # noqa: E402
from furniture.features import TYPES, build_groups  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FEATS = SIZE + MATERIALS


def model():
    return Pipeline([("prep", spec_prep(SIZE, colour=False)), ("model", SVC(C=3, gamma="scale"))])


mebel = pd.read_csv(ROOT / "data" / "processed" / "furniture.csv")
mebel = size_features(mebel)
mebel["group"] = build_groups(mebel)
abo = load_abo()
abo = abo[abo[["height_cm", "long_cm", "short_cm"]].notna().all(axis=1)].reset_index(drop=True)
cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)

within_mebel = cross_val_predict(model(), mebel[FEATS], mebel["furniture_type"], groups=mebel["group"], cv=cv)
within_abo = cross_val_predict(model(), abo[FEATS], abo["furniture_type"], groups=abo["group"], cv=cv)
transfer = model().fit(mebel[FEATS], mebel["furniture_type"]).predict(abo[FEATS])

f = lambda yt, yp: f1_score(yt, yp, average="macro", labels=TYPES)
print(pd.DataFrame({
    "listings scored": [len(mebel), len(abo), len(abo)],
    "macro-F1": [f(mebel["furniture_type"], within_mebel), f(abo["furniture_type"], within_abo),
                 f(abo["furniture_type"], transfer)],
}, index=["mebel.kz -> mebel.kz (grouped CV)", "ABO -> ABO (grouped CV)", "mebel.kz -> ABO (transfer)"]).round(3))
print(classification_report(abo["furniture_type"], transfer, labels=TYPES, zero_division=0, digits=3))
print(pd.crosstab(abo["furniture_type"], pd.Series(transfer, name="predicted")).reindex(index=TYPES, columns=TYPES, fill_value=0))
print(pd.concat({"mebel": mebel.groupby("furniture_type")[["height_cm", "long_cm", "short_cm"]].median(),
                 "abo": abo.groupby("furniture_type")[["height_cm", "long_cm", "short_cm"]].median()}, axis=1).round(0))
