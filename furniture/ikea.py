"""IKEA Saudi Arabia price table (second market, Final): inputs and models for the replication in final.ipynb 7.5.

Same question as on mebel.kz: does the product text add to the specifications? IKEA has no material or colour
fields, so the specifications are the sizes and the type; the text is the product line name + short description
(e.g. "HEMNES Chest of 8 drawers, 160x96 cm"). Target: log(price in SAR). Grouped by product line.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from furniture.price import tfidf

ROOT = Path(__file__).resolve().parents[1]
IKEA_SIZE = ["height_cm", "long_cm", "short_cm", "footprint_m2", "volume_m3", "height_to_long", "short_to_long"]


def load_ikea_sa():
    d = pd.read_csv(ROOT / "data" / "processed" / "ikea_sa.csv")
    w, dp = d["width_cm"], d["depth_cm"]
    d["long_cm"], d["short_cm"] = np.fmax(w, dp), np.fmin(w, dp)
    d["footprint_m2"] = d["long_cm"] * d["short_cm"] / 1e4
    d["volume_m3"] = d["footprint_m2"] * d["height_cm"] / 100
    d["height_to_long"] = d["height_cm"] / d["long_cm"]
    d["short_to_long"] = d["short_cm"] / d["long_cm"]
    for c in ["long_cm", "short_cm", "height_cm", "footprint_m2", "volume_m3"]:
        d[c] = np.log1p(d[c])                                   # sizes span 1 cm to 4 m: compare them on a log scale
    return d


def ikea_pipe(model, text=False):
    parts = [("size", StandardScaler(), IKEA_SIZE),
             ("type", OneHotEncoder(handle_unknown="ignore"), ["furniture_type"])]
    if text:
        parts.append(("text", tfidf(), "text"))
    return Pipeline([("prep", ColumnTransformer(parts)), ("model", model)])
