"""Shared data preparation for the Final notebook and the app.

The feature code is the same as in endterm.ipynb (section 0), moved here so the
notebook and the Streamlit app turn raw inputs into the same columns.
"""
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TYPES = ["sofa", "armchair", "chair", "bed", "wardrobe", "chest_of_drawers", "desk", "coffee_table", "tv_stand"]
NUMERIC = ["width_cm", "depth_cm", "height_cm", "footprint_m2", "volume_m3", "height_to_width",
           "depth_to_width", "log_price", "log_price_per_m3"]
COLOUR = ["main_color"]


def add_features(d, materials=None):
    """Derived inputs from the Midterm (section 2.4). `materials` fixes the list of mat_ columns
    (pass the training list at prediction time so a single row gets every column)."""
    d = d.copy()
    d["main_color"] = d["main_color"].fillna("unknown")
    d["materials"] = d["materials"].fillna("")
    d["footprint_m2"] = d["width_cm"] * d["depth_cm"] / 1e4
    d["volume_m3"] = d["width_cm"] * d["depth_cm"] * d["height_cm"] / 1e6
    d["height_to_width"] = d["height_cm"] / d["width_cm"]
    d["depth_to_width"] = d["depth_cm"] / d["width_cm"]
    d["log_price"] = np.log(d["price_kzt"])
    d["log_price_per_m3"] = np.log(d["price_kzt"] / d["volume_m3"])
    mats = d["materials"].str.get_dummies(sep="|").add_prefix("mat_")
    if materials is not None:
        mats = mats.reindex(columns=materials, fill_value=0)
    return pd.concat([d, mats], axis=1)


def material_columns(d):
    return sorted(c for c in d.columns if c.startswith("mat_"))


def make_prep(materials):
    return ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        ("materials", "passthrough", materials),
        ("colour", OneHotEncoder(handle_unknown="ignore", min_frequency=10), COLOUR),
    ])


def make_pipeline(model, materials):
    return Pipeline([("prep", make_prep(materials)), ("model", model)])


# ---------------------------------------------------------------- grouping

def model_word(name):
    """The product's model name, e.g. 'Диван прямой Линс Велюр Голубой' -> 'Линс',
    'Журнальный стол Хэйко-1 Черный' -> 'Хэйко'. The first capitalised word after the first word,
    cut at a hyphen or digit. Colour and fabric words come after it, so colour variants share it."""
    for tok in str(name).replace("​", "").split()[1:]:
        if tok[:1].isupper():
            return re.split(r"[-\d]", tok)[0] or tok
    return ""


def build_groups(d):
    """Listings that are the same design end up in one group. Two listings are joined when they share
    (1) the store's series, (2) the type and model name, or (3) the type, all three sizes and the price.
    Joins are transitive (union-find), so a chain of shared keys forms one group."""
    n = len(d)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    no_series = d["series"].astype(str).values == d["sku"].astype(str).values
    keys = {
        "series": np.where(no_series, None, "s:" + d["series"].astype(str).values),
        "model": ("m:" + d["furniture_type"] + "|" + d["name"].map(model_word)).values,
        "spec": ("x:" + d[["furniture_type", "width_cm", "depth_cm", "height_cm", "price_kzt"]]
                 .astype(object).astype(str).agg("|".join, axis=1)).values,
    }
    for ks in keys.values():
        first = {}
        for i, k in enumerate(ks):
            if k is None:
                continue
            if k in first:
                parent[find(i)] = find(first[k])
            else:
                first[k] = i
    roots = [find(i) for i in range(n)]
    return pd.Series(["g%04d" % r for r in roots], index=d.index, name="group")
