"""Amazon Berkeley Objects (ABO) furniture table: loading, grouping and specification features.

Sizes are compared orientation-free (height, longer and shorter horizontal side), because ABO's
width/length order is not consistent and mebel.kz uses its own convention (bed "width" = length).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MATERIAL_TOKENS = ["chipboard", "mdf", "fibreboard", "solid_wood", "plywood", "veneer", "metal", "plastic", "glass",
                   "mirror", "fabric", "leather", "foam", "spring", "rattan", "stone"]
MATERIALS = ["mat_" + m for m in MATERIAL_TOKENS]
SIZE = ["height_cm", "long_cm", "short_cm", "footprint_m2", "volume_m3", "height_to_long", "short_to_long"]
WEIGHT = ["log_weight_kg", "log_kg_per_m3"]


def size_features(d):
    """Orientation-free size inputs, the same for both marketplaces."""
    d = d.copy()
    w, dp = d["width_cm"], d["depth_cm"]
    d["long_cm"] = np.fmax(w, dp)
    d["short_cm"] = np.fmin(w, dp)
    d["footprint_m2"] = d["long_cm"] * d["short_cm"] / 1e4
    d["volume_m3"] = d["footprint_m2"] * d["height_cm"] / 100
    d["height_to_long"] = d["height_cm"] / d["long_cm"]
    d["short_to_long"] = d["short_cm"] / d["long_cm"]
    mats = d["materials"].fillna("").str.get_dummies(sep="|").add_prefix("mat_")
    d = pd.concat([d.drop(columns=[c for c in d.columns if c.startswith("mat_")]), mats.reindex(columns=MATERIALS, fill_value=0)], axis=1)
    d["main_color"] = d["main_color"].fillna("unknown")
    return d


def load_abo():
    d = pd.read_csv(ROOT / "data" / "processed" / "abo_furniture.csv")
    # implausible sizes (0 or above 5 m, e.g. a unit mix-up) are treated as missing
    for c in ["width_cm", "depth_cm", "height_cm"]:
        d.loc[(d[c] < 5) | (d[c] > 500), c] = np.nan
    d.loc[(d["weight_kg"] <= 0) | (d["weight_kg"] > 500), "weight_kg"] = np.nan
    d = size_features(d)
    d["log_weight_kg"] = np.log(d["weight_kg"])
    d["log_kg_per_m3"] = np.log(d["weight_kg"] / d["volume_m3"])
    d["group"] = abo_groups(d)
    return d.reset_index(drop=True)


GENERIC_PHOTO = 5


def generic_photos(d):
    imgs = d[["item_id", "main_image_id", "other_image_ids"]].copy()
    imgs["all"] = imgs["main_image_id"].fillna("") + "|" + imgs["other_image_ids"].fillna("")
    long = imgs.assign(img=imgs["all"].str.split("|")).explode("img")
    long = long[long["img"] != ""]
    counts = long.groupby("img")["item_id"].nunique()
    return set(counts[counts > GENERIC_PHOTO].index)


def abo_groups(d):
    """One product listed on several marketplaces is one group: listings are joined when they share the
    item id, the model number, the main photo, or a product-specific other photo (union-find, joins chain).
    Generic photos shared by more than GENERIC_PHOTO listings (brand logos, fabric swatches, room scenes)
    are not used for joining; `photo_views` also leaves them out of the photo models."""
    n = len(d)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    generic = generic_photos(d)
    first = {}
    for i, r in enumerate(d.itertuples()):
        keys = ["i:" + str(r.item_id)]
        if isinstance(r.model_number, str) and len(r.model_number.strip()) > 2:
            keys.append("m:" + r.model_number.strip().lower())
        imgs = [r.main_image_id] + (r.other_image_ids.split("|") if isinstance(r.other_image_ids, str) else [])
        keys += ["p:" + x for x in imgs if isinstance(x, str) and x not in generic]
        for k in keys:
            if k in first:
                parent[find(i)] = find(first[k])
            else:
                first[k] = i
    return pd.Series(["a%05d" % find(i) for i in range(n)], index=d.index)


def spec_prep(numeric, materials=MATERIALS, colour=True):
    parts = [("numeric", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                                   ("scale", StandardScaler())]), numeric),
             ("materials", "passthrough", materials)]
    if colour:
        parts.append(("colour", OneHotEncoder(handle_unknown="ignore", min_frequency=10), ["main_color"]))
    return ColumnTransformer(parts)
