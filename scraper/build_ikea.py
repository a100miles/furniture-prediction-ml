"""IKEA sources (Final, after the survey): IKEA Saudi Arabia listings with price, and IKEA catalogue photos.

1. IKEA Saudi Arabia (ikea.com/sa, scraped 20 April 2020 by A. Kallam, Kaggle; copy used by TidyTuesday 2020-11-03):
   price in SAR, category, width/depth/height in cm, product line name, short description, designer.
   -> data/external/ikea_sa/ikea.csv (raw) and data/processed/ikea_sa.csv (our nine types, all three sizes).
2. IKEA catalogue photos (Tautkute et al. 2017, github.com/IvonaTau/ikea): "all images are property of IKEA.COM and
   are only allowed for non-commercial use". Only photos we can label are downloaded: the bed / chair / couch folders,
   and product photos whose IKEA article number is an IKEA SA item of one of our types.
   -> data/external/ikea_images/ and data/processed/ikea_images.csv (manifest).
   The repository's .p (pickle) label files are NOT downloaded or loaded: unpickling a downloaded file can run code.

Run: .venv/bin/python scraper/build_ikea.py
"""
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
SA_URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2020/2020-11-03/ikea.csv"
IMG_TREE = "https://api.github.com/repos/IvonaTau/ikea/git/trees/master?recursive=1"
IMG_RAW = "https://raw.githubusercontent.com/IvonaTau/ikea/master/"
SA_RAW = ROOT / "data" / "external" / "ikea_sa" / "ikea.csv"
IMG_DIR = ROOT / "data" / "external" / "ikea_images"
HEADERS = {"User-Agent": "furniture-course-project (university ML course, non-commercial)"}

# first part of the short description (before the first comma) -> our type; checked by reading every category
TYPE_RULES = [
    ("tv_stand", r"^tv bench"),
    ("chest_of_drawers", r"^(modular )?chest of \d+ drawers"),
    ("wardrobe", r"^(corner |open )?wardrobe"),
    ("coffee_table", r"^coffee table"),
    ("desk", r"^desk"),
    ("armchair", r"^armchair"),
    ("sofa", r"^(\d-seat |corner )?sofa(-bed)?$|^\d-seat sofa"),
    ("bed", r"^(ext |bunk |loft )?bed frame|^divan bed|^day-bed"),
    ("chair", r"^(swivel |office )?chair$|^chair with armrests$"),
]
FOLDER_TYPES = {"bed": "bed", "chair": "chair", "couch": "sofa"}


def ikea_type(desc):
    first = str(desc).strip().split(",")[0].strip().lower()
    for t, pat in TYPE_RULES:
        if re.search(pat, first):
            return t
    return None


def article(item_id):
    s = f"{int(item_id):08d}"
    return f"{s[:3]}.{s[3:6]}.{s[6:]}"


def build_sa():
    if not SA_RAW.exists():
        SA_RAW.parent.mkdir(parents=True, exist_ok=True)
        SA_RAW.write_bytes(requests.get(SA_URL, headers=HEADERS, timeout=60).content)
    d = pd.read_csv(SA_RAW, index_col=0).drop_duplicates("item_id")      # an item can be in several categories
    d["short_description"] = d["short_description"].str.split().str.join(" ")
    d["furniture_type"] = d["short_description"].map(ikea_type)
    n_typed = d["furniture_type"].notna().sum()
    d = d[d["furniture_type"].notna()].rename(columns={"width": "width_cm", "depth": "depth_cm", "height": "height_cm",
                                                         "price": "price_sar"})
    full = d[["width_cm", "depth_cm", "height_cm"]].notna().all(axis=1) & (d[["width_cm", "depth_cm", "height_cm"]] > 0).all(axis=1)
    print(f"IKEA SA: {d.shape[0]} of {n_typed} typed items; {full.sum()} with all three sizes")
    d = d[full].copy()
    # one group per product line (e.g. all HEMNES items): variants of a design share a line name and its price level
    d["line"] = d["name"].str.split("/").str[0].str.strip().str.upper()
    d["group"] = d.groupby("line").ngroup()
    d["text"] = d["name"].str.strip() + " " + d["short_description"]
    d["article"] = d["item_id"].map(article)
    cols = ["item_id", "article", "name", "line", "short_description", "designer", "category", "furniture_type",
            "price_sar", "width_cm", "depth_cm", "height_cm", "text", "group", "link"]
    out = ROOT / "data" / "processed" / "ikea_sa.csv"
    d[cols].to_csv(out, index=False)
    print(out, len(d), "items,", d["group"].nunique(), "product lines")
    print(d["furniture_type"].value_counts().to_string())
    return d


def build_images(sa):
    tree = requests.get(IMG_TREE, headers=HEADERS, timeout=60).json()
    assert not tree.get("truncated"), "repository listing truncated"
    want = []
    sa_type = dict(zip(sa["article"], sa["furniture_type"]))
    for e in tree["tree"]:
        m = re.match(r"images/([^/]+)/((\d{3})\.(\d{3})\.(\d{2}))[^/]*\.jpe?g$", e["path"], re.I)
        if e["type"] != "blob" or not m:
            continue
        folder, art = m.group(1), m.group(2)
        if folder in FOLDER_TYPES:
            want.append({"path": e["path"], "article": art, "furniture_type": FOLDER_TYPES[folder], "label_from": "folder"})
        elif art in sa_type:
            want.append({"path": e["path"], "article": art, "furniture_type": sa_type[art], "label_from": "IKEA SA item"})
    want = pd.DataFrame(want).drop_duplicates("path")
    ok = []
    for p in want["path"]:
        f = IMG_DIR / p
        if not f.exists():
            r = requests.get(IMG_RAW + requests.utils.quote(p), headers=HEADERS, timeout=60)
            if r.status_code == 200:
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(r.content)
            time.sleep(0.2)
        ok.append(f.exists())
    want["ok"] = ok
    want["in_ikea_sa"] = want["article"].isin(sa_type)
    out = ROOT / "data" / "processed" / "ikea_images.csv"
    want.to_csv(out, index=False)
    print(out, f"{want['ok'].sum()} of {len(want)} photos downloaded")
    print(pd.crosstab(want["furniture_type"], want["label_from"]).to_string())


if __name__ == "__main__":
    np.random.seed(42)
    build_images(build_sa())
