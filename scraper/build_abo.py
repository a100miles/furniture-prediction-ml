"""Build data/processed/abo_furniture.csv from the Amazon Berkeley Objects (ABO) listings.

ABO (Amazon.com, CC BY 4.0): https://amazon-berkeley-objects.s3.amazonaws.com/index.html
Input:  data/external/abo/listings/listings_*.json.gz, data/external/abo/images.csv.gz
Output: data/processed/abo_furniture.csv, one row per listing that maps to one of our nine types.

Run: .venv/bin/python scraper/build_abo.py
"""
import gzip
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ABO = ROOT / "data" / "external" / "abo"
OUT = ROOT / "data" / "processed" / "abo_furniture.csv"

# The type comes from the listing's category path (the store's own category, as with mebel.kz).
# Rules are checked in order; the first match wins. English plus the main other ABO marketplaces.
TYPE_RULES = [
    ("tv_stand", r"television stands|tv stands|tv & media furniture/(?!media storage)|tv-möbel|tv-lowboards|meubles tv|mobili porta tv|muebles para tv"),
    ("wardrobe", r"wardrobe|armoire|kleiderschr|armadi|armarios|garde-robe"),
    ("chest_of_drawers", r"dressers|chest of drawers|kommoden|commodes|cassettiere|cómodas|comodas|cajoneras"),
    ("coffee_table", r"coffee tables|kaffeetische|couchtische|tables basses|tavolini|mesas de centro|cocktail tables"),
    ("desk", r"desks|schreibtische|bureaux|scrivanie|escritorios"),
    ("bed", r"beds, frames & bases/beds|/bed frames$|/betten$|/lits$|/letti$|/camas$"),
    ("sofa", r"sofas|couches|divani|canapés|sofás|loveseats|futons|sectional"),
    ("armchair", r"living room furniture/chairs|accent chairs|armchairs|sessel|poltrone|fauteuils|sillones|recliners"),
    ("chair", r"dining room furniture/chairs|esszimmerstühle|dining chairs|desk chairs|bürostühle|sedie|chaises|sillas|home office chairs"),
]
# Paths that are about accessories, outdoor, pets or children, or about other furniture, are skipped.
EXCLUDE_PATH = (r"headboard|cover|bean bag|patio|outdoor|garden|\b(dogs?|cats?|pets?|kids?|baby|doll|toys?)\b|kids'|"
                r"swatch|cushion|slipcover|parts|hardware|ottoman|stool|nightstand|bedside|end tables|console|"
                r"storage & organi|organi[sz]er|aparadores|sideboard")
# Amazon's product type must agree with the category, otherwise the listing is an accessory filed there
# (door handles under wardrobes, TV wall mounts under TV stands, pet beds under beds, ...).
ALLOWED_PRODUCT_TYPES = {
    "sofa": {"SOFA", "HOME_FURNITURE_AND_DECOR", "FURNITURE", "HOME"},
    "armchair": {"CHAIR", "HOME_FURNITURE_AND_DECOR", "FURNITURE"},
    "chair": {"CHAIR", "HOME_FURNITURE_AND_DECOR", "FURNITURE"},
    "bed": {"BED", "BED_FRAME", "HOME_FURNITURE_AND_DECOR", "FURNITURE"},
    "wardrobe": {"CABINET", "HOME_FURNITURE_AND_DECOR", "FURNITURE", "HOME"},
    "chest_of_drawers": {"DRESSER", "CABINET", "HOME_FURNITURE_AND_DECOR", "FURNITURE"},
    "desk": {"DESK", "TABLE", "HOME_FURNITURE_AND_DECOR", "FURNITURE"},
    "coffee_table": {"TABLE", "HOME_FURNITURE_AND_DECOR", "FURNITURE"},
    "tv_stand": {"TABLE", "CABINET", "SHELF", "HOME_FURNITURE_AND_DECOR", "FURNITURE", "HOME"},
}
# Same tokens as the mebel.kz table (scraper/build_dataset.py), matched on the English material text.
MATERIAL_TOKENS = {
    # Amazon calls chipboard and similar boards "engineered wood"; mebel.kz says ЛДСП/ДСП
    "chipboard": r"particle ?board|chipboard|engineered wood|manufactured wood|composite wood|\bmfc\b",
    "mdf": r"\bmdf\b|medium[- ]density",
    "fibreboard": r"fib(re|er) ?board|hardboard|\bhdf\b",
    "solid_wood": r"solid wood|hardwood|\bwood\b|wooden|oak|pine|beech|birch|walnut|acacia|rubberwood|mango|teak|ash\b",
    "plywood": r"plywood",
    "veneer": r"veneer",
    "metal": r"metal|steel|iron|chrome|alumin",
    "plastic": r"plastic|polypropylene|\bpp\b|acrylic|resin",
    "glass": r"glass",
    "mirror": r"mirror",
    "fabric": r"fabric|textile|velvet|linen|polyester|chenille|boucle|bouclé|microfib|cotton|tweed|upholster",
    "leather": r"leather",
    "foam": r"foam",
    "spring": r"spring",
    "rattan": r"rattan|wicker",
    "stone": r"stone|marble|ceramic|granite",
}
COLOUR_MAP = {
    "white": "white", "black": "black", "grey": "grey", "gray": "grey", "beige": "beige", "brown": "brown",
    "green": "green", "blue": "blue", "pink": "pink", "red": "red", "yellow": "yellow", "orange": "orange",
    "purple": "purple",
}
TO_CM = {"inches": 2.54, "centimeters": 1.0, "millimeters": 0.1, "feet": 30.48, "meters": 100.0}
TO_KG = {"pounds": 0.4536, "kilograms": 1.0, "grams": 0.001, "ounces": 0.02835}


def english(values):
    """First English value of a multilingual ABO field, else the first value."""
    if not values:
        return None
    en = [v.get("value") for v in values if str(v.get("language_tag", "")).startswith("en")]
    return (en or [values[0].get("value")])[0]


def size_cm(dims, key):
    d = (dims or {}).get(key)
    if not d:
        return None
    nv = d.get("normalized_value") or d
    f = TO_CM.get(nv.get("unit"))
    return round(nv["value"] * f, 1) if f and nv.get("value") else None


def weight_kg(w):
    if not w:
        return None
    nv = w[0].get("normalized_value") or w[0]
    f = TO_KG.get(nv.get("unit"))
    return round(nv["value"] * f, 2) if f and nv.get("value") else None


def furniture_type(paths, product_type):
    for p in paths:
        if re.search(EXCLUDE_PATH, p):
            continue
        for name, rx in TYPE_RULES:
            if re.search(rx, p):
                return (name, p) if product_type in ALLOWED_PRODUCT_TYPES[name] else (None, p)
    return None, None


def colour(values):
    for v in values or []:
        for s in v.get("standardized_values") or []:
            c = COLOUR_MAP.get(str(s).lower())
            if c:
                return c
    return "other" if values else None


def build():
    rows = []
    for f in sorted((ABO / "listings").glob("listings_*.json.gz")):
        for line in gzip.open(f, "rt", encoding="utf-8"):
            d = json.loads(line)
            paths = [(x.get("node_name") or "").lower() for x in d.get("node") or []]
            ptype = (d.get("product_type") or [{}])[0].get("value")
            t, path = furniture_type(paths, ptype)
            if t is None:
                continue
            dims = d.get("item_dimensions")
            mat_text = " ".join(filter(None, [english(d.get("material")), english(d.get("fabric_type"))])).lower()
            rows.append({
                "item_id": d["item_id"],
                "domain": d.get("domain_name"),
                "furniture_type": t,
                "category_path": path,
                "product_type": ptype,
                "name": english(d.get("item_name")),           # shown only to explain, never a model input
                "brand": english(d.get("brand")),
                "model_number": (d.get("model_number") or [{}])[0].get("value"),
                "width_cm": size_cm(dims, "width"),
                "depth_cm": size_cm(dims, "length"),
                "height_cm": size_cm(dims, "height"),
                "weight_kg": weight_kg(d.get("item_weight")),
                "materials": "|".join(sorted(k for k, rx in MATERIAL_TOKENS.items() if re.search(rx, mat_text))),
                "material_text": mat_text or None,
                "main_color": colour(d.get("color")),
                "main_image_id": d.get("main_image_id"),
                "other_image_ids": "|".join(d.get("other_image_id") or []),
            })
    df = pd.DataFrame(rows)
    images = pd.read_csv(ABO / "images.csv.gz").set_index("image_id")["path"]
    df["main_image_path"] = df["main_image_id"].map(images)
    df.to_csv(OUT, index=False)
    print(f"{len(df)} listings -> {OUT.relative_to(ROOT)}")
    print(df["furniture_type"].value_counts().to_string())
    return df


if __name__ == "__main__":
    build()
