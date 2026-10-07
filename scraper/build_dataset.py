"""Turn the raw scraped records into one clean table for modelling.

Input : data/raw/mebel_raw.jsonl   (one JSON object per product page)
Output: data/processed/furniture.csv

One output row = one product listing on mebel.kz.
Run from the project root:  python scraper/build_dataset.py
"""
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "mebel_raw.jsonl"
OUT_PATH = ROOT / "data" / "processed" / "furniture.csv"

# Target label from the site's own category (JSON-LD "category"). Exact names,
# taken from the categories that appeared in the collected pages. Every other
# category is dropped: sofa modules ("Модули"), office and computer chairs,
# nightstands, kids' low cabinets, garden tables, lamps, mattress toppers, ...
CATEGORY_TO_TYPE = {
    "Прямые диваны": "sofa",
    "Модульные диваны": "sofa",
    "Угловые диваны": "sofa",
    "Садовые диваны": "sofa",
    "Офисные диваны": "sofa",
    "Кушетки": "sofa",              # on mebel.kz these are straight sofas
    "Кресла": "armchair",
    "Садовые кресла": "armchair",
    "Стулья для кухни": "chair",
    "Садовые стулья": "chair",
    "Стулья детские": "chair",
    "Кровати": "bed",
    "Детские кровати": "bed",
    "Шкафы распашные": "wardrobe",
    "Шкафы-купе": "wardrobe",
    "Детские шкафы": "wardrobe",
    "Угловые шкафы": "wardrobe",
    "Комоды": "chest_of_drawers",
    "Детские комоды": "chest_of_drawers",
    "Столы компьютерные и письменные": "desk",
    "Детские столы": "desk",
    "Журнальные столики": "coffee_table",
    "Тумбы под телевизор": "tv_stand",
}

# Material words normalised to short tokens. Only the *values* of material
# fields are used, never the field names (see leakage check in the notebook).
MATERIAL_TOKENS = {
    "chipboard": r"лдсп|дсп",
    "mdf": r"мдф",
    "fibreboard": r"двп|лхдф|хдф",
    "solid_wood": r"массив|мебельный щит|бук|берез|сосн|дуб",
    "plywood": r"фанер",
    "veneer": r"шпон",
    "metal": r"метал|сталь|хром",
    "plastic": r"пластик|полипропилен",
    "glass": r"стекл",
    "mirror": r"зеркал",
    "fabric": r"ткан|текстил|велюр|шенил|рогожк|букле|микрофибр",
    "leather": r"кожа|экокож",
    "foam": r"ппу|пенополиуретан",
    "spring": r"пружин",
    "rattan": r"ротанг",
    "stone": r"камен|керамик|мрамор",
}
MATERIAL_FIELD = re.compile(r"^(материал|каркас)", re.I)

COLOR_MAP = {
    "белый": "white", "черный": "black", "серый": "grey", "бежевый": "beige",
    "коричневый": "brown", "зеленый": "green", "синий": "blue", "голубой": "light_blue",
    "розовый": "pink", "красный": "red", "желтый": "yellow", "оранжевый": "orange",
    "фиолетовый": "purple", "бордовый": "burgundy", "терракотовый": "terracotta",
    "оливковый": "olive", "графит": "grey", "мятный": "mint",
}


def to_float(text):
    if text is None:
        return None
    m = re.search(r"\d+(?:[.,]\d+)?", str(text))
    return float(m.group().replace(",", ".")) if m else None


def label_for(category):
    return CATEGORY_TO_TYPE.get((category or "").strip())


def materials_of(specs):
    text = " ".join(v for k, v in specs.items() if MATERIAL_FIELD.match(k)).lower()
    return sorted(t for t, p in MATERIAL_TOKENS.items() if re.search(p, text))


def main_color(specs, product):
    raw = specs.get("Цвет") or specs.get("Цвет корпуса") or product.get("color") or ""
    first = raw.split(",")[0].strip().lower().replace("ё", "е")
    return COLOR_MAP.get(first, "other" if first else None)


def build():
    rows = []
    with RAW_PATH.open() as f:
        for line in f:
            rec = json.loads(line)
            product = rec.get("product") or {}
            specs = rec.get("specs") or {}
            dims = rec.get("dimensions") or {}
            rows.append({
                "url": rec["url"],
                "scraped_at": rec["scraped_at"][:10],
                "sku": product.get("sku"),
                "name": product.get("name"),
                "site_category": product.get("category"),
                "series": specs.get("Серия"),
                "furniture_type": label_for(product.get("category")),
                "width_cm": to_float(dims.get("Длина")),
                "depth_cm": to_float(dims.get("Ширина") or dims.get("Глубина")),
                "height_cm": to_float(dims.get("Высота")),
                "price_kzt": to_float(product.get("price")),
                "materials": "|".join(materials_of(specs)),
                "main_color": main_color(specs, product),
                "n_spec_fields": len(specs),
            })
    df = pd.DataFrame(rows)
    print(f"raw records: {len(df)}")
    df = df.drop_duplicates("url")
    unlabelled = df["furniture_type"].isna()
    print(f"dropped, category not one of the nine types: {unlabelled.sum()}")
    print("  most common dropped categories:",
          df.loc[unlabelled, "site_category"].value_counts().head(12).to_dict())
    df = df[~unlabelled]
    incomplete = df[["width_cm", "depth_cm", "height_cm", "price_kzt"]].isna().any(axis=1)
    print(f"dropped, missing a dimension or price: {incomplete.sum()}")
    df = df[~incomplete]
    # Listings without a series name are their own group.
    df["series"] = df["series"].fillna(df["sku"].astype(str))
    df.to_csv(OUT_PATH, index=False)
    print(f"saved {len(df)} rows to {OUT_PATH.relative_to(ROOT)}")
    print(df["furniture_type"].value_counts().to_string())


if __name__ == "__main__":
    build()
