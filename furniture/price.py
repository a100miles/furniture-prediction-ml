"""Price prediction on the mebel.kz listings: inputs, name text and the models compared in final.ipynb, section 7.

Target: log(price in KZT). Inputs never include the price itself (`log_price`, `log_price_per_m3`), the
series, the article number or the URL. The product name is used as text, either as it is or with the
design's own name masked (`mask_name`), to check whether the text only memorises the price of a design.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from furniture.features import model_word

SIZE_NUM = ["width_cm", "depth_cm", "height_cm", "footprint_m2", "volume_m3", "height_to_width", "depth_to_width"]
CATEGORICAL = ["furniture_type", "main_color"]
E5_MODEL = "intfloat/multilingual-e5-small"
ROOT = Path(__file__).resolve().parents[1]
E5_CACHE = ROOT / "data" / "cache"
MASK = "модель"


def mask_name(name, series=""):
    """Remove the design's own name from the product name, keep the rest (type words, size, fabric, colour,
    mechanism). 'Диван прямой Линс Велюр Голубой' -> 'Диван прямой модель Велюр Голубой'.
    Removed: the model word (`model_word`) and its variants such as 'Хэйко-1', and every word of the series."""
    words = str(name).replace("​", "").split()
    mw = model_word(name).lower()
    series_words = {w.lower() for w in re.findall(r"\w+", str(series)) if not w.isdigit()}
    out = []
    for i, w in enumerate(words):
        low = w.lower()
        core = re.split(r"[-\d]", low)[0]
        if i > 0 and ((mw and (low.startswith(mw) or core == mw)) or core in series_words or low in series_words):
            if not out or out[-1] != MASK:
                out.append(MASK)
        else:
            out.append(w)
    return " ".join(out)


def text_column(d, masked):
    if masked:
        return [mask_name(n, s) for n, s in zip(d["name"], d["series"])]
    return d["name"].astype(str).str.replace("​", "", regex=False).tolist()


def spec_prep(materials):
    return ColumnTransformer([
        ("numeric", StandardScaler(), SIZE_NUM),
        ("materials", "passthrough", materials),
        ("categorical", OneHotEncoder(handle_unknown="ignore", min_frequency=5), CATEGORICAL),
    ])


def tfidf():
    """Words (1-2) plus character 2-5-grams inside words, so 'Велюр' / 'велюровый' share pieces."""
    return FeatureUnion([
        ("words", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("chars", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True)),
    ])


def specs_text_prep(materials, text_col="text"):
    return ColumnTransformer([
        ("specs", spec_prep(materials), SIZE_NUM + materials + CATEGORICAL),
        ("text", tfidf(), text_col),
    ])


def specs_embedding_prep(materials, emb_cols):
    return ColumnTransformer([
        ("specs", spec_prep(materials), SIZE_NUM + materials + CATEGORICAL),
        ("emb", StandardScaler(), emb_cols),
    ])


def e5_embed(texts, cache_name=None, batch_size=64):
    """Sentence embeddings from multilingual-e5-small (pretrained, not fine-tuned). Cached by text list."""
    from sentence_transformers import SentenceTransformer
    cache = E5_CACHE / f"{cache_name}.npz" if cache_name else None
    if cache is not None and cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["texts"]) == list(texts):
            return z["emb"]
    model = SentenceTransformer(E5_MODEL, device="cpu")
    emb = model.encode(["query: " + t for t in texts], batch_size=batch_size, normalize_embeddings=True,
                       show_progress_bar=False)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, texts=np.array(texts, dtype=object), emb=emb)
    return emb


def app_inputs(d, materials, e5_columns=None):
    """Turn raw rows (sizes, `materials` text, colour, type, name) into the price model's input columns.
    Same derived sizes as `add_features`, without anything computed from the price."""
    d = d.copy()
    d["main_color"] = d["main_color"].fillna("unknown")
    d["footprint_m2"] = d["width_cm"] * d["depth_cm"] / 1e4
    d["volume_m3"] = d["width_cm"] * d["depth_cm"] * d["height_cm"] / 1e6
    d["height_to_width"] = d["height_cm"] / d["width_cm"]
    d["depth_to_width"] = d["depth_cm"] / d["width_cm"]
    mats = d["materials"].fillna("").str.get_dummies(sep="|").add_prefix("mat_").reindex(columns=materials, fill_value=0)
    d = pd.concat([d.drop(columns=[c for c in materials if c in d.columns]), mats], axis=1)
    d["text"] = d["name"].fillna("").astype(str).str.replace("​", "", regex=False)
    if e5_columns:
        emb = e5_embed(d["text"].tolist())
        d = pd.concat([d, pd.DataFrame(emb, columns=e5_columns, index=d.index)], axis=1)
    return d


def price_metrics(y_log, pred_log, weight=None):
    y, p = np.exp(y_log), np.exp(pred_log)
    w = np.ones_like(y) if weight is None else np.asarray(weight)
    err = np.abs(p - y)
    r2 = 1 - np.sum(w * (y_log - pred_log) ** 2) / np.sum(w * (y_log - np.average(y_log, weights=w)) ** 2)
    return {"MAE (KZT)": np.average(err, weights=w), "MAPE": np.average(err / y, weights=w),
            "median APE": float(np.median(err / y)), "R² (log)": r2}
