"""Local demo app for the Final: predict the furniture type from specifications and/or a photo.

Run:  .venv/bin/streamlit run app.py
Needs the models saved by final.ipynb (section 7): models/structured_best.joblib, models/abo_photo.joblib,
models/abo_specs_photo.joblib. The photo models use ResNet-18 weights from torchvision (downloaded once).
"""
import io

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from furniture.abo import MATERIAL_TOKENS, size_features
from furniture.features import TYPES, add_features
from furniture.photos import resnet18_features

st.set_page_config(page_title="Furniture type", page_icon="🪑", layout="centered")
COLOURS = ["white", "black", "grey", "beige", "brown", "green", "blue", "light_blue", "pink", "red", "yellow",
           "orange", "purple", "burgundy", "terracotta", "olive", "mint", "other"]


@st.cache_resource
def load_models():
    return (joblib.load("models/structured_best.joblib"), joblib.load("models/abo_photo.joblib"),
            joblib.load("models/abo_specs_photo.joblib"))


def show_probabilities(classes, proba):
    s = pd.Series(proba, index=classes).reindex(TYPES).fillna(0).sort_values(ascending=False)
    st.metric("Predicted type", s.index[0].replace("_", " "), f"{s.iloc[0]:.0%} probability")
    st.bar_chart(s.rename("probability"))
    if s.iloc[0] < 0.6:
        st.info("The model is unsure: the top two types are close. Check the listing by hand.")


def size_inputs(prefix, price=True, weight=False):
    c1, c2, c3 = st.columns(3)
    w = c1.number_input("Width, cm", 10.0, 500.0, 120.0, key=prefix + "w")
    d = c2.number_input("Depth, cm", 10.0, 500.0, 45.0, key=prefix + "d")
    h = c3.number_input("Height, cm", 10.0, 500.0, 75.0, key=prefix + "h")
    out = {"width_cm": w, "depth_cm": d, "height_cm": h}
    if price:
        out["price_kzt"] = st.number_input("Price, KZT", 5000.0, 10_000_000.0, 150_000.0, step=5000.0, key=prefix + "p")
    if weight:
        out["weight_kg"] = st.number_input("Weight, kg", 0.5, 500.0, 30.0, key=prefix + "kg")
    out["materials"] = "|".join(st.multiselect("Materials", MATERIAL_TOKENS, ["chipboard"], key=prefix + "m"))
    out["main_color"] = st.selectbox("Main colour", COLOURS, key=prefix + "c")
    return out


st.title("Furniture type")
st.caption("Machine Learning Algorithms, Final project. Sultanbek Zhumagali, Tamerlan Yessimov (SE-2420).")
try:
    structured, photo_model, combined = load_models()
except FileNotFoundError:
    st.error("Models not found. Run final.ipynb first (section 7 saves them to models/).")
    st.stop()

tab_specs, tab_photo = st.tabs(["From specifications (mebel.kz)", "From a photo (Amazon / ABO)"])

with tab_specs:
    st.write(f"Model: **{structured['name']}**, trained on mebel.kz listings. "
             f"Grouped CV macro-F1 {structured['cv_macro_f1']:.3f} on mebel.kz; it does not transfer well to other "
             "stores (see the notebook, section 6).")
    row = pd.DataFrame([size_inputs("s")])
    if st.button("Predict", key="predict_specs"):
        X = add_features(row, materials=structured["materials"])[structured["features"]]
        pipe = structured["pipeline"]
        show_probabilities(pipe.classes_, pipe.predict_proba(X)[0])

with tab_photo:
    st.write("Models trained on Amazon Berkeley Objects listings (photos: Amazon.com, CC BY 4.0). "
             f"Photo only: CV macro-F1 {photo_model['cv_macro_f1']:.3f}; photo + specifications: "
             f"{combined['cv_macro_f1']:.3f}.")
    upload = st.file_uploader("Product photo (JPG or PNG, ideally on a plain background)", type=["jpg", "jpeg", "png"])
    use_specs = st.checkbox("Also use sizes, weight and materials")
    specs = size_inputs("p", price=False, weight=True) if use_specs else None
    if upload is not None:
        img = Image.open(io.BytesIO(upload.getvalue())).convert("RGB")
        st.image(img, width=240)
        side = max(img.size)
        sq = Image.new("RGB", (side, side), (255, 255, 255))
        sq.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
        x = torch.from_numpy(np.array(sq.resize((224, 224)))).permute(2, 0, 1).unsqueeze(0)
        emb = pd.DataFrame(resnet18_features(x), columns=photo_model["embedding_columns"])
        if specs is None:
            pipe = photo_model["pipeline"]
            show_probabilities(pipe.classes_, pipe.predict_proba(emb)[0])
        else:
            s = size_features(pd.DataFrame([specs]))
            s["log_weight_kg"] = np.log(s["weight_kg"])
            s["log_kg_per_m3"] = np.log(s["weight_kg"] / s["volume_m3"])
            X = pd.concat([s.reset_index(drop=True), emb], axis=1)
            pipe = combined["pipeline"]
            show_probabilities(pipe.classes_, pipe.predict_proba(X)[0])
