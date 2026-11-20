# Furniture type from specifications and photos - Final

Course project, Machine Learning Algorithms, Direction 21 "Furniture" (classification).
Team: Sultanbek Zhumagali, Tamerlan Yessimov (SE-2420).

**Question:** can the type of a furniture item (sofa, armchair, chair, bed, wardrobe, chest of drawers, desk, coffee table, TV stand) be predicted from its sizes, materials, colour and price - and how much of that holds up under an honest test, in another store, and with photos?

The Final answers the Endterm feedback first (colour variants across the split, per-class metrics for the small classes, photo terms), then adds a second marketplace with openly licensed photos (Amazon Berkeley Objects), photo models (a small CNN and a pretrained ResNet-18), a combined model, a cross-store test and a local app.

## What is where

| Path | What it is |
|---|---|
| `final.ipynb` | **The Final notebook**, saved with all outputs. 1 regrouping, 2 per-class metrics, 3 photo terms, 4 summary of the feedback, 5 ABO: specs vs photo vs both, 6 mebel.kz model in another store, 7 saved models. |
| `frontend/`, `api.py` | Local React/TypeScript app (exact dependency versions in `package.json` + `package-lock.json`) with a Python model API: type from specifications (mebel.kz model) or from a photo (+ optional specs, ABO model). The decorative hero photo `frontend/public/hero-chair.jpg` is "Brown wooden chair on black wooden floor" by Sergio Camalich, Unsplash License (https://unsplash.com/photos/MoZX1sNVNsg); it is never used by a model. |
| `app.py` | Legacy Streamlit version retained as a fallback. |
| `furniture/` | Shared code: `features.py` (mebel.kz inputs and grouping), `abo.py` (ABO table, grouping, inputs), `photos.py` (image loading, ResNet-18 features, small CNN). |
| `tools/build_final_notebook.py` | Writes `final.ipynb` (cells); `tools/abo_experiment.py` runs the ABO cross-validation; `tools/cross_market.py` is the stand-alone cross-store check. |
| `scraper/build_abo.py`, `scraper/download_abo_images.py` | Build the ABO furniture table and download its photos. |
| `models/` | `structured_best.joblib` (SVM, mebel.kz), `abo_photo.joblib`, `abo_specs_photo.joblib`. Written by `final.ipynb` section 7. |
| `data/processed/` | `furniture.csv` (mebel.kz), `test_urls.csv` (the 793 fixed test listings), `abo_furniture.csv`, `abo_images.csv`. |
| `data/external/abo/` | ABO metadata and photos (not in the submission archive; re-download with the scraper scripts). |
| `data/cache/` | ResNet-18 features and the cross-validated ABO predictions (re-created when deleted). |
| `CHANGES.md`, `DATA_CARD.md`, `CONTRIBUTIONS.md` | Change log, data sources and licences, who did what. |
| `endterm.ipynb`, `furniture_classification.ipynb` | Endterm and Midterm notebooks, unchanged. |

## Run

Python 3.12+ and [uv](https://docs.astral.sh/uv/) (or plain `pip`).

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/python scraper/build_abo.py
.venv/bin/python scraper/download_abo_images.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace final.ipynb
cd frontend
npm ci
npm run build
cd ..
.venv/bin/python api.py
```

On Windows, replace `.venv/bin/python` with `.venv/Scripts/python.exe`, then open `http://127.0.0.1:8000`.

* The ABO metadata must be in `data/external/abo/` first (listings and `images.csv.gz`, see `DATA_CARD.md`).
* `final.ipynb` takes about 10 minutes when `data/cache/` is present. Without it, the ResNet-18 features (about 10 minutes) and the ABO cross-validation with the small CNN (about 40 minutes, CPU) are computed again.
* Every random step uses seed 42.
