# Furniture listings: type, then price, from specifications and a second source - Final

Course project, Machine Learning Algorithms, Direction 21 "Furniture".
Team: Sultanbek Zhumagali, Tamerlan Yessimov (SE-2420).

**Main question (Final):** how well can the price of a mebel.kz furniture listing be predicted from its specifications (sizes, materials, colour, type), and does a second source on the same listings, the product name, add what the specifications miss?

**Earlier question, now a finding:** the type (sofa, armchair, chair, bed, wardrobe, chest of drawers, desk, coffee table, TV stand) can be predicted from the specifications with grouped-CV macro-F1 about 0.95, and from the three sizes alone with test macro-F1 0.918. The teacher's message of 20 November pointed this out and suggested price on the same listings and test set; `final.ipynb` section 4.1 explains the change.

## Results in one table

| | Grouped 5-fold CV | Test (793 listings, 92 designs) |
|---|---|---|
| Price, median of the type | MAE 94,800 KZT | MAE 97,900 KZT |
| Price, specifications only (LightGBM; random forest: 61,400 / 54,300) | MAE 60,600 KZT, MAPE 24.9% | MAE 53,700 KZT, MAPE 22.1% |
| **Price, specifications + name (TF-IDF, SVR), chosen** | **MAE 59,100 KZT, MAPE 23.0%** | **MAE 51,700 KZT, MAPE 20.6%** |
| Price, specifications + name (pretrained multilingual-e5-small, SVR) | MAE 60,200 KZT | MAE 54,200 KZT |
| Type, specifications (SVM) | macro-F1 0.949 ± 0.019 | 0.974 per listing, 0.940 per design |
| Type, specifications without price (SVM, used by the app when no price is given) | macro-F1 0.950 ± 0.021 | 0.971 |
| Type, three sizes only (random forest) | | 0.918 |
| Type on Amazon (ABO), specs / photo (ResNet-18) / both | 0.662 / 0.766 / 0.794 | (CV only) |
| Type on Amazon, stronger photo models: fine-tuned ResNet-18 / CLIP / **DINOv2** / **DINOv2 + specs** | 0.789 / 0.801 / **0.844** / **0.851** | (CV only) |
| Same photo models on 221 IKEA catalogue photos never seen in training: ResNet-18 / CLIP / fine-tuned / **DINOv2** | | macro-F1 0.755 / 0.648 / 0.767 / **0.880** |
| Price in a second market (IKEA Saudi Arabia, 620 items): best specs-only (ridge) vs specs + text (SVR) | MAPE 46.6% vs **39.9%** | (CV only) |

## What is where

| Path | What it is |
|---|---|
| `final.ipynb` | **The Final notebook**, saved with all outputs. Existing solutions · 1 regrouping · 2 per-class metrics · 3 data sources, rights, matching, missing inputs · 4 feedback and the new formulation · 5 ABO photos: specs vs photo vs both, stronger encoders (5.6), IKEA outside test (5.7) · 6 type model in another store · **7 price: specs vs specs + name, masking ablation, run-to-run variation, errors, second market IKEA SA (7.4)** · 8 saved models · 9 conclusions. |
| `frontend/`, `api.py` | Local React/TypeScript app (exact dependency versions in `package.json` + `package-lock.json`) with a Python model API. Three modes: type from specifications, with or without the price (mebel.kz), type from a photo (+ optional specs; DINOv2, trained on ABO), price from specifications (+ optional product name, mebel.kz). Results are given in plain words (price ± typical error, the range of 8 in 10 similar listings, a score out of 100); a **How it works** page explains the aim, data, training, models and limits. Inputs are checked in the browser and again in the API (HTTP 422 with a message); a name with too few known words, or naming another type, is not used for the price (`furniture/price.py: check_name`). |
| `app.py` | Legacy Streamlit version (type models only), kept as a fallback. |
| `furniture/` | Shared code: `features.py` (mebel.kz inputs and grouping), `price.py` (price inputs, name masking, TF-IDF, e5 embeddings, metrics), `abo.py` (ABO table, grouping, inputs), `photos.py` (image loading, ResNet-18 / CLIP / DINOv2 features, small CNN, fine-tuned ResNet-18), `ikea.py` (IKEA SA price inputs). |
| `tools/build_final_notebook.py` | Writes `final.ipynb` (cells). `tools/abo_experiment.py` runs the ABO cross-validation; `tools/abo_encoders_experiment.py` adds CLIP, DINOv2, the fine-tuned ResNet-18 and the IKEA photo test on the same folds; `tools/price_experiment.py` is a stand-alone version of the price comparison; `tools/cross_market.py` is the stand-alone cross-store check; `tools/make_archive.py` builds the submission ZIP. |
| `scraper/` | mebel.kz collection (Midterm), `build_abo.py` and `download_abo_images.py` for ABO, `build_ikea.py` for the two IKEA sources. |
| `models/` | `structured_best.joblib` (type, SVM, mebel.kz), `abo_photo.joblib`, `abo_specs_photo.joblib` (ResNet-18), `abo_photo_dinov2.joblib`, `abo_specs_photo_dinov2.joblib` (DINOv2), `structured_noprice.joblib` (type without price, SVM), `price_best.joblib` (price: best specs-only model, LightGBM, and the specs + name model). Written by `final.ipynb` section 8. |
| `data/processed/` | `furniture.csv` (mebel.kz), `test_urls.csv` (the 793 fixed test listings), `abo_furniture.csv`, `abo_images.csv`, `ikea_sa.csv` (IKEA SA table), `ikea_images.csv` (IKEA photo manifest). |
| `data/external/` | ABO image index, licence and photos; IKEA SA raw file; IKEA photos (none in the submission archive; re-download with the scraper scripts). |
| `data/cache/` | ResNet-18, CLIP and DINOv2 features, the cross-validated ABO predictions and the name embeddings (re-created when deleted). |
| `CHANGES.md`, `DATA_CARD.md`, `CONTRIBUTIONS.md` | Change log, data sources and licences, who did what and AI use. |
| `Furniture Final Slides.pdf`, `slides/` | Final slides (13, about 10 minutes with the demo). `slides/Furniture Final Slides.html` opens in a browser (arrow keys); `slides/speaker-notes.md` has who says what; `slides/final/` holds the slide sources, rebuilt with `tools/build_slides.py --pdf`. |
| `endterm.ipynb`, `furniture_classification.ipynb` | Endterm and Midterm notebooks, unchanged. |

## Run

Python 3.12+ and [uv](https://docs.astral.sh/uv/) (or plain `pip`), Node.js 20+ for the frontend.

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
# ABO image index, then the photos (the processed table data/processed/abo_furniture.csv is included)
mkdir -p data/external/abo
curl -L -o data/external/abo/images.csv.gz https://amazon-berkeley-objects.s3.amazonaws.com/images/metadata/images.csv.gz
.venv/bin/python scraper/download_abo_images.py
# IKEA SA table and the labelled IKEA photos (about 1 minute)
.venv/bin/python scraper/build_ikea.py
.venv/bin/jupyter nbconvert --to notebook --execute --inplace final.ipynb
cd frontend
npm ci
npm run build
cd ..
.venv/bin/python api.py
```

On Windows, replace `.venv/bin/python` with `.venv/Scripts/python.exe`, then open `http://127.0.0.1:8000`.

* Online demo: https://furniture-model-studio.onrender.com (Render free plan, `render.yaml`; it sleeps when unused, so the first request can take about a minute). Its 512 MB of memory is not enough for DINOv2, so it sets `PHOTO_ENCODER=resnet18` and answers photos with the earlier ResNet-18 models; the local app uses DINOv2. Each answer names its model.

* To rebuild `abo_furniture.csv` from scratch, download ABO's 16 listing files to `data/external/abo/listings/` first and run `scraper/build_abo.py` (see `DATA_CARD.md`).
* `final.ipynb` takes about 15 minutes when `data/cache/` is present. Without it, the ResNet-18 features (about 10 minutes), the ABO cross-validation with the small CNN (about 40 minutes, CPU) and the name embeddings (2 minutes, plus a one-time ~470 MB model download from Hugging Face) are computed again. The stronger photo models of section 5.6 (`tools/abo_encoders_experiment.py`: CLIP and DINOv2 features about 20 minutes, one-time downloads of ~600 MB and ~90 MB; fine-tuning ResNet-18 about 50 minutes) are re-created from `data/cache/abo_oof_v2.npz` when it is present.
* Every random step uses seed 42. Section 7.2 reports how much the CV scores move with other fold cuts.

## Open question for the teacher

The team asks the teacher to confirm two points: (1) is ABO acceptable as a secondary image source, given that mebel.kz photos may not be reused; (2) are product names acceptable as the text source for the price model? Page descriptions and photos of mebel.kz are not used.
