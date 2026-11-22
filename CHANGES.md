# Change log: Endterm to Final

Same data, same 793 test listings, same grouped 5-fold CV (seed 42). Evidence is in `final.ipynb`; `endterm.ipynb` is unchanged. Section numbers below refer to `final.ipynb`.

| # | Endterm feedback (Moodle) | Fix | Before | After | Section |
|---|---|---|---|---|---|
| 1 | Check the terms for the product photos | robots.txt and the site's copyright notice checked before any download, recorded in `DATA_CARD.md` | photo collection planned | robots.txt allows image files, but the copyright notice forbids reuse without consent: no mebel.kz photos downloaded; photos from ABO (CC BY 4.0) instead; every source's rights, matching and missing inputs in one table | 3 |
| 2 | Scores ≥ 0.95: check colour variants across the split | new grouping in `furniture/features.py: build_groups` joins listings that share series, type + model name, or type + sizes + price | 41 training twins removed; 359 training groups; forest CV 0.958 ± 0.006 | 197 removed; 315 groups; 44 test listings had a missed twin; forest CV 0.933 ± 0.037, SVM 0.949 ± 0.019 (now chosen) | 1.2-1.6 |
| 3 | Same feedback: is 0.97 meaningful? | test score also counted per design (weight 1 / listings in the design) | SVM test 0.974 per listing | SVM test 0.940 per design; best models wrong on 6-9 of 92 designs | 1.5 |
| 4 | Coffee tables are 4%: report precision, recall, PR-AUC | per-class precision, recall, F1 and one-vs-rest PR-AUC, cross-validated and on test, with PR curves | macro-F1 only | coffee table (SVM, CV): precision 0.962, recall 0.940, PR-AUC 0.991; weakest classes are TV stand, desk, chest (CV F1 0.875-0.914) | 2 |
| 5 | Keep the talk within the time (Final: 10 minutes including the demo, hard stop) | slide plan with a timed split, target 9:00 | - | to rehearse | slides |
| 6 | Teacher (message after the Endterm): rethink the overall problem formulation | cross-store test: SVM trained on mebel.kz, applied to 2,886 Amazon (ABO) listings with the inputs both stores share | only one store tested | sizes only: 0.890 at home, 0.606 on Amazon; with materials 0.945 at home, 0.528 on Amazon; case furniture F1 0.30-0.50 | 6 |
| 7 | Endterm Final plan: photos for chest / TV stand, small CNN + pretrained CNN | photos from ABO (CC BY 4.0) instead of mebel.kz; small CNN, ResNet-18 + logistic regression, specs + photo; grouped 5-fold CV with settings chosen inside each fold | specs only | ABO macro-F1: specs 0.662, small CNN 0.605, ResNet-18 0.766, specs + photo 0.794 (fresh run of 21 Nov; first run: 0.597 and 0.771); TV-stand recall 0.08 -> 0.64 | 5 |
| 8 | Endterm Final plan: local app | React app (`frontend/`) + model API (`api.py`); Streamlit `app.py` kept as a fallback | - | three modes: type from specifications, type from a photo (+ optional specs), **price** from specifications (+ optional name); every input validated in the browser and again in the API (HTTP 422 with a message) | 8 |
| 9 | Teacher, 20 Nov: the type is decided by the sizes (three sizes about 0.92); reformulate, e.g. as price on the same listings and test set | checked the number; main question moved to price; type kept as a finding | type as the main question | three sizes, random forest: test macro-F1 0.918 (confirmed); new main question: price | 4.1 |
| 10 | Same message: photos should add something | photos of mebel.kz are not allowed; second source on the **same** listings: the product name (own TF-IDF features vs pretrained multilingual-e5-small), same 793 test URLs, same groups, seed 42 | - | price, CV MAE (KZT): median of type 94,800; best specs-only (forest) 61,400; **specs + name (TF-IDF, SVR) 59,100**; specs + name (e5, SVR) 60,200. Test MAE 54,300 → 51,700 (-4.7%) | 7 |
| 11 | Leakage risk of the name (twins give away the price) | grouped split + ablation with the design name masked | - | three quarters of the gain stays with the name masked (2,366 → 1,767 KZT) | 7.1 |
| 12 | Run-to-run variation (rubric) | the key comparisons repeated over six fold cuts | one cut | price: chosen model better in 6 of 6 cuts (+2,400 to +4,100 KZT); type SVM 0.948 ± 0.003 | 7.2 |
| 13 | Existing solutions (rubric) | short review: IKEA price tutorial, Mercari price challenge, multilingual E5, ABO, ResNet | - | notebook, section "Existing solutions" | intro |
| 14 | User survey (22 Nov): the models are a weak part | stronger frozen photo encoders (CLIP, DINOv2) and a fine-tuned ResNet-18 on the same ABO folds; LightGBM added to the price models | ABO photo: ResNet-18 0.766, specs + photo 0.794; price specs-only best: forest 61,436 KZT | ABO photo: **DINOv2 0.844**, CLIP 0.801, fine-tuned ResNet-18 0.789; **specs + DINOv2 0.851** (TV-stand recall 0.78). Price: **LightGBM 60,591** becomes the best specs-only model (test 53,666); the chosen model (SVR + name, 59,071) is unchanged; its CV gain over specs-only is now 1,500-4,300 KZT in 6 of 6 cuts | 5.6, 7, 7.2 |
| 15 | Same survey: the data is too limited (one store) | two IKEA sources as checks in other markets (licences in `DATA_CARD.md`; Bonn Furniture Styles rejected, licence cannot be verified) | one store for price; ABO only for photos | IKEA Saudi Arabia (620 items): specs + text (SVR) beats specs only, MAPE 46.6% → 39.9%, 4 of 5 folds. IKEA photos (221, outside test): DINOv2 macro-F1 0.880, ResNet-18 0.755 | 5.7, 7.4 |
| 16 | App test by two users (22 Nov): technical terms, the aim is unclear, foam breaks the price mode, a price is required for the type, made-up names move the price | plain-language results (price ± typical error, "8 in 10 similar listings cost between ...", score out of 100); a **How it works** page (aim, data, training, models, results, limits, glossary); foam hidden in price mode; a type model **without price** (`models/structured_noprice.joblib`, section 8); names with too few known words or naming another type fall back to specifications (`furniture/price.py: check_name`); photo mode uses DINOv2 | "asdkjh qwe" priced a sofa at 602,500 vs 440,700 from specs; foam → HTTP 422; price required | name ignored with a reason; foam not offered; type without price: CV macro-F1 **0.950** (with price 0.949): the price adds nothing to the type | 8, app |

**Result:** the main question of the Final is the **price**; after the survey, the best photo encoder is DINOv2 and the best specifications-only price model is LightGBM, but the chosen price model stays the same; specifications + product name (SVR on TF-IDF) is the chosen price model, saved in `models/price_best.joblib` with the best specifications-only model. For the type, the chosen mebel.kz model changes from the random forest to the SVM (C = 3, gamma = 0.1). The forest's Endterm lead was partly due to colour twins of test designs in training. `models/structured_best.joblib` now holds this SVM (with probabilities); the Endterm forest is no longer saved there.

---

# Change log: Midterm to Endterm

Everything stays the same unless listed here: the data (`data/processed/furniture.csv`, collected 7 October 2026), the 793 test listings (`data/processed/test_urls.csv`), grouped 5-fold cross-validation with seed 42, and macro-F1 as the main metric. Details and evidence are in `endterm.ipynb` (the section is named in each row).

## Fixes

| # | Weakness (source) | Fix | Before | After | Section |
|---|---|---|---|---|---|
| 1 | 97 listings without a series were grouped by their own SKU, so their colour variants could be split between training and validation folds (Midterm limitations) | listings without a series are grouped by their exact size, price and materials (colour variants share them) | 512 groups | 359 training groups + 100 test groups, none shared | 1.2 |
| 2 | 14 test listings had a colour twin in the training rows, so the test score was slightly optimistic (Midterm limitations) | 41 training listings that are colour twins of test listings are removed; the test set is unchanged | train 3,168 rows; SVM test macro-F1 0.978 | train 3,127 rows; SVM test macro-F1 0.975 | 1.2, 1.3 |
| 3 | Same SVM, CV estimate with the fixed groups | - | CV 0.941 ± 0.024 | CV 0.946 ± 0.018 | 1.3 |
| 4 | The single decision tree was unstable: largest fold spread, train 0.993 vs CV 0.899 (Midterm 3.6, Endterm plan) | tuned random forest and gradient boosting | tree CV 0.901 ± 0.033 | forest CV 0.958 ± 0.006, boosting 0.952 ± 0.008 | 2 |
| 5 | Only one setting per model was tuned, by hand (Midterm plan) | SVM `C` and `gamma` searched together; ensembles and MLP searched with `GridSearchCV` / `RandomizedSearchCV` on the same grouped folds | SVM C = 3: 0.946 | SVM C = 3, gamma = 0.1: 0.947 | 2.3 |
| 6 | Only Week 3-4 model families compared (Midterm plan) | added an MLP with a grouped learning-curve check | - | MLP CV 0.939 ± 0.011, converged after 175 epochs | 4 |
| 7 | Redundant numeric inputs (Midterm plan) | PCA: 3 components explain 90%; PCA inside KNN tested | KNN 0.888 ± 0.020 | KNN + PCA(4) 0.900 ± 0.030 (within the spread, not adopted) | 3.1, 3.3 |
| 8 | Teacher's feedback at the Midterm defense: show in more detail what was done | the slides now include a step-by-step method slide (split, grouped folds, searches, choice by CV, one test run) and an appendix with every setting tried and chosen; every notebook section opens with a plain explanation of what it does | Midterm slides: results and models, without the procedure | Endterm slides 4 and 11; notebook section notes | slides |

## Changes in the result

* **Best model by CV:** SVM (Midterm) → **tuned random forest** (CV 0.958 ± 0.006, test 0.969).
* The tuned SVM scores higher on the test set (0.980). The choice stays with CV, because the test set holds only 100 designs and its errors come in blocks of colour variants (notebook section 5).
* **Weak spot:** still the mid-size case furniture. For the forest it is mostly small desks predicted as chests of drawers ("Крейн" desks), plus the "Тумба Воямполь" chest/TV-stand labels already seen at the Midterm.
* The chosen model is saved in `models/structured_best.joblib` for the Final's app.

## Remaining issues (not fixed)

* Store labels at the chest / TV stand / desk border are partly debatable; there is no hand-checked ground truth.
* Specifications cannot show function (a desk with a 190 cm shelving tower has a wardrobe's height). This is the target of the Final's photo source.
* No weight, one store, one city, one price snapshot (unchanged from the Midterm).
