# Change log: Endterm to Final

Same data, same 793 test listings, same grouped 5-fold CV (seed 42). Evidence is in `final.ipynb`; `endterm.ipynb` is unchanged.

| # | Endterm feedback (Moodle) | Fix | Before | After | Section |
|---|---|---|---|---|---|
| 1 | Check the terms for the product photos | robots.txt and the site's copyright notice checked before any download, recorded in `DATA_CARD.md` | photo collection planned | robots.txt allows image files, but the copyright notice forbids reuse without consent: no photos downloaded, permission or another source needed | 3 |
| 2 | Scores ≥ 0.95: check colour variants across the split | new grouping in `furniture/features.py: build_groups` joins listings that share series, type + model name, or type + sizes + price | 41 training twins removed; 359 training groups; forest CV 0.958 ± 0.006 | 197 removed; 315 groups; 44 test listings had a missed twin; forest CV 0.933 ± 0.037, SVM 0.949 ± 0.019 (now chosen) | 1.2-1.6 |
| 3 | Same feedback: is 0.97 meaningful? | test score also counted per design (weight 1 / listings in the design) | SVM test 0.974 per listing | SVM test 0.940 per design; best models wrong on 6-9 of 92 designs | 1.5 |
| 4 | Coffee tables are 4%: report precision, recall, PR-AUC | per-class precision, recall, F1 and one-vs-rest PR-AUC, cross-validated and on test, with PR curves | macro-F1 only | coffee table (SVM, CV): precision 0.962, recall 0.940, PR-AUC 0.991; weakest classes are TV stand, desk, chest (CV F1 0.875-0.914) | 2 |
| 5 | Keep the talk within 7 minutes | timed rehearsal, target 6:30 | - | to do | slides |
| 6 | Teacher (message after the Endterm): rethink the overall problem formulation | cross-store test: SVM trained on mebel.kz, applied to 2,886 Amazon (ABO) listings with the inputs both stores share | only one store tested | sizes only: 0.890 at home, 0.606 on Amazon; with materials 0.945 at home, 0.528 on Amazon; case furniture F1 0.30-0.50 | 6 |
| 7 | Endterm Final plan: photos for chest / TV stand, small CNN + pretrained CNN | photos from ABO (CC BY 4.0) instead of mebel.kz; small CNN, ResNet-18 + logistic regression, specs + photo; grouped 5-fold CV with settings chosen inside each fold | specs only | ABO macro-F1: specs 0.662, small CNN 0.597, ResNet-18 0.771, specs + photo 0.794; TV-stand recall 0.08 -> 0.64 | 5 |
| 8 | Endterm Final plan: local app | `app.py` (Streamlit): specifications or photo (+ optional specs), probabilities shown | - | loads the models from `models/`; checked with Streamlit's AppTest | 7 |

**Result:** the chosen mebel.kz model changes from the random forest to the SVM (C = 3, gamma = 0.1). The forest's Endterm lead was partly due to colour twins of test designs in training. `models/structured_best.joblib` now holds this SVM (with probabilities); the Endterm forest is no longer saved there.

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
