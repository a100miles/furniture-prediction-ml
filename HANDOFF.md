# Handoff: finishing the MLA Final

Written 20 November 2026 for Tamerlan (or his AI agent) to continue and finish the Final. Read this first, then `README.md` and `CHANGES.md`.

## Deadlines and format

| What | When |
|---|---|
| Submission (archive on Moodle, both partners upload the same file) | **Sunday 22 November 2026** (time not stated; assume 23:59) |
| Defense | **Tuesday 24 November 2026** |
| Talk + app demo | **10 minutes, hard stop** (the teacher stops the talk when time is up). Rehearse to 9:00. |

The Final rubric text was never pasted into the project. Check Moodle for it and for the archive name (the Endterm one was `Endterm_Zhumagali_Sultanbek_SE-2420.zip`, so probably `Final_Zhumagali_Sultanbek_SE-2420.zip`).

## The teacher's latest message (20 Nov, summary)

1. The type is almost fully decided by the sizes: the three dimensions alone give a random forest a test macro-F1 of about 0.92. Photos would only make the type easier to see, so they would not show much for this question.
2. Suggestion: reformulate the project, for example as **price prediction on the same listings and test set**, where photos have a real chance to add something. If we do, explain the change in the report and treat the type result as a finding.
3. Whatever we choose, check what the mebel.kz terms say about using product photos.
4. Deadline 22 Nov, defense 24 Nov, 10 minutes for talk + demo.

Earlier Endterm feedback (Moodle): photo terms; check colour variants across the split; per-class precision/recall/PR-AUC for coffee tables; keep within time. All of this is already answered in `final.ipynb` (see `CHANGES.md`).

### Already checked on 20 Nov (scratch run, not yet in the notebook)

Same 793 test listings, Final grouping (`build_groups`), training twins of test groups removed (2,971 train rows):

| Check | Result |
|---|---|
| Type from width, depth, height only, random forest (500 trees, balanced) | **test macro-F1 0.918**: the teacher's number is confirmed |
| Price (log KZT) from sizes + materials + type + colour, random forest, grouped 5-fold CV | R² (log) 0.739, MAE 63,700 KZT (median price 229,600) |
| Baseline: median price of the type | MAE 91,300 KZT |
| Same price model, one test run | R² (log) 0.839, MAE 52,500 KZT, MAPE 21% |

So price is a harder, non-trivial target where specs leave a real error. That is the teacher's point.

## The blocker: photos for price

The teacher wants price on **mebel.kz** listings, with photos. But:

* `DATA_CARD.md`, section "Product photos: terms check": mebel.kz robots.txt allows image files, **but the site footer says copyright belongs to TOO «Mebel Group KZ» and forbids reproduction or use of site material without their consent.** So we did **not** download any mebel.kz photos.
* The photos we do have are from **Amazon Berkeley Objects (ABO)**, CC BY 4.0, 3,332 furniture listings. **ABO has no price.** So ABO cannot be used for price.

Options (team decides, ideally tell the teacher in one short message):

1. **Recommended:** reformulate to price prediction on mebel.kz **from specifications** (sizes, materials, colour, type), same test set. Report honestly that photos cannot be used because of the copyright notice (that is exactly the check the teacher asked for), and keep the ABO photo experiment as supporting evidence that pretrained photo features carry information specs lack (on ABO they lift type macro-F1 from 0.662 to 0.794, TV-stand recall 0.08 to 0.64). The type result becomes a finding: "type is a size question (0.92 from 3 dims), and the 0.95 does not transfer to another store (0.606)".
2. Write to mebel.kz (info / contact form) asking for permission to use photos for a university project. Unlikely to get an answer before 22 Nov. Do not download photos without written consent.
3. Find another openly licensed furniture dataset that has **price and photos** (check the licence before downloading anything). Not checked yet; only worth it if found within a few hours.

## Current state (all pushed)

Repo: https://github.com/a100miles/furniture-prediction-ml (private), branch `main`, commit `5c762fa` and later.

| Path | What |
|---|---|
| `final.ipynb` | Final notebook, 61 cells, saved with outputs, runs from a fresh kernel (about 10 min with cache). Sections: 1 colour-variant regrouping, 2 per-class metrics + PR-AUC, 3 photo terms, 4 feedback summary + problem formulation, 5 ABO specs vs photo vs both, 6 cross-store test, 7 saved models, 8 conclusions. |
| `tools/build_final_notebook.py` | **Generates `final.ipynb`.** Edit this, then regenerate and execute, rather than hand-editing the notebook. |
| `furniture/features.py` | mebel.kz features (`add_features`, `make_pipeline`) and the grouping (`build_groups`: union-find on series, type + model word, type + sizes + price). |
| `furniture/abo.py`, `furniture/photos.py` | ABO table, grouping, inputs; image loading, ResNet-18 features, small CNN. |
| `tools/abo_experiment.py`, `tools/cross_market.py` | ABO cross-validation (writes `data/cache/abo_oof.npz`); cross-store check. |
| `scraper/build_abo.py`, `scraper/download_abo_images.py` | Build the ABO table and download its photos. |
| `app.py` | Streamlit app, two tabs: type from specs (mebel.kz SVM) and type from a photo (+ optional specs, ABO). Tested: default specs give "chest of drawers 98%". |
| `models/` | `structured_best.joblib` (SVM, mebel.kz), `abo_photo.joblib`, `abo_specs_photo.joblib`. |
| `CHANGES.md`, `DATA_CARD.md`, `CONTRIBUTIONS.md`, `README.md` | Updated for the Final, except the who-did-what part of CONTRIBUTIONS. |
| `endterm.ipynb`, `furniture_classification.ipynb` | Endterm and Midterm, do not change. |

Not in git (re-create): `.venv`, `data/external/abo/` (190 MB, ABO metadata + photos, see `DATA_CARD.md` and `README.md`), `data/cache/` (ResNet features and ABO predictions; rebuilt when missing, about 50 min on CPU).

Setup:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/streamlit run app.py
```

Key numbers already in the notebook:

* mebel.kz type, after stricter grouping: SVM CV macro-F1 0.949 (chosen), forest 0.933; SVM test 0.940 counted per design.
* Coffee tables (SVM, CV): precision 0.962, recall 0.940, PR-AUC 0.991.
* ABO, grouped 5-fold CV macro-F1: specs 0.662, small CNN 0.597, ResNet-18 + LR 0.771, specs + photo 0.794.
* Cross-store: mebel.kz model on 2,886 ABO listings, sizes only 0.890 at home vs 0.606 on Amazon.

## Next steps (if option 1 is chosen)

1. **Price section in the notebook** (add cells in `tools/build_final_notebook.py`, new section before "Saving models"):
   * Target `log(price_kzt)`. Same 793 test URLs, same `build_groups`, same removal of training twins, `GroupKFold(5)` (or the existing `StratifiedGroupKFold` on type), seed 42.
   * Inputs: sizes + derived (`add_features`), materials, colour, furniture type. **Do not use `series`, `sku`, `name` or `url` as inputs** (they identify the listing's price through twins).
   * Baselines: overall median, median of the type. Models: ridge/linear, KNN, random forest, gradient boosting, SVR or MLP; tune inside the folds; choose by CV, test once.
   * Metrics: MAE in KZT, MAPE, R² on log price; per-type MAE table; predicted vs actual plot; biggest errors with names (why: brand/series premium, fabric type, mechanisms the specs do not record).
   * Optional: does the predicted type (or type probabilities) help price? Does the price model transfer? (ABO has no price, so no.)
2. **Reframe the report**: section 4 "Problem formulation, revisited" and section 8 conclusions: type = finding (sizes decide it, 0.918 from three dims; per-class; does not transfer); the main question becomes price. Add a `CHANGES.md` row for the teacher's 20 Nov message.
3. **App**: add a third tab "Price from specifications" using the saved price model (`models/price_best.joblib`), with a range (for example the CV MAE or the 10th-90th percentile of the residuals).
4. **Photo terms**: already in `DATA_CARD.md`; mention it on a slide with the decision.
5. **Docs**: `README.md` (question changes to price), `CONTRIBUTIONS.md` (Final who-did-what + speaking split; AI disclosure already there, add this handoff's work), requirements unchanged.
6. **Fresh-kernel run**: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace final.ipynb`.
7. **Slides** (about 9-10, both speak, no code) + **live app demo** inside 10 minutes. Suggested split: problem + why we reformulated (1.5 min), data/terms/split (1 min), type finding (1.5 min), price models and errors (2.5 min), photos on ABO (1 min), demo (1.5 min), conclusions (1 min). Rehearse with a timer.
8. **Archive**: exclude `.venv`, `data/external`, `data/cache`; include notebooks with outputs, code, `models/`, `data/processed/`, docs, slides PDF. Open it and check. Both upload by 22 Nov.

## Risks

* Price labels are one snapshot (October 2026) of one store, with discounts possibly mixed in; say so.
* Colour variants share a price, so grouping matters even more for price than for type: keep the Final grouping.
* The teacher's main concern was the problem formulation; do not spend the remaining time on more type models.
* Rubric and exact deadline time are unconfirmed; check Moodle.
* AI use must be disclosed (`CONTRIBUTIONS.md`); both partners must be able to explain every section.
