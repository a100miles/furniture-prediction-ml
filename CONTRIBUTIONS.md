# Contributions and AI tools

## Endterm: who did what

The table assigns responsibility for completing, checking and defending each part. **Lead** means owning the method, interpretation and corrections; **review** means checking the lead's work and being able to explain it. AI-assisted code and text are disclosed below.

| Part of the Endterm (`endterm.ipynb`) | Sultanbek Zhumagali | Tamerlan Yessimov |
|---|---|---|
| Section 1: same split, the two grouping fixes, before/after, `CHANGES.md` | Lead | Review |
| Section 2: random forest, gradient boosting and SVM tuning | Lead | Review |
| Section 3: PCA, k-means, stability check, link to the model's errors | Review | Lead |
| Section 4: MLP learning curve and tuning | Lead | Review |
| Section 5: comparison table, per-type F1, confusion matrix, error examples | Review | Lead |
| Section 6: conclusions, limitations and the Final plan (photo source check) | Review | Lead |
| Slides: method, fixes, ensembles and MLP; rehearse those explanations | Lead | Review |
| Slides: results, PCA/clustering, errors, conclusions and Final plan; rehearse those explanations | Review | Lead |
| Fresh-kernel notebook run, saved model reload check, unchanged test split | Lead; record successful execution | Check the outputs and split counts |
| Check notebook numbers against slides; proofread README and contributions | Review the modelling details | Lead; reconcile results and wording |
| Build and inspect `Endterm_Zhumagali_Sultanbek_SE-2420.zip` | Lead; verify required files | Independently check archive contents |
| Upload the same archive and prepare for questions about every section | Upload own copy; shared preparation | Upload own copy; shared preparation |

### Defense roles: Thursday 12 November 2026

The talk is **7 minutes**, followed by **5 minutes of questions**. Both partners speak. Slide 11 is a backup appendix and is outside the timed talk.

| Speaker | Slides | Focus | Time |
|---|---|---|---|
| Sultanbek | 1-5 | Question, Midterm recap, grouping fixes, shared pipeline/CV method, ensembles and tuning | 3 min |
| Tamerlan | 6 | PCA, three stable clusters and where the errors concentrate | 45 s |
| Sultanbek | 7 | MLP learning curve, grouped validation and tuned architecture | 40 s |
| Tamerlan | 8-10 | Fair comparison, named error examples, conclusions, limitations and Final plan | 2 min 15 s |

The speaker notes in the slide deck follow this split (about 6 min 40 s in total).

Sultanbek should explain why groups prevent leakage, how settings were selected, and how MLP convergence was checked. Tamerlan should explain PCA versus clustering, silhouette/ARI, macro-F1, selection by CV rather than test score, and how photos could address the observed errors. Both must also explain the other partner's sections and identify their own work and the AI assistance.

### Checks done before submitting (11 November 2026)

- [x] Fresh-kernel run of `endterm.ipynb` (`jupyter nbconvert --execute`, about 5.5 minutes): all 26 code cells finished without errors, every printed result is identical to the previous saved run, and the saved-model reload check passes.
- [x] Inconsistencies found in the Codex review fixed: the gradient boosting wording in section 2 and on slide 5 (leaf size 20 is the middle of the values tried, not the strongest), and two answers in our study guide (k range 2 to 10; the "Пенни" desk as the listing every model gets wrong).
- [x] Slide numbers checked against the notebook; slides re-exported with the corrected surname and the speaker split above.
- [x] Archive built and inspected; both partners upload the same file by **Wednesday 11 November 2026, 23:59**.

## Final: who did what

Same convention as the Endterm: **lead** owns the method, the interpretation and the corrections; **review** checks the lead's work and must be able to explain it. **This table is a proposal written on 21 November 2026; both partners confirm or correct it before submission.**

| Part of the Final (`final.ipynb` and app) | Sultanbek Zhumagali | Tamerlan Yessimov |
|---|---|---|
| Existing solutions; problem reformulation after the teacher's 20 Nov message (sections 0, 4) | Lead | Review |
| Section 1: colour variants, new grouping, re-scoring, per-design scores | Lead | Review |
| Section 2: per-class metrics, coffee tables, PR-AUC | Review | Lead |
| Section 3: photo terms, rights / matching / missing-inputs table, `DATA_CARD.md` | Review | Lead |
| Section 5: ABO photos, small CNN, ResNet-18, specs + photo | Lead | Review |
| Section 5.6-5.7 (added 22 Nov): CLIP, DINOv2, fine-tuned ResNet-18, IKEA photo test | Lead | Review |
| Section 6: cross-store test | Review | Lead |
| Section 7: price models, name text (TF-IDF, e5), masking ablation | Lead | Review |
| Section 7.2-7.3: run-to-run variation, price error analysis | Review | Lead |
| Section 7 LightGBM and 7.4 IKEA Saudi Arabia (added 22 Nov) | Review | Lead |
| Section 8-9: saved models, conclusions, limitations | shared | shared |
| App (`api.py`, `frontend/`): price mode, input checks, demo script | Review | Lead |
| Slides and the timed talk | shared | shared |
| Message to the teacher (ABO and product names) | Lead | Review |
| Fresh-kernel run, archive build and check, both uploads | Lead; build | Check the archive independently |

### Defense roles: Tuesday 24 November 2026

The talk and the app demo together are **10 minutes, hard stop**. Rehearse to 9:00. Both partners speak.

| Speaker | Part | Time |
|---|---|---|
| Sultanbek | Question, the teacher's feedback and why the question moved to price; existing solutions | 1:15 |
| Tamerlan | Data, rights (no mebel.kz photos; names yes, descriptions no; ABO), the fixed split and groups | 1:00 |
| Sultanbek | Type as a finding: colour twins, 0.95 in CV, 0.918 from three sizes, per-class, 0.61 in another store | 1:15 |
| Sultanbek | Price: baselines, specs-only models, specs + name (own vs pretrained), masking check | 1:45 |
| Tamerlan | Run-to-run variation and the price errors (modular sofas, the *Монблан* chest, fabric words) | 1:15 |
| Tamerlan | Photos on ABO: ResNet-18 vs small CNN vs specs, TV stands | 0:45 |
| Tamerlan | Live demo: price with and without the name, one invalid input, type from a photo | 1:00 |
| Sultanbek | Conclusions and limitations | 0:45 |

Total 9:00. Each partner must also be able to explain the other's parts and say which work was AI-assisted.

### Checks before submitting (Final)

- [x] `final.ipynb` run from a fresh kernel (`jupyter nbconvert --execute`) after the price section was added; the saved-model reload checks pass (21 November 2026).
- [x] App checked against the saved models: the three modes and the invalid-input answers (21 November 2026).
- [ ] Both partners have read sections 4, 7 and 9 and can explain them.
- [ ] Message to the teacher sent (ABO as a secondary photo source; product names as the text source).
- [ ] Slides made and the talk rehearsed to 9:00 with the demo.
- [ ] Archive `Final_Zhumagali_Sultanbek_SE-2420.zip` (name to be checked on Moodle) built, opened and checked; both partners upload the same file by Sunday 22 November 2026.

## Final: AI tools we used

| Tool | What we used it for |
|---|---|
| **Claude Code** (Anthropic, model Claude Opus 5.5) | At the team's request, after the Endterm feedback: found the colour variants the Endterm grouping missed and wrote the new grouping (`furniture/features.py`); added the per-class and PR-AUC analysis; checked the mebel.kz photo terms (robots.txt and the copyright notice) and recorded them; proposed Amazon Berkeley Objects as a licensed photo source, wrote `scraper/build_abo.py` and `scraper/download_abo_images.py` and downloaded the data; wrote the photo code (`furniture/abo.py`, `furniture/photos.py`: ResNet-18 features, small CNN), the cross-validation script `tools/abo_experiment.py`, the cross-store check, the notebook builder `tools/build_final_notebook.py` and its interpretation texts, `app.py`, and the updates to `README.md`, `DATA_CARD.md` and `CHANGES.md`. Ran the notebook from a fresh kernel and tested the app with Streamlit's AppTest. |
| **Claude Code** (Anthropic, model Claude Opus 5.5), 21 November 2026 | At Sultanbek's request, after the teacher's 20 Nov message and a Codex review of the rubric: wrote the price section (`furniture/price.py`, `tools/price_experiment.py`, notebook sections 4.1 and 7: specs-only models, product-name text with TF-IDF and multilingual-e5-small, the masking ablation, run-to-run variation, error analysis), the existing-solutions section and the rights / matching / missing-inputs table, the price model file and the price mode of the app (`api.py`, `frontend/src/PredictPage.tsx`) with stricter input checks; re-downloaded the ABO photos, re-ran `final.ipynb` from a fresh kernel, and updated `README.md`, `CHANGES.md`, `DATA_CARD.md` and this file (the who-did-what table above is its proposal). |
| **Claude Code** (Anthropic, model Claude Opus 5.5), 22 November 2026 | At Sultanbek's request, after a small user survey of the app (main weaknesses: models and data size): checked the licences of three proposed datasets (Bonn Furniture Styles rejected: licence cannot be verified; IKEA SA and IKEA catalogue photos accepted with the weaknesses recorded in `DATA_CARD.md`); wrote `scraper/build_ikea.py`, `furniture/ikea.py`, the CLIP / DINOv2 feature extraction and the fine-tuned ResNet-18 (`furniture/photos.py`, `tools/abo_encoders_experiment.py`), LightGBM in the price comparison, notebook sections 5.6, 5.7 and 7.4 and the updated notes and conclusions; ran the experiments and the notebook; updated `README.md`, `CHANGES.md`, `DATA_CARD.md` and this file. Also ran two split checks for the team (70/15/15 and 75/25 grouped splits over five seeds, not in the notebook). |
| **OpenAI Codex** | Reviewed the project against the Final rubric and listed what was still missing (existing solutions, rights and matching documentation, run-to-run variation, app input checks, archive name, timing). That list was checked against the rubric PDF and extended with the idea of using the product names as a second source on the same listings. |

AI generated substantial code and draft interpretations for the Final, as recorded above. The team is responsible for checking the results and being able to explain every section.

---

## Endterm: AI tools we used

| Tool | What we used it for |
|---|---|
| **Claude Code** (Anthropic, model Claude Opus 5.5) | Collected the Endterm and Final requirements from Moodle and wrote a study guide with explanations of the methods and practice questions, which we used to prepare (not part of this archive). Wrote the code of `endterm.ipynb` (grouping fix, grid and random searches, PCA, k-means, MLP learning curve, comparison table, error tables, model saving) and drafted the interpretation texts and `CHANGES.md`. Checked that mebel.kz's robots.txt allows product images, for the Final plan. Built the slide deck. On 11 November, at the team's request, fixed the inconsistencies found in the Codex review, re-ran the notebook from a fresh kernel, compared its outputs with the saved run, re-exported the slides and built the archive. |
| **OpenAI Codex** | Reviewed local project files and saved notebook outputs; checked dataset/split counts; drafted an internal review task list (kept outside this archive); corrected Sultanbek's surname in documents, notebooks and slide PDFs; assigned Endterm completion/review and defense responsibilities at the team's request; synchronized contributions and the team-confirmed deadline across project documents. Codex did not retrain models, execute the full notebook, check Moodle, rehearse or upload the submission during this review. |

AI generated substantial code and draft interpretations, as recorded above. The team is responsible for verifying the results, correcting unsupported explanations and understanding every section. Lead/reviewer assignments describe responsibility, rather than claiming sole manual authorship of AI-generated material.

---

## Midterm: who did what

| Part of the project | Sultanbek Zhumagali | Tamerlan Yessimov |
|---|---|---|
| Choosing the question and the source (mebel.kz) | shared | shared |
| First scraping prototype (sitemap reading, `main.py`, `part1.py`) | ✓ | ✓ |
| Checking the source: robots.txt, comparing scraped rows with the live pages | | ✓ |
| Running the full collection (`scraper/collect_mebel.py`) and keeping the raw data and log | ✓ | |
| Building the clean table and the category mapping (`scraper/build_dataset.py`) | ✓ | |
| Data card (`DATA_CARD.md`) and data ethics section | | ✓ |
| Cleaning and leakage check (notebook section 2.1-2.3) | ✓ | |
| Exploration and feature ideas (section 2.4) | | ✓ |
| Modelling: split, pipeline, baseline, tree / KNN / SVM, cross-validation (section 3) | ✓ | |
| Error analysis and confusion matrix (section 4) | | ✓ |
| Conclusions and limitations (section 5) | shared | shared |
| Reviewing the notebook, re-running it top to bottom | shared | shared |

## Midterm: AI tools we used

| Tool | What we used it for |
|---|---|
| **Claude Code** (Anthropic, model Claude Opus 5.5), in the terminal | Inspected the structure of mebel.kz pages (JSON-LD, spec table, sitemap) and wrote the scraper `scraper/collect_mebel.py` (rate limiting, resumable runs, logging). Wrote `scraper/build_dataset.py` (category-to-type mapping, material and colour normalisation). Drafted the analysis notebook `furniture_classification.ipynb` (cleaning checks, leakage checks, pipelines, cross-validation, confusion-matrix analysis), and drafted this file, `DATA_CARD.md` and `README.md`. |

How we used the AI output: we ran every script ourselves and checked scraped rows against the live product pages. We re-ran the notebook from top to bottom and read every output before writing our interpretation. We are responsible for every claim in the notebook. The modelling methods are the ones from the Week 3-4 course notebooks (`week3-4/`).
