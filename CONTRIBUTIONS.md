# Contributions and AI tools (Final)

Team: Sultanbek Zhumagali, Tamerlan Yessimov (SE-2420).

## Who did what

**Lead** owns the method, the interpretation and the corrections; **review** checks the lead's work and can explain it. Both partners confirmed this table on 22 November 2026.

| Part of the Final (`final.ipynb` and app) | Sultanbek Zhumagali | Tamerlan Yessimov |
|---|---|---|
| Existing solutions; problem reformulation after the teacher's 20 Nov message (sections 0, 4) | Lead | Review |
| Section 1: colour variants, new grouping, re-scoring, per-design scores | Lead | Review |
| Section 2: per-class metrics, coffee tables, PR-AUC | Review | Lead |
| Section 3: photo terms, rights / matching / missing-inputs table, `DATA_CARD.md` | Review | Lead |
| Section 5: ABO photos, small CNN, ResNet-18, specs + photo | Lead | Review |
| Sections 5.6-5.7: CLIP, DINOv2, fine-tuned ResNet-18, IKEA photo test | Lead | Review |
| Section 6: cross-store test | Review | Lead |
| Section 7: price models, name text (TF-IDF, e5), masking ablation | Lead | Review |
| Sections 7.2-7.3: run-to-run variation, price error analysis | Review | Lead |
| Section 7 LightGBM and 7.4 IKEA Saudi Arabia | Review | Lead |
| Sections 8-9: saved models, conclusions, limitations | shared | shared |
| App (`api.py`, `frontend/`): price mode, input checks, demo | Review | Lead |
| Slides and the talk | shared | shared |
| Fresh-kernel run, archive build and check | Lead | Review |

## AI tools we used

| Tool | What we used it for |
|---|---|
| **Claude Code** (Anthropic, Claude Opus 5.5) | Regrouping of colour variants (`furniture/features.py`), per-class and PR-AUC analysis, the photo-terms check, the ABO and IKEA data scripts (`scraper/build_abo.py`, `scraper/download_abo_images.py`, `scraper/build_ikea.py`), photo code (`furniture/abo.py`, `furniture/photos.py`: small CNN, ResNet-18, CLIP, DINOv2, fine-tuning), the price section (`furniture/price.py`: TF-IDF, multilingual-e5, LightGBM, masking ablation, run-to-run variation), the notebook builder `tools/build_final_notebook.py` and draft interpretation texts, the app's API and price mode, draft documentation (`README.md`, `CHANGES.md`, `DATA_CARD.md`), fresh-kernel runs and a check of the submission archive. |
| **OpenAI Codex** | Reviewed the project against the Final rubric and listed what was missing (existing solutions, rights and matching documentation, run-to-run variation, app input checks, archive name, timing). |

AI generated substantial code and draft interpretations, as listed above. We ran the code, checked the results against the outputs and are responsible for every claim; each of us can explain every section, including the partner's.
