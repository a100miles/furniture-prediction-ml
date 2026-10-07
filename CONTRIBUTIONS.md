# Contributions and AI tools

## Who did what

| Part of the project | Sultanbek Zhumagaliev | Tamerlan Yessimov |
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

## AI tools we used

| Tool | What we used it for |
|---|---|
| **Claude Code** (Anthropic, model Claude Opus 5.5), in the terminal | Inspected the structure of mebel.kz pages (JSON-LD, spec table, sitemap) and wrote the scraper `scraper/collect_mebel.py` (rate limiting, resumable runs, logging). Wrote `scraper/build_dataset.py` (category-to-type mapping, material and colour normalisation). Drafted the analysis notebook `furniture_classification.ipynb` (cleaning checks, leakage checks, pipelines, cross-validation, confusion-matrix analysis), and drafted this file, `DATA_CARD.md` and `README.md`. |

How we used the AI output: we ran every script ourselves and checked scraped rows against the live product pages. We re-ran the notebook from top to bottom and read every output before writing our interpretation. We are responsible for every claim in the notebook. The modelling methods are the ones from the Week 3-4 course notebooks (`week3-4/`).
