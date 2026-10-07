# Furniture type from specifications (mebel.kz)

Course project, Direction 21 "Furniture" (classification).

**Question:** can the type of a furniture item (sofa, armchair, chair, bed, wardrobe, chest of drawers, desk, coffee table, TV stand) be predicted from its width, depth, height, materials, colour and price?

## What is where

| Path | What it is |
|---|---|
| `furniture_classification.ipynb` | The main notebook: question and ethics, cleaning, leakage check, exploration, modelling, error analysis, conclusions, contributions. Saved with all outputs. |
| `scraper/collect_mebel.py` | Data collection code (sitemap -> product pages -> raw JSONL). |
| `scraper/build_dataset.py` | Raw JSONL -> clean CSV (category mapping, dimensions, materials, colour). |
| `data/raw/mebel_raw.jsonl` | Raw data: one JSON object per product page, with the collection timestamp. |
| `data/raw/collection_log.json` | Collection date, counts and politeness settings. |
| `data/processed/furniture.csv` | The table the notebook uses. |
| `DATA_CARD.md` | Data card: source, what one row is, every field, how it was collected and checked. |
| `CONTRIBUTIONS.md` | Who did what, and which AI tools were used for what. |
| `week3-4/` | Course demo notebooks the modelling follows. |

## Reproduce

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
# 1. collect (about 1.5 hours at the polite rate; resumable - rerun to continue)
.venv/bin/python scraper/collect_mebel.py
# 2. build the table
.venv/bin/python scraper/build_dataset.py
# 3. run the notebook top to bottom and save outputs
.venv/bin/jupyter nbconvert --to notebook --execute --inplace furniture_classification.ipynb
```

Steps 1-2 are optional: the collected data is already in `data/`.
