# Data card - mebel.kz furniture listings

## Source
* **Website:** [mebel.kz](https://mebel.kz/), the online store of TOO "Mebel Group KZ", Almaty, Kazakhstan.
* **Entry point:** the public sitemap `https://mebel.kz/sitemap.almaty.xml` (9,019 product URLs on the collection date).
* **Collection date:** 7 October 2026, from 11:46 to 13:17 UTC (16:46-18:17 Almaty time). Every raw record has its own `scraped_at` timestamp, and `data/raw/collection_log.json` stores the run details.
* **Collected by:** `scraper/collect_mebel.py` (raw pages -> `data/raw/mebel_raw.jsonl`), then `scraper/build_dataset.py` (raw -> `data/processed/furniture.csv`).

## Size of the final table
**3,961 rows** in `data/processed/furniture.csv` (from 4,630 raw pages), in 512 product series:

| type | rows | type | rows | type | rows |
|---|---|---|---|---|---|
| wardrobe | 699 | sofa | 425 | desk | 318 |
| bed | 659 | tv_stand | 389 | chair | 309 |
| chest_of_drawers | 652 | armchair | 342 | coffee_table | 168 |

## What one row describes
**One row = one product listing on mebel.kz:** one furniture model in one colour/fabric, with the size and the price shown on its product page on the collection date. Colour variants of the same model are separate listings and share the same `series`.

## How the data was collected
1. Download the sitemap and keep the `/product/` URLs (9,019).
2. Keep the URLs whose address contains a furniture word (`divan`, `kreslo`, `stul`, `krovat`, `shkaf`, `komod`, `stol`, `tumba`). Remove accessories (covers, armrests, legs, mattresses, mirrors), kitchen/dining tables, nightstands, office chairs and bar stools. This leaves 6,000 candidates.
3. To keep the collection time reasonable and the classes less unbalanced, keep **at most 700 URLs per type**, guessed from the URL. The sample is a reproducible random one (seed 42). This leaves 4,630 pages.
4. Download every page once:
   * at most 2 connections at a time, and at least 0.7 s between requests;
   * a User-Agent that names this as a student research scraper;
   * up to 4 retries with back-off.

   Result: 4,630 of 4,630 pages, 0 failures.
5. From each page, store:
   * the schema.org `Product` JSON-LD block (name, SKU, category, price, currency, availability);
   * the breadcrumbs;
   * the overall size block ("Размеры");
   * every name/value pair of the specification table, including the product series.

   Raw HTML is not stored.
6. The first version of the size parser missed two page layouts ("Размеры" without a colon, and "Глубина габаритная"). The parser was fixed, and the 520 pages with incomplete sizes were downloaded again; 353 of them then had all three sizes (`refetches` in the collection log).
7. `build_dataset.py`:
   * maps the store category to one of nine types using an explicit dictionary (`CATEGORY_TO_TYPE`);
   * drops other categories (502 pages, mostly the 392 separate modules of modular sofas);
   * drops listings without all three sizes or a price (167);
   * normalises materials and colour.
8. A later attempt to top up sofas and beds (cap 800) stopped: mebel.kz started answering every page, including the home page, with HTTP 500. Nothing from that attempt was saved, and we did not keep retrying.

## Fields (`data/processed/furniture.csv`)

| Field | Type | Description | Source on the page |
|---|---|---|---|
| `url` | text | product page address | sitemap |
| `scraped_at` | date | day the page was downloaded | scraper clock (UTC) |
| `sku` | text | store article number | JSON-LD `sku` |
| `name` | text | product name, e.g. "Комод Леванте 1 Белый". **Not a model input** (it contains the type) | JSON-LD `name` |
| `site_category` | text | the store's category, e.g. "Шкафы распашные". **Not a model input** | JSON-LD `category` |
| `series` | text | product series, e.g. "Леванте"; listings without a series get their SKU. Used only to group rows in cross-validation | spec table "Серия" |
| `furniture_type` | category | **target**: sofa, armchair, chair, bed, wardrobe, chest_of_drawers, desk, coffee_table, tv_stand | mapped from `site_category` |
| `width_cm` | number | outer size "Длина" (front width; for beds the bed length) | "Размеры" block |
| `depth_cm` | number | outer size "Ширина" or "Глубина (габаритная)" | "Размеры" block |
| `height_cm` | number | outer size "Высота" | "Размеры" block |
| `price_kzt` | number | price in Kazakhstani tenge on the collection date | JSON-LD `offers.price` |
| `materials` | text, `\|`-separated | material tokens found in the values of every "Материал ..." and "Каркас" field: chipboard, mdf, fibreboard, solid_wood, plywood, veneer, metal, plastic, glass, mirror, fabric, leather, foam, spring, rattan, stone | spec table |
| `main_color` | category | first colour of the "Цвет" field (or "Цвет корпуса"), translated to English; colours outside the list -> `other` | spec table |
| `n_spec_fields` | integer | number of spec fields on the page. **Not a model input** (it shows a leak, see notebook 2.3) | spec table |

The raw file `data/raw/mebel_raw.jsonl` keeps every spec field as name/value text, so other features can be rebuilt without downloading again.

## How we checked the source
* **robots.txt:** `/product/` pages are allowed. No disallowed paths (`/order`, `/terms`, `/data`, `/params`, ...) were requested.
* **Manual comparison:** several scraped records (a bed, a sofa, a wardrobe, a chest, a chair, a TV stand, an armchair) were compared with the live pages. Sizes, price and materials matched.
* **Price consistency:** the JSON-LD price equals the price shown on the page (for example 191,330 ₸ for "Кровать Дона 180 Береза Серый").
* **Automatic checks in the notebook (section 2.1):** sizes between 10 and 500 cm, price at least 5,000 ₸, no duplicate URLs, and duplicate spec combinations counted.
* **Category check:** we read sample product names in every mapped category. "Кушетки" on mebel.kz are straight sofas, so they are mapped to sofa. Some listings in "Комоды" are named "Тумба" (low cabinets): this is the store's own labelling and remains as label noise.

## Known limitations
* **No weight.** mebel.kz does not publish product weight, so the weight feature from the task description is missing.
* **One store, one city.** Almaty sitemap, mostly the store's own brand. Other marketplaces may describe sizes differently.
* **Sample, not census.** At most 700 pages per type were downloaded (seeded random sample).
* **Labels are the store's categories.** They are mostly right, but not perfect (see the "Тумба" in "Комоды" example).
* **Size convention.** "Длина x Ширина x Высота" is read as width x depth x height. For beds, "Длина" is the length of the bed.
* **Near-duplicates.** Colour variants share size, price and series.
* **Price snapshot.** Prices are from one day, in KZT, and include any discount shown that day.

## Ethics
Only public catalogue data was collected: no accounts, no personal data, no reviews. The collection was rate-limited and identified itself. The data is used for a university course project only.
