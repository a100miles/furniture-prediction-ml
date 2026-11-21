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

## Product photos: terms check (Final, Endterm feedback 1)
Checked on 19 November 2026, before any photo was downloaded.

* **robots.txt** (`https://mebel.kz/robots.txt`): the `*` group explicitly allows `*.jpg`, `*.jpeg` and `*.png`; there is no crawl-delay. So crawlers are *technically* allowed to fetch the image files.
* **Site terms:** mebel.kz has no separate terms-of-use page; the footer links only to the public offer contracts (for buying), the privacy policy and payment security. The binding statement is the footer copyright notice of TOO «Mebel Group KZ» (expanded with "Читать далее"). It says all site resources, explicitly including photographic information, are copyright objects, and it forbids any reproduction, including use and copying, of the site's content without the rights holder's prior consent.
* **Conclusion:** robots.txt is about crawling, not about the right to reuse. Downloading the product photos, training on them and putting them in our submission archive is reproduction and use of copyrighted photos, which the notice forbids without consent. **We do not download the photos unless the store gives written permission.**
* **Options considered for the Final** (outcome below):
  1. ask Mebel Group KZ for written permission to use product photos for a non-commercial course project, keeping the photos out of the archive (store only URLs);
  2. use an openly licensed furniture image dataset (licence recorded per image) for the photo models; it cannot be linked to our listings, so the photo model is evaluated on its own test set and only the structured model uses the 793 saved listings;
  3. keep the Final on the specification data only.
* **Outcome (20 November 2026):** option 2 for the photos (ABO, below). For the price question (teacher's message of 20 November), the second source on the same listings is the **product name** (`name`), not the photos and not the page descriptions.
* The same notice also covers the text of the pages. Our table holds measured facts (sizes, price, material and colour categories) extracted from the pages, not copied text or images, and is used only for this course project; the raw JSONL with page text is shared only with the course staff through Moodle, not published.

## Product names as a model input (Final, price)
* **What:** the `name` field of each listing (e.g. "Диван прямой Линс Велюр Голубой": type words, the design name, size, fabric, colour). Present for all 3,961 listings; already in `furniture.csv` since the Midterm.
* **Why it is acceptable:** a name is a short factual label of the product, of the same kind as the sizes and materials we extract. It is used only as a model input; it is not republished beyond the course archive. The long **page descriptions** (about 200 characters of marketing text, in the raw JSONL only) are **not used**: they are page text covered by the copyright notice.
* **Matching:** the name is a field of the same row (same URL), so it has the same split, groups and test URLs as the specifications.
* **Leakage check:** the design name inside the product name could identify a design's price through colour twins. The grouped split keeps every design on one side; in addition, `furniture/price.py: mask_name` replaces the design name and series words with a placeholder for an ablation (notebook section 7.1).
* **Pretrained encoder:** `intfloat/multilingual-e5-small` (MIT licence), downloaded from Hugging Face on first use (~470 MB, not in the archive); embeddings are cached in `data/cache/`.
* **Confirmation pending:** the team asks the teacher to confirm that product names are an acceptable text source (and ABO as a secondary photo source).

## Second source for the Final: Amazon Berkeley Objects (ABO)
Chosen because the mebel.kz photos may not be reused (see above).

* **What:** product listings from Amazon marketplaces (amazon.com, .co.uk, .de, .in, ...) with catalogue photos, released as the ABO dataset. https://amazon-berkeley-objects.s3.amazonaws.com/index.html
* **Licence:** Creative Commons Attribution 4.0 (CC BY 4.0) according to the dataset page and its licence file, copy in `data/external/abo/LICENSE-CC-BY-4.0.txt` (checked again 21 November 2026). The CVPR 2022 paper (Collins et al.) states CC BY-NC 4.0; our use is non-commercial, so we meet both. Credit for the data and images: **Amazon.com**. Credit for building the dataset: Matthieu Guillaumin, Thomas Dideriksen, Kenan Deng, Himanshu Arora (Amazon.com), Jasmine Collins and Jitendra Malik (UC Berkeley). Changes we made: selected furniture listings, mapped categories to our nine types, converted units, and (in the notebook) resized photos.
* **Downloaded on 19 November 2026** (photos and image index downloaded again on 21 November 2026 for the fresh notebook run: 10,893 of 10,893 images, none failed): the 16 listing metadata files and the image index (90 MB), then the 256 px photos of the selected listings: main photo plus up to 3 other views, 10,893 images (101 MB). Stored in `data/external/abo/`, not in the submission archive.
* **Table:** `scraper/build_abo.py` -> `data/processed/abo_furniture.csv` (3,371 listings), `scraper/download_abo_images.py` -> `data/processed/abo_images.csv` (photo manifest by `item_id`).
* **Labels:** from the listing's category path (the store's own category, as for mebel.kz), kept only when Amazon's product type agrees (this drops door handles filed under wardrobes, TV wall mounts under TV stands, pet beds under beds). Accessories, outdoor, children's and storage-organiser paths are skipped.
* **Fields:** height, width, length (converted to cm), weight (kg), materials (same 16 tokens as mebel.kz, matched on the English text), colour, photos. **No price.** ABO's width/length order is not consistent across listings (for many sofas "width" is the depth), so models compare sizes as height, larger and smaller horizontal side.
* **Class balance:** sofa 1,047, armchair 965, chair 607, bed 259, coffee table 202, desk 136, wardrobe 66, chest of drawers 53, TV stand 36. The three case-furniture classes we care most about are the smallest.
* **Duplicates:** the same product is often listed on several marketplaces (614 repeated model numbers, 328 repeated main photos). Listings that share a model number or a photo are one group in every split.

## Ethics
Only public catalogue data was collected: no accounts, no personal data, no reviews. The collection was rate-limited and identified itself. The data is used for a university course project only.
