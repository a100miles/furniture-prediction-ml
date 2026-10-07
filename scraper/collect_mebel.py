"""Collect furniture product pages from mebel.kz.

Reads the public sitemap, keeps product URLs whose slug looks like one of the
nine furniture types, downloads each page politely (one request at a time,
with a pause) and stores the structured data in data/raw/mebel_raw.jsonl.

Run from the project root:
    python scraper/collect_mebel.py            # full run, resumable
    python scraper/collect_mebel.py --limit 20 # quick test
"""
import argparse
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

SITEMAP_URL = "https://mebel.kz/sitemap.almaty.xml"
ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "mebel_raw.jsonl"
LOG_PATH = ROOT / "data" / "raw" / "collection_log.json"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) student-research-scraper "
    "(Astana IT University ML course project; 1 request/second)"
)
# Politeness: at most WORKERS connections at once and at least
# MIN_INTERVAL seconds between the starts of any two requests.
WORKERS = 2
MIN_INTERVAL = 0.7

# Slug words for the nine target types. Accessories (covers, armrests, legs)
# are excluded. The final label comes from the site's own category later.
SLUG_KEEP = re.compile(r"divan|kreslo|stul|krovat|shkaf|komod|stol|tumba")
SLUG_DROP = re.compile(
    r"chehol|podlokotnik|modul-dlya|osnovanie|nakidka|podushk|matras|polka|"
    r"zerkalo|ruchk|opora|nozhk|kuhonnyj-stol|obedennyj-stol|prikrovatn|"
    r"ofisnoe-kreslo|barnyj|polubarnyj|stol-transformer"
)

# Rough type guess from the URL, used only to cap how many pages we fetch per
# type (the real label comes from the page's category later).
SLUG_GUESS = [
    ("chest_of_drawers", r"komod"),
    ("tv_stand", r"tv-tumba|tumba-pod-tv|tumba-tv|tumba-dlya-tv"),
    ("coffee_table", r"jurnal|zhurnal|kofejn"),
    ("desk", r"pismenn|kompyutern|stol"),
    ("wardrobe", r"shkaf"),
    ("bed", r"krovat"),
    ("sofa", r"divan|kanape"),
    ("armchair", r"kreslo"),
    ("chair", r"stul"),
]
CAP_PER_TYPE = 700
SEED = 42


def slug_guess(url):
    slug = url.rsplit("/", 1)[-1]
    for label, pattern in SLUG_GUESS:
        if re.search(pattern, slug):
            return label
    return "other"


def capped(urls, done, cap):
    """Keep at most `cap` URLs per guessed type: pages already saved first,
    then a reproducible random sample of the rest."""
    by_type = {}
    for u in urls:
        by_type.setdefault(slug_guess(u), []).append(u)
    rng = random.Random(SEED)
    keep = []
    for label in sorted(by_type):
        group = by_type[label]
        saved = [u for u in group if u in done]
        rest = [u for u in group if u not in done]
        rng.shuffle(rest)
        keep += saved[:cap] + rest[: max(0, cap - len(saved))]
    return sorted(keep)


_lock = threading.Lock()
_last_start = [0.0]


def wait_turn():
    with _lock:
        pause = _last_start[0] + MIN_INTERVAL - time.monotonic()
        if pause > 0:
            time.sleep(pause)
        _last_start[0] = time.monotonic()


def get(session, url, tries=4):
    for attempt in range(tries):
        wait_turn()
        try:
            r = session.get(url, timeout=30)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(2 ** (attempt + 1))
    return None


def product_urls(session):
    xml = get(session, SITEMAP_URL).text
    locs = [loc.text.strip() for loc in BeautifulSoup(xml, "xml").find_all("loc")]
    urls = []
    for u in locs:
        if "/product/" not in u:
            continue
        slug = u.rsplit("/", 1)[-1]
        if SLUG_KEEP.search(slug) and not SLUG_DROP.search(slug):
            urls.append(u)
    return sorted(set(urls)), sum("/product/" in u for u in locs)


def parse_product(html, url):
    soup = BeautifulSoup(html, "lxml")
    product, breadcrumbs = None, []
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except json.JSONDecodeError:
            continue
        if data.get("@type") == "Product":
            product = {
                k: data.get(k)
                for k in ("name", "sku", "category", "color", "description")
            }
            offers = data.get("offers") or {}
            product["price"] = offers.get("price")
            product["currency"] = offers.get("priceCurrency")
            product["availability"] = offers.get("availability")
        elif data.get("@type") == "BreadcrumbList":
            breadcrumbs = [i.get("name") for i in data.get("itemListElement", [])]

    # Specification block: <div class="StringParameter-module__..__parameter">
    #   <span class="..__name">Name: </span><span> value</span></div>
    # and the dimension block "Размеры: Длина X см х Ширина Y см х Высота Z см".
    specs = {}
    for div in soup.find_all("div", class_=re.compile(r"StringParameter-module__\w+__parameter")):
        spans = div.find_all("span", recursive=False)
        if len(spans) >= 2:
            name = spans[0].get_text(" ", strip=True).rstrip(":").strip()
            value = " ".join(s.get_text(" ", strip=True) for s in spans[1:]).strip()
            if name and name not in specs:
                specs[name] = value

    series = soup.find("span", attrs={"data-testid": "link-in-series-characteristic"})
    if series:
        specs["Серия"] = series.get_text(" ", strip=True)

    # The overall size comes right after "Размеры" and before the seat /
    # sleeping-place sizes. Two layouts exist: "Длина 250 см" on one line, or
    # "Длина" and "250 см" on separate lines; depth may be "Глубина габаритная".
    text = soup.get_text("\n", strip=True)
    dims = {}
    # (anchored on "Длина" so the size *picker* "Размеры (3)" is skipped)
    m = re.search(r"\nРазмеры:?\n(?=Длина)", text)
    if m:
        block = text[m.end(): m.end() + 400].replace("\n", " ")
        block = re.split(r"Посадочное место|Спальное место|В разложенном|Механизм|Материал|Каркас|Цвет", block)[0]
        for key, val in re.findall(r"(Длина|Ширина|Высота|Глубина)(?:\s+габаритная)?\s+([\d.,]+)\s*см", block):
            dims.setdefault(key, val)

    return {
        "url": url,
        "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "product": product,
        "breadcrumbs": breadcrumbs,
        "dimensions": dims,
        "specs": specs,
    }


def refetch_incomplete():
    """Re-download saved pages with fewer than three dimensions (written before
    the parser knew the second "Размеры" layout) and replace them in place."""
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    records = [json.loads(line) for line in RAW_PATH.open()]
    todo = [i for i, r in enumerate(records) if len(r.get("dimensions") or {}) < 3]
    print(f"{len(todo)} saved pages with incomplete sizes", flush=True)

    def fetch(i):
        r = get(session, records[i]["url"])
        return i, None if r is None else parse_product(r.text, records[i]["url"])

    fixed = 0
    with ThreadPoolExecutor(WORKERS) as pool:
        for i, record in pool.map(fetch, todo):
            if record is not None:
                fixed += len(record["dimensions"]) >= 3
                records[i] = record
    with RAW_PATH.open("w") as out:
        for r in records:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    log = json.loads(LOG_PATH.read_text())
    log.setdefault("refetches", []).append({
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pages_refetched": len(todo), "now_complete": fixed,
    })
    LOG_PATH.write_text(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"refetched {len(todo)}, {fixed} now have all three sizes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cap", type=int, default=CAP_PER_TYPE, help="max pages per guessed type")
    parser.add_argument("--refetch-incomplete", action="store_true",
                        help="download again the saved pages whose size block was not parsed")
    args = parser.parse_args()
    if args.refetch_incomplete:
        return refetch_incomplete()

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    urls, n_products_in_sitemap = product_urls(session)
    done = set()
    if RAW_PATH.exists():
        with RAW_PATH.open() as f:
            done = {json.loads(line)["url"] for line in f if line.strip()}
    n_candidates = len(urls)
    urls = capped(urls, done, args.cap)
    todo = [u for u in urls if u not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{n_products_in_sitemap} products in sitemap, {n_candidates} candidates, {len(urls)} after cap, "
          f"{len(done)} already saved, {len(todo)} to fetch (cap {args.cap} per type)", flush=True)

    def fetch(url):
        r = get(session, url)
        return url, None if r is None else parse_product(r.text, url)

    ok = failed = 0
    with RAW_PATH.open("a") as out, ThreadPoolExecutor(WORKERS) as pool:
        for i, (url, record) in enumerate(pool.map(fetch, todo), 1):
            if record is None:
                failed += 1
            else:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                ok += 1
            if i % 100 == 0:
                print(f"{i}/{len(todo)} ok={ok} failed={failed}", flush=True)

    log = json.loads(LOG_PATH.read_text()) if LOG_PATH.exists() else {"runs": []}
    log.update({
        "source": "https://mebel.kz/",
        "sitemap": SITEMAP_URL,
        "user_agent": USER_AGENT,
        "parallel_connections": WORKERS,
        "min_seconds_between_requests": MIN_INTERVAL,
        "products_in_sitemap": n_products_in_sitemap,
        "candidate_urls": n_candidates,
        "candidate_urls_after_cap": len(urls),
        "cap_per_type": args.cap,
        "sample_seed": SEED,
    })
    log["runs"].append({
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fetched_ok": ok,
        "failed": failed,
    })
    LOG_PATH.write_text(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"done: ok={ok} failed={failed}")


if __name__ == "__main__":
    main()
