"""Download the 256 px ABO catalogue photos of the furniture listings in data/processed/abo_furniture.csv.

Main photo plus up to 3 other views per listing. Files go to data/external/abo/images/small/<path>,
the same layout as the ABO archive. Already downloaded files are skipped, so the script can be re-run.
Writes data/processed/abo_images.csv: item_id, image_id, view (0 = main), path, ok.

Run: .venv/bin/python scraper/download_abo_images.py
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
ABO = ROOT / "data" / "external" / "abo"
BASE = "https://amazon-berkeley-objects.s3.amazonaws.com/images/small/"
MAX_OTHER_VIEWS = 3
WORKERS = 8


def main():
    d = pd.read_csv(ROOT / "data" / "processed" / "abo_furniture.csv")
    paths = pd.read_csv(ABO / "images.csv.gz").set_index("image_id")["path"]
    rows = []
    for _, r in d.iterrows():
        others = r["other_image_ids"].split("|")[:MAX_OTHER_VIEWS] if isinstance(r["other_image_ids"], str) else []
        for view, img in enumerate([r["main_image_id"], *others]):
            if isinstance(img, str) and img in paths.index:
                rows.append({"item_id": r["item_id"], "domain": r["domain"], "image_id": img, "view": view,
                             "path": paths[img]})
    man = pd.DataFrame(rows)
    todo = man.drop_duplicates("image_id")

    session = requests.Session()
    session.headers["User-Agent"] = "student-course-project (ABO dataset download)"

    def fetch(rel):
        dest = ABO / "images" / "small" / rel
        if dest.exists() and dest.stat().st_size > 0:
            return rel, True
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            resp = session.get(BASE + rel, timeout=30)
            if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image/"):
                dest.write_bytes(resp.content)
                return rel, True
        except requests.RequestException:
            pass
        return rel, False

    with ThreadPoolExecutor(WORKERS) as pool:
        ok = dict(pool.map(fetch, todo["path"]))
    man["ok"] = man["path"].map(ok)
    man.to_csv(ROOT / "data" / "processed" / "abo_images.csv", index=False)
    print(f"{len(todo)} unique images, downloaded or present: {sum(ok.values())}, failed: {len(ok) - sum(ok.values())}")


if __name__ == "__main__":
    main()
