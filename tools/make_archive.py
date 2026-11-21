"""Build Final_Zhumagali_Sultanbek_SE-2420.zip next to the project folder (not inside it).

Run: .venv/bin/python tools/make_archive.py   (put "Furniture Final Slides.pdf" in the project folder first)
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]).resolve()
OUT = ROOT.parent / "Final_Zhumagali_Sultanbek_SE-2420.zip"
TOP = "Final_Zhumagali_Sultanbek_SE-2420"

INCLUDE_FILES = ["final.ipynb", "endterm.ipynb", "furniture_classification.ipynb", "main.ipynb", "main.py", "part1.py",
                 "api.py", "app.py", "requirements.txt", "README.md", "CHANGES.md", "DATA_CARD.md", "CONTRIBUTIONS.md",
                 "Furniture Type Midterm Slides.pdf", "Furniture Type Endterm Slides.pdf", "Furniture Final Slides.pdf",
                 ".gitignore"]
INCLUDE_DIRS = ["furniture", "tools", "scraper", "models", "data/processed", "data/raw", "week3-4", "frontend"]
SKIP_PARTS = {"__pycache__", "node_modules", ".ipynb_checkpoints"}
SKIP_SUFFIX = {".pyc", ".tsbuildinfo"}

files = []
for f in INCLUDE_FILES:
    p = ROOT / f
    if p.exists():
        files.append(p)
    else:
        print("missing (not added):", f)
for d in INCLUDE_DIRS:
    for p in sorted((ROOT / d).rglob("*")):
        if p.is_file() and not (set(p.relative_to(ROOT).parts) & SKIP_PARTS) and p.suffix not in SKIP_SUFFIX:
            files.append(p)

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in files:
        z.write(p, f"{TOP}/{p.relative_to(ROOT).as_posix()}")
print(OUT, f"{OUT.stat().st_size / 1e6:.1f} MB", len(files), "files")
