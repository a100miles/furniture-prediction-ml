"""Build the Final slides from slides/final/ (deck.json, deck.css, slides/*.html).

Writes:
  slides/Furniture Final Slides.html   one page per slide, open in a browser (arrow keys / space to move)
  slides/speaker-notes.md              the speaker notes of every slide
  Furniture Final Slides.pdf           (with --pdf) printed with headless Edge or Chrome

Run: .venv/bin/python tools/build_slides.py --pdf
"""
import html
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "slides" / "final"
OUT_HTML = ROOT / "slides" / "Furniture Final Slides.html"
OUT_NOTES = ROOT / "slides" / "speaker-notes.md"
OUT_PDF = ROOT / "Furniture Final Slides.pdf"

PAGE_CSS = """
html, body { margin: 0; padding: 0; background: #111; }
section { position: relative; width: 1920px; height: 1080px; overflow: hidden; margin: 0 auto 24px; }
aside { display: none; }
@page { size: 1920px 1080px; margin: 0; }
@media print {
  html, body { background: none; }
  section { margin: 0; break-after: page; }
  * { animation: none !important; }
}
@media screen {
  body.one section { display: none; margin: 0 auto; }
  body.one section.on { display: flex; }
}
"""
NAV_JS = """
const S = [...document.querySelectorAll('section')]; let i = 0;
const fit = () => { const k = Math.min(innerWidth / 1920, innerHeight / 1080);
  S.forEach(s => { s.style.transform = document.body.classList.contains('one') ? `scale(${k})` : '';
                   s.style.transformOrigin = 'top left'; }); };
const show = () => { S.forEach((s, j) => s.classList.toggle('on', j === i)); fit(); };
document.body.classList.add('one'); show(); addEventListener('resize', fit);
addEventListener('keydown', e => {
  if (['ArrowRight', 'PageDown', ' '].includes(e.key)) i = Math.min(i + 1, S.length - 1);
  else if (['ArrowLeft', 'PageUp'].includes(e.key)) i = Math.max(i - 1, 0);
  else if (e.key === 'Home') i = 0; else if (e.key === 'End') i = S.length - 1; else return;
  e.preventDefault(); show(); });
addEventListener('click', () => { i = Math.min(i + 1, S.length - 1); show(); });
"""


def main():
    deck = json.loads((SRC / "deck.json").read_text(encoding="utf-8"))
    fonts = "\n".join(f'<link rel="stylesheet" href="{f["href"]}">' for f in deck["faces"].values() if "href" in f)
    slides, notes = [], [f"# {deck['title']}: speaker notes\n"]
    for n, sid in enumerate(deck["order"], 1):
        text = (SRC / "slides" / f"{sid}.html").read_text(encoding="utf-8").strip()
        slides.append(text)
        m = re.search(r"<aside>(.*?)</aside>", text, re.S)
        if m:
            notes.append(f"## Slide {n} ({sid})\n\n{html.unescape(m.group(1).strip())}\n")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(deck['title'])}</title>
{fonts}
<style>{(SRC / 'deck.css').read_text(encoding='utf-8')}
{PAGE_CSS}</style></head>
<body>
{chr(10).join(slides)}
<script>{NAV_JS}</script>
</body></html>
"""
    OUT_HTML.write_text(page, encoding="utf-8")
    OUT_NOTES.write_text("\n".join(notes), encoding="utf-8")
    print("wrote", OUT_HTML.relative_to(ROOT), "and", OUT_NOTES.relative_to(ROOT), f"({len(slides)} slides)")

    if "--pdf" in sys.argv:
        # print a copy without the navigation script, so every slide is on the page
        printable = OUT_HTML.with_name("_print.html")
        printable.write_text(page.replace(f"<script>{NAV_JS}</script>", ""), encoding="utf-8")
        browser = next((b for b in [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                                    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                                    shutil.which("chromium"), shutil.which("google-chrome")] if b and Path(b).exists()), None)
        if browser is None:
            sys.exit("no Edge or Chrome found for the PDF")
        subprocess.run([browser, "--headless", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=10000",
                        f"--print-to-pdf={OUT_PDF}", printable.as_uri()], check=True, capture_output=True)
        printable.unlink()
        print("wrote", OUT_PDF.relative_to(ROOT))


if __name__ == "__main__":
    main()
