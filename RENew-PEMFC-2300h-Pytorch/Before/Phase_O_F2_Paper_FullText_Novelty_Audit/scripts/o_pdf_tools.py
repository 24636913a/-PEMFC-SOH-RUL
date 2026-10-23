"""Internal literature extraction/rendering only. No project data/model imports."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import fitz

PHASE = Path(__file__).resolve().parents[1]
REPO = PHASE.parent.parent
PAPERS = REPO / "参考论文"
RUNTIME = PHASE / "runtime"


def inventory():
    rows = []
    RUNTIME.mkdir(parents=True, exist_ok=True)
    for i, source in enumerate(sorted(PAPERS.glob("*.pdf")), 1):
        doc = fitz.open(source)
        first = doc[0].get_text(sort=True)
        key = f"local_{i:02d}"
        dest = RUNTIME / key
        dest.mkdir(exist_ok=True)
        (dest/"first_page.txt").write_text(first,encoding="utf-8")
        doi = re.findall(r"10\.\d{4,9}/[^\s<>]+", first, re.I)
        all_text = []
        for n, page in enumerate(doc, 1):
            text = page.get_text(sort=True)
            (dest/f"page_{n:02d}.txt").write_text(text,encoding="utf-8")
            all_text.append(f"\n=== PDF PAGE {n} ===\n"+text)
        (dest/"full_text.txt").write_text("".join(all_text),encoding="utf-8")
        rows.append({"local_id":key,"path":source.relative_to(REPO).as_posix(),
                     "pages":len(doc),"sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
                     "first_page_doi_candidates":"; ".join(dict.fromkeys(doi)),
                     "first_page_text":first,"metadata":doc.metadata})
    out = PHASE/"evidence"
    out.mkdir(exist_ok=True)
    # Full texts remain ignored internal material; compact identity is refined by reading.
    (RUNTIME/"inventory.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps([{k:v for k,v in r.items() if k not in ("first_page_text","metadata")} for r in rows],ensure_ascii=False,indent=2))


def render(filename, pages):
    path = (PAPERS/filename).resolve()
    if path.parent != PAPERS.resolve() or path.suffix != ".pdf":
        raise ValueError("LITERATURE_PATH_ONLY")
    doc = fitz.open(path)
    target = RUNTIME/"rendered"/path.stem
    target.mkdir(parents=True,exist_ok=True)
    for p in pages:
        if not 1 <= p <= len(doc):
            raise ValueError("INVALID_PDF_PAGE")
        output = target/f"page_{p:02d}.png"
        doc[p-1].get_pixmap(matrix=fitz.Matrix(1.6,1.6)).save(output)
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode",choices=["inventory","render"])
    parser.add_argument("--filename")
    parser.add_argument("--pages",type=int,nargs="+")
    args = parser.parse_args()
    if args.mode == "inventory":
        inventory()
    else:
        render(args.filename,args.pages)
