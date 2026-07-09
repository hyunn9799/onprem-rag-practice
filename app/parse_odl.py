# -*- coding: utf-8 -*-
r"""
① 문서 파싱/청킹: PDF 4권 → hyosung_chunks.json

파이프라인 (CLAUDE.md 확정 아키텍처):
  1. 한컴 오픈데이터로더(ODL)로 PDF 파싱 → JSON
  2. JSON 요소를 페이지 단위로 재조립 (1 페이지 = 1 청크, 마크다운 보존)
  3. ODL 텍스트가 100자 미만인 페이지만 pytesseract 한국어 OCR 보완
  4. 최종 텍스트가 사실상 빈 페이지는 청크에서 제외
"""
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import fitz  # pymupdf
import opendataloader_pdf
import pytesseract
from PIL import Image

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
ODL_DIR = BASE / "out_odl"
TESSDATA_DIR = BASE / "tessdata"  # kor/eng traineddata
OUT_PATH = BASE / "hyosung_chunks.json"

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

DOCS = [
    ("2021_Hyosung_Power_Technology_Magazine_vol.1_kor.pdf", "hyosung_vol1",
     "효성중공업 전력기술 매거진 Vol.1 (2021)"),
    ("2022_Hyosung_Power_Technology_Magazine_vol.2_kor.pdf", "hyosung_vol2",
     "효성중공업 전력기술 매거진 Vol.2 (2022)"),
    ("2023_Hyosung_Power_Technology_Magazine_vol.3_kor.pdf", "hyosung_vol3",
     "효성중공업 전력기술 매거진 Vol.3 (2023)"),
    ("2024_Hyosung_Power_Technology_Magazine_vol.4_kor.pdf", "hyosung_vol4",
     "효성중공업 전력기술 매거진 Vol.4 (2024)"),
]

# ODL 추출이 사실상 실패한 페이지만 OCR로 대체한다.
# 100~300자 페이지는 ODL이 이미 핵심 텍스트를 건진 상태라, OCR을 덧붙이면
# 같은 내용이 음절 단위로 띄어진 노이즈로 중복됨 (vol2 p4에서 확인).
OCR_TRIGGER = 100  # ODL 텍스트가 이보다 짧으면 OCR 보완
MIN_CHUNK = 30     # OCR 후에도 이보다 짧으면 내용 없는 페이지로 간주하고 제외

# "04 HYOSUNG HEAVY INDUSTRIES 05" 형태의 러닝헤더(쪽번호+매거진명) 제거
RUNNING_HEADER = re.compile(
    r"^#{0,6}\s*\d{1,2}\s+HYOSUNG HEAVY INDUSTRIES(\s+\d{1,2})?\s*$", re.MULTILINE
)


# ---------- ODL JSON → 페이지별 마크다운 ----------

def node_text(node):
    """노드에서 순수 텍스트만 재귀 추출 (표 셀 내용 등)."""
    if node.get("type") == "image":
        return ""
    content = node.get("content")
    if isinstance(content, str):
        return content
    parts = (node_text(k) for k in node.get("kids", []) or [])
    return " ".join(p for p in parts if p)


def table_to_md(tbl):
    ncols = tbl.get("number of columns", 0)
    if not ncols:
        return ""
    lines = []
    for i, row in enumerate(tbl.get("rows", [])):
        cells = [""] * ncols
        for c in row.get("cells", []):
            col = c.get("column number", 1) - 1
            txt = " ".join(filter(None, (node_text(k) for k in c.get("kids", []) or [])))
            if 0 <= col < ncols:
                cells[col] = txt.replace("|", "/").replace("\n", " ")
        lines.append("| " + " | ".join(cells) + " |")
        if i == 0:
            lines.append("|" + "---|" * ncols)
    return "\n".join(lines)


def node_to_md(node):
    t = node.get("type", "")
    if t == "image":
        return ""
    if t == "heading":
        lvl = int(node.get("heading level") or 2)
        return "#" * min(lvl + 1, 6) + " " + (node.get("content") or "")
    if t == "table":
        return table_to_md(node)
    if t == "list":
        items = node.get("list items", [])
        return "\n".join("- " + ((it.get("content") or node_text(it)).strip())
                         for it in items)
    if t == "text block":
        parts = (node_to_md(k) for k in node.get("kids", []) or [])
        return "\n\n".join(p for p in parts if p)
    if "content" in node:
        return node.get("content") or ""
    return node_text(node)


def clean_page_md(md):
    md = RUNNING_HEADER.sub("", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


# ---------- OCR 보완 ----------

def ocr_page(pdf_path, page_no):
    """페이지를 300dpi로 렌더링해 한국어+영어 OCR."""
    with fitz.open(pdf_path) as doc:
        pix = doc[page_no - 1].get_pixmap(dpi=300)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
    # 주의: Windows에서 pytesseract는 config의 따옴표를 벗기지 않고 그대로 전달함
    cfg = f"--tessdata-dir {TESSDATA_DIR}"
    txt = pytesseract.image_to_string(img, lang="kor+eng", config=cfg)
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


# ---------- 메인 ----------

def main():
    ODL_DIR.mkdir(exist_ok=True)
    chunks = []
    stats = []

    for fname, doc_id, source in DOCS:
        pdf = DATA_DIR / fname
        jpath = ODL_DIR / (pdf.stem + ".json")
        if not jpath.exists():
            print(f"[odl] parsing {fname} ...", flush=True)
            opendataloader_pdf.convert(str(pdf), output_dir=str(ODL_DIR),
                                       format=["json"], quiet=True)
        d = json.load(open(jpath, encoding="utf-8"))

        pages = defaultdict(list)
        for node in d["kids"]:
            md = node_to_md(node)
            if md.strip():
                pages[node.get("page number")].append(md)

        n_pages = d["number of pages"]
        n_ocr, n_skip = 0, 0
        for p in range(1, n_pages + 1):
            text = clean_page_md("\n\n".join(pages.get(p, [])))
            extraction = "odl"
            if len(text) < OCR_TRIGGER:
                odl_len = len(text)
                ocr = ocr_page(pdf, p)
                if len(ocr) >= MIN_CHUNK:
                    text = (text + "\n\n" + ocr).strip() if text else ocr
                    extraction = "ocr"
                    n_ocr += 1
                    print(f"[ocr] {doc_id} p{p}: odl={odl_len}c + ocr={len(ocr)}c", flush=True)
            if len(text) < MIN_CHUNK:
                n_skip += 1
                print(f"[skip] {doc_id} p{p}: only {len(text)} chars", flush=True)
                continue
            chunks.append({
                "chunk_id": f"{doc_id}_p{p:03d}",
                "doc_id": doc_id,
                "source": source,
                "file_name": fname,
                "page": p,
                "text": text,
                "extraction": extraction,
            })
        stats.append((doc_id, n_pages, n_ocr, n_skip))

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print("\n=== summary ===")
    for doc_id, n_pages, n_ocr, n_skip in stats:
        print(f"{doc_id}: pages={n_pages} ocr={n_ocr} skipped={n_skip}")
    print(f"total chunks: {len(chunks)} -> {OUT_PATH}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()