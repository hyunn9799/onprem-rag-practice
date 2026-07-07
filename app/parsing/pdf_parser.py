"""PDF 파서 (1번 담당 영역) — data/raw/*.pdf → data/processed/*.jsonl.

페이지 단위 청킹(요구사항)을 위해 **페이지당 한 레코드**로 출력한다.
    한 줄 = {"doc_id": <파일stem>, "text": <페이지 텍스트>, "page": <1-base>, "source": <파일명>}
이 레코드가 적재에서 page 전략으로 그대로 1 청크(chunk_id = doc_id#p{page})가 된다.

파이프라인:  make parse (이 파일) → make ingest (data/processed 를 읽어 적재)

표 추출은 지금은 extract_text 가 잡는 흐름 텍스트로 갈음한다(중복 방지). 구조적 표
추출(마크다운 표)은 확장 포인트로 남긴다 — page.extract_tables() 사용.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import pdfplumber

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"


def make_doc_id(pdf_path: Path) -> str:
    """파일명에서 짧고 안정적인 doc_id 를 만든다.

    '2023_Hyosung_..._vol.3_kor.pdf' -> 'hyosung-2023-vol3'
    규칙에 안 맞으면 파일 stem 을 슬러그화(소문자-하이픈).
    """
    stem = pdf_path.stem
    year = re.search(r"(20\d{2})", stem)
    vol = re.search(r"vol[._\s]*?(\d+)", stem, re.IGNORECASE)
    if year and vol:
        return f"hyosung-{year.group(1)}-vol{vol.group(1)}"
    return re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")


def extract_pages(pdf_path: Path) -> List[Dict]:
    """PDF 한 개 → 페이지별 레코드. [{"page": 1, "text": "..."}, ...]"""
    pages: List[Dict] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append({"page": i + 1, "text": text})
    return pages


def pdf_to_records(pdf_path: Path) -> List[Dict]:
    """PDF 한 개 → 페이지당 적재 레코드. 빈(이미지 전용) 페이지는 건너뛴다."""
    doc_id = make_doc_id(pdf_path)
    source = pdf_path.name
    records: List[Dict] = []
    for p in extract_pages(pdf_path):
        text = (p.get("text") or "").strip()
        if not text:
            continue
        records.append(
            {"doc_id": doc_id, "text": text, "page": p["page"], "source": source}
        )
    return records


def parse_all(raw_dir: str = RAW_DIR, processed_dir: str = PROCESSED_DIR) -> int:
    raw, out = Path(raw_dir), Path(processed_dir)
    out.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(raw.glob("*.pdf"))
    if not pdfs:
        print(f"{raw_dir} 에 PDF 가 없습니다. (data/raw 에 PDF 를 넣으세요)")
        return 0

    total = 0
    for pdf in pdfs:
        records = pdf_to_records(pdf)
        dest = out / f"{pdf.stem}.jsonl"
        with dest.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        total += len(records)
        print(f"{pdf.name} -> {dest.name} ({len(records)} pages)")
    return total


if __name__ == "__main__":
    n = parse_all()
    print(f"parsed page-records: {n}")
