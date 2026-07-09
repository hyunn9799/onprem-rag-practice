"""문서 청킹 (순수 로직, 무거운 의존성 없음 → 단위 테스트 용이).

chunk_id 규칙 — 결정론이 핵심이다(재적재 시 같은 내용이면 같은 id 로 upsert):
- page 가 있으면 suffix = p{page}           예: doc#p12
- page 가 없으면 suffix = 내용 해시 8자      예: doc#a3f2b1c8
  (배치 내 위치를 쓰면 호출/파일 순서에 따라 id 가 바뀌어 stale 중복이 쌓인다)
- char 서브청킹은 레코드 네임스페이스 밑에 번호를 붙인다: doc#p12-0, doc#a3f2b1c8-1
  (네임스페이스 없이 번호만 쓰면 같은 doc_id 의 페이지들끼리 id 가 충돌해 덮어쓴다)

page 전략인데 page 가 없는 레코드는 char 로 폴백한다 — 그대로 두면 API 로 들어온
긴 문서가 통짜 1청크가 되어 임베딩이 8192 토큰에서 조용히 잘리기 때문.
빈/공백 텍스트 레코드는 배치 전체를 실패시키지 않도록 건너뛴다(경고 로그).
"""
from __future__ import annotations

import hashlib
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# 출처 메타데이터 계약은 app/schema.py 가 단일 소스 (여기선 사용만).
from app.schema import META_FIELDS


def _content_suffix(text: str) -> str:
    """page 없는 레코드용 결정론적 suffix — 내용이 같으면 항상 같은 id."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def chunk_document(
    doc_id: str,
    text: str,
    size: int = 500,
    overlap: int = 80,
    meta: Dict | None = None,
    id_prefix: str | None = None,
) -> List[Dict]:
    """문자수 슬라이딩 윈도우 청킹.

    id_prefix: 레코드 네임스페이스(예: "p12"). 주면 chunk_id = doc#p12-0, doc#p12-1 ...
    """
    if size <= overlap:
        raise ValueError(f"size({size}) 는 overlap({overlap}) 보다 커야 합니다(무한 루프 방지).")
    meta = meta or {}
    base = f"{doc_id}#{id_prefix}-" if id_prefix else f"{doc_id}#"
    chunks: List[Dict] = []
    start, idx = 0, 0
    while start < len(text):
        piece = text[start : start + size].strip()
        if piece:
            chunk = {"chunk_id": f"{base}{idx}", "doc_id": doc_id, "text": piece}
            chunk.update(meta)  # META_FIELDS(page/source) 전파
            chunks.append(chunk)
            idx += 1
        start += size - overlap
    return chunks


def to_chunks(
    doc: Dict,
    strategy: str = "page",
    size: int = 500,
    overlap: int = 80,
) -> List[Dict]:
    """문서 레코드 하나 → chunk 리스트.

    - "page" 전략: 1 레코드(=1 페이지) = 1 청크 (chunk_id = doc#p{page}).
      page 가 없으면 char 폴백(통짜 청크로 인한 임베딩 잘림 방지).
    - "char" 전략: 레코드 네임스페이스(p{page} 또는 내용해시) 밑에서 슬라이딩 윈도우.
    """
    text = (doc.get("text") or "").strip()
    if not text:
        # 빈 레코드 하나가 임베딩 단계에서 배치 전체를 실패시키지 않도록 skip.
        logger.warning("빈 텍스트 레코드 건너뜀: doc_id=%s", doc.get("doc_id"))
        return []

    meta = {k: doc[k] for k in META_FIELDS if k in doc}
    page = doc.get("page")

    if strategy == "page" and page is not None:
        chunk = {"chunk_id": f"{doc['doc_id']}#p{page}", "doc_id": doc["doc_id"], "text": text}
        chunk.update(meta)
        return [chunk]

    if strategy == "page":
        logger.warning(
            "page 전략인데 page 가 없어 char 폴백(내용해시 네임스페이스): doc_id=%s",
            doc.get("doc_id"),
        )

    prefix = f"p{page}" if page is not None else _content_suffix(text)
    return chunk_document(doc["doc_id"], text, size=size, overlap=overlap, meta=meta, id_prefix=prefix)
