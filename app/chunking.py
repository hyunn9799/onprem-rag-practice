"""문서 청킹 (순수 로직, 무거운 의존성 없음 → 단위 테스트 용이).

학습용 문자수 슬라이딩 윈도우. size > overlap 을 강제해 무한 루프를 막는다.
출처 메타데이터(page/section/source)는 각 chunk 로 전파한다.
실무 확장: 문단/문장 경계 우선 split + 표/제목 보존.
"""
from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# chunk 에 함께 실어 나르는 출처 메타데이터 키(있을 때만).
META_FIELDS = ("page", "section", "source")


def chunk_document(
    doc_id: str,
    text: str,
    size: int = 500,
    overlap: int = 80,
    meta: Dict | None = None,
) -> List[Dict]:
    if size <= overlap:
        raise ValueError(f"size({size}) 는 overlap({overlap}) 보다 커야 합니다(무한 루프 방지).")
    meta = meta or {}
    chunks: List[Dict] = []
    start, idx = 0, 0
    while start < len(text):
        piece = text[start : start + size].strip()
        if piece:
            chunk = {"chunk_id": f"{doc_id}#{idx}", "doc_id": doc_id, "text": piece}
            chunk.update(meta)  # page/section/source 전파
            chunks.append(chunk)
            idx += 1
        start += size - overlap
    return chunks


def to_chunks(
    doc: Dict,
    strategy: str = "page",
    size: int = 500,
    overlap: int = 80,
    index: int = 0,
) -> List[Dict]:
    """문서 레코드 하나 → chunk 리스트.

    - "page" 전략: 1 레코드(=1 페이지) = 1 청크. chunk_id = doc_id#p{page}.
      (요구사항: 페이지 단위 청킹. 페이지가 BGE-M3 8192 토큰에 들어가는 게 확인됨)
    - "char" 전략: 문자수 슬라이딩 윈도우로 재분할.

    index: page 가 없을 때 chunk_id 충돌을 막기 위한 배치 내 위치(유니크 suffix).
    """
    meta = {k: doc[k] for k in META_FIELDS if k in doc}
    if strategy == "page":
        page = doc.get("page")
        if page is not None:
            suffix = f"p{page}"
        else:
            # page 없으면 index 로 유니크 보장(안 그러면 같은 doc_id 가 전부 #0 로 덮어씀).
            suffix = str(index)
            logger.warning(
                "page 전략인데 page 가 없어 index(%d)로 대체합니다 — 출처 추적 저하: doc_id=%s",
                index,
                doc.get("doc_id"),
            )
        chunk = {"chunk_id": f"{doc['doc_id']}#{suffix}", "doc_id": doc["doc_id"], "text": doc["text"]}
        chunk.update(meta)
        return [chunk]
    return chunk_document(doc["doc_id"], doc["text"], size=size, overlap=overlap, meta=meta)
