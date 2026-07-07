"""문서 적재 파이프라인: chunk -> 임베딩 -> Milvus + OpenSearch 동시 저장.

실무 포인트:
- chunk_id 를 doc_id#순번 으로 결정론적으로 만든다. 재적재 시 두 store 모두 upsert 라
  중복이 쌓이지 않고, RRF 병합 키가 항상 일치한다.
- 출처 메타데이터(page/section/source)는 chunk 에 실어 두 store 로 함께 전파한다.
- chunker 는 학습용 문자수 슬라이딩 윈도우(size > overlap 검증). 실제로는 문단/문장 경계 우선.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

from app.chunking import to_chunks
from app.config import settings
from app.embeddings.bge_m3_embedder import BGEM3Embedder
from app.retrievers.milvus_retriever import MilvusRetriever
from app.retrievers.opensearch_retriever import OpenSearchRetriever


def ingest(documents: List[Dict], embedder: BGEM3Embedder | None = None) -> int:
    """documents: [{"doc_id": str, "text": str, "page"?, "section"?, "source"?}, ...]"""
    embedder = embedder or BGEM3Embedder()
    milvus, opensearch = MilvusRetriever(), OpenSearchRetriever()
    milvus.ensure_collection()
    opensearch.ensure_index()

    all_chunks: List[Dict] = []
    for i, d in enumerate(documents):
        all_chunks.extend(
            to_chunks(
                d,
                strategy=settings.chunk_strategy,
                size=settings.chunk_size,
                overlap=settings.chunk_overlap,
                index=i,
            )
        )

    if not all_chunks:
        return 0

    embeddings = embedder.embed_texts([c["text"] for c in all_chunks])
    milvus.upsert(all_chunks, embeddings)      # 벡터 검색용
    opensearch.index_chunks(all_chunks)        # BM25 검색용
    return len(all_chunks)


def load_documents(processed_dir: str = "data/processed") -> List[Dict]:
    """data/processed/*.jsonl 에서 파싱된 문서를 읽는다.

    한 줄 = 문서 하나: {"doc_id": str, "text": str, ...}. '#' 줄은 주석으로 무시.
    PDF -> 이 JSONL 변환은 파서 담당(1번) 몫이다. (data/README.md 계약 참고)
    """
    docs: List[Dict] = []
    for path in sorted(Path(processed_dir).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            docs.append(json.loads(line))
    return docs


def reindex(embedder: BGEM3Embedder | None = None) -> int:
    """두 store 를 drop 후 data/processed 로 재구축한다(저장소 불일치 복구용).

    삭제 전에 문서를 먼저 로드/검증한다 → 빈 processed 로 기존 색인이 날아가지 않게.
    """
    docs = load_documents()
    if not docs:
        raise SystemExit("data/processed 가 비어 있어 재구축할 문서가 없습니다.")
    MilvusRetriever().drop()
    OpenSearchRetriever().drop()
    return ingest(docs, embedder=embedder)


SAMPLE_DOCS: List[Dict] = [
    {
        "doc_id": "policy-001",
        "text": "본 계약의 지체상금은 지연 1일당 계약금액의 0.1%로 한다. "
        "다만 천재지변 등 불가항력 사유가 인정되는 경우 지체상금을 면제할 수 있다. "
        "하자보수 보증기간은 준공일로부터 2년으로 한다.",
        "source": "sample",
    }
]


if __name__ == "__main__":
    docs = load_documents()
    if not docs:
        # 실문서가 없을 때 조용히 샘플을 색인하면 '색인 성공' 착시가 생긴다.
        # 샘플 적재는 명시적 플래그로만 허용하고, 기본은 실패시킨다.
        if os.getenv("INGEST_ALLOW_SAMPLE") == "1":
            print("[경고] data/processed 가 비어 샘플 문서로 적재합니다(INGEST_ALLOW_SAMPLE=1).")
            docs = SAMPLE_DOCS
        else:
            print(
                "data/processed 에 문서가 없습니다. 파서로 채우거나, "
                "개발용이면 INGEST_ALLOW_SAMPLE=1 로 실행하세요(make ingest-sample).",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        print(f"data/processed 에서 문서 {len(docs)}건 로드")

    n = ingest(docs)
    print(f"ingested chunks: {n}")
