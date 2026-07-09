"""문서 적재 파이프라인: pre-chunked 청크 -> 임베딩 -> Milvus + OpenSearch 동시 저장.

팀 계약: 청킹/chunk_id 발급은 파서(1번, app/parse_odl.py) 소관 — 여기는
발급된 청크를 검증하고 저장만 한다. chunk_id 없는 레코드는 즉시 실패.

실무 포인트:
- chunk_id 가 결정론적이라 재적재 시 upsert 로 수렴한다.
- 문서 단위 replace: 같은 doc_id 의 기존 청크를 지우고 새로 넣는다
  (문서가 짧아져도 stale 청크가 남지 않는다).
- 임베딩을 '먼저' 수행하고 삭제/저장은 그 다음 — 임베딩 실패가 데이터 손실로
  이어지지 않게 한다(부작용 없는 단계 먼저).
- 출처 메타데이터(META_FIELDS: page/source)는 두 store 로 함께 전파한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

from app.config import settings
from app.schema import META_FIELDS
from app.embeddings.bge_m3_embedder import BGEM3Embedder
from app.retrievers.milvus_retriever import MilvusRetriever
from app.retrievers.opensearch_retriever import OpenSearchRetriever


def _build_chunks(documents: List[Dict]) -> List[Dict]:
    """pre-chunked 전용: 청킹/chunk_id 발급은 파서(1번, parse_odl.py) 소관.

    chunk_id 없는 레코드는 조용히 잘못 저장되는 대신 즉시 실패시킨다 —
    계약 위반이 데이터 오염이 아니라 명확한 에러로 드러나게.
    """
    all_chunks: List[Dict] = []
    for d in documents:
        if not d.get("chunk_id"):
            raise ValueError(
                f"chunk_id 없는 레코드: doc_id={d.get('doc_id')!r} — "
                "팀 계약은 pre-chunked(파서가 chunk_id 발급) 만 받습니다."
            )
        text = (d.get("text") or "").strip()
        if not text:
            continue
        chunk = {"chunk_id": d["chunk_id"], "doc_id": d["doc_id"], "text": text}
        chunk.update({k: d[k] for k in META_FIELDS if k in d})
        all_chunks.append(chunk)
    return all_chunks


def _ingest_chunks(chunks: List[Dict], embedder: BGEM3Embedder) -> int:
    """검증 완료된 청크를 임베딩해 두 store 에 저장한다."""
    if not chunks:
        return 0
    milvus, opensearch = MilvusRetriever(), OpenSearchRetriever()
    milvus.ensure_collection()
    opensearch.ensure_index()

    # 부작용 없는 단계(임베딩)를 먼저 — 여기서 실패해도 기존 데이터는 무사하다.
    embeddings = embedder.embed_texts([c["text"] for c in chunks])

    # 문서 단위 replace: 기존 청크 제거 후 저장(축소 재적재 시 고아 방지).
    doc_ids = sorted({c["doc_id"] for c in chunks})
    milvus.delete_by_doc_id(doc_ids)
    milvus.upsert(chunks, embeddings)      # 벡터 검색용
    opensearch.delete_by_doc_id(doc_ids)
    opensearch.index_chunks(chunks)        # BM25 검색용
    return len(chunks)


def ingest(documents: List[Dict], embedder: BGEM3Embedder | None = None) -> int:
    """documents: [{"chunk_id": str, "doc_id": str, "text": str, "page"?, "source"?}, ...]"""
    return _ingest_chunks(_build_chunks(documents), embedder or BGEM3Embedder())


def load_documents(processed_dir: str = "data/processed") -> List[Dict]:
    """data/processed 에서 파싱된 문서/청크를 읽는다.

    - *.jsonl : 한 줄 = 레코드 하나. '#' 줄은 주석으로 무시.
    - *.json  : 레코드 배열 (팀 파서의 hyosung_chunks.json 형식).
    모든 레코드는 pre-chunked(chunk_id 필수) — 검증은 _build_chunks 가 한다.
    """
    docs: List[Dict] = []
    for path in sorted(Path(processed_dir).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            docs.append(json.loads(line))
    for path in sorted(Path(processed_dir).glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            docs.extend(data)
    return docs


def reindex(embedder: BGEM3Embedder | None = None) -> int:
    """두 store 를 drop 후 data/processed 로 재구축한다(저장소 불일치 복구용).

    ⚠️ data/processed 만 복원한다 — API(POST /ingest)로만 적재한 문서는 복원되지 않는다.
    drop 은 되돌릴 수 없으므로, 실패할 수 있는 전제(청크 계약 검증, 임베딩 서버 연결)를
    전부 drop '전에' 수행한다.
    """
    # 계약 위반(chunk_id 누락 등)은 여기서(=drop 전에) ValueError 로 터진다.
    chunks = _build_chunks(load_documents())
    if not chunks:
        raise SystemExit("data/processed 가 비어 있어 재구축할 문서가 없습니다.")

    embedder = embedder or BGEM3Embedder()
    try:
        embedder.embed_query("연결 확인")  # drop 전 임베딩 서버 생존 확인
    except Exception as e:
        raise SystemExit(
            f"임베딩 서버에 연결할 수 없어 중단합니다(기존 색인은 보존됨): {e}\n"
            "  .env 의 EMBEDDING_API_URL 을 확인하세요(runbook FAQ: Lightning URL 변경)."
        )

    print("[주의] reindex 는 data/processed 만 복원합니다 — API 로만 적재한 문서는 사라집니다.")
    MilvusRetriever().drop()
    OpenSearchRetriever().drop()
    return _ingest_chunks(chunks, embedder)


def rebuild_bm25() -> int:
    """OpenSearch(BM25)만 재구축 — analyzer 변경(nori 등) 적용용.

    임베딩이 필요 없어 임베딩 서버 없이도 실행 가능하다. Milvus 는 건드리지 않으며
    chunk_id 는 결정론적이라 두 store 의 키는 계속 일치한다.
    """
    chunks = _build_chunks(load_documents())
    if not chunks:
        raise SystemExit("data/processed 가 비어 있어 재구축할 문서가 없습니다.")

    opensearch = OpenSearchRetriever()
    opensearch.drop()
    opensearch.ensure_index()          # settings.opensearch_analyzer 로 생성
    opensearch.index_chunks(chunks)
    return len(chunks)


SAMPLE_DOCS: List[Dict] = [
    {
        "chunk_id": "policy-001_p001",  # pre-chunked 계약에 맞춰 샘플도 id 를 갖는다
        "doc_id": "policy-001",
        "page": 1,
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
        # 샘플 적재는 명시적 설정으로만 허용하고, 기본은 실패시킨다.
        if settings.ingest_allow_sample:
            print("[경고] data/processed 가 비어 샘플 문서로 적재합니다(INGEST_ALLOW_SAMPLE).")
            docs = SAMPLE_DOCS
        else:
            print(
                "data/processed 에 청크가 없습니다. 팀 파서 산출물(hyosung_chunks.json)을 "
                "넣거나, 개발용이면 INGEST_ALLOW_SAMPLE=1 로 실행하세요(make ingest-sample).",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        print(f"data/processed 에서 문서 {len(docs)}건 로드")

    n = ingest(docs)
    print(f"ingested chunks: {n}")
