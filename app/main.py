"""FastAPI 서비스. 모델은 lifespan 에서 1회 로드(HTTP 클라이언트 생성) 후 재사용.

실행:  uvicorn app.main:app --host 0.0.0.0 --port 8000

- /health : 앱 생존(liveness). 의존성과 무관하게 프로세스가 살아있는지.
- /ready  : 의존성 준비(readiness). OpenSearch/Milvus/TEI 연결을 실제로 확인.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Dict, List

import httpx
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from app.hybrid_search import HybridRAG
from app.ingestion import ingest

STATE: Dict[str, HybridRAG] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["rag"] = HybridRAG()   # 서버 부팅 시 임베더/리랭커/LLM 클라이언트 생성
    yield
    STATE.clear()


app = FastAPI(title="On-prem Hybrid RAG", lifespan=lifespan)


class Doc(BaseModel):
    """pre-chunked 청크 레코드 (팀 계약: chunk_id 는 파서가 발급)."""
    chunk_id: str
    doc_id: str
    text: str
    page: int | None = None
    source: str | None = None


class IngestRequest(BaseModel):
    documents: List[Doc]


class AskRequest(BaseModel):
    question: str


class SearchRequest(BaseModel):
    """3번(챗봇) → 2번(검색) 계약: POST /search {query, top_k}. 합의 기본값 top_k=5."""
    query: str
    top_k: int = Field(5, ge=1, le=50)


def _to_search_result(hit: Dict) -> Dict:
    """retrieve() 결과 → /search 응답 계약 형태.

    score: 리랭커 on 이면 rerank_score(0~1 정규화), off 폴백이면 rrf_score.
           (스케일이 다르므로 3번은 threshold 를 rerank 기준으로만 걸 것 — 계약 문서화 필요)
    """
    return {
        "chunk_id": hit["chunk_id"],
        "text": hit["text"],
        "score": hit.get("rerank_score", hit.get("rrf_score", 0.0)),
        "metadata": {
            "doc_id": hit.get("doc_id"),
            "source": hit.get("source"),
            "page": hit.get("page"),
        },
    }


@app.post("/ingest")
def ingest_endpoint(req: IngestRequest):
    # exclude_none: page/source 미지정(None)이 메타로 섞이지 않게.
    docs = [d.model_dump(exclude_none=True) for d in req.documents]
    try:
        n = ingest(docs, embedder=STATE["rag"].embedder)
    except ValueError as e:  # 계약 위반(chunk_id 누락 등)은 500 이 아니라 422 로
        raise HTTPException(status_code=422, detail=str(e))
    return {"ingested_chunks": n}


@app.post("/search")
def search_endpoint(req: SearchRequest):
    """검색만 수행(LLM 없음) — 3번 챗봇이 소비하는 팀 계약 엔드포인트.

    warnings 는 계약 외 추가 필드(부분 장애 알림용) — JSON 소비자는 모르는 필드를
    무시하므로 하위호환 안전. 3번에게 공유해 두면 좋다.
    """
    hits, warnings = STATE["rag"].retrieve(req.query, top_k=req.top_k)
    results = [_to_search_result(h) for h in hits]
    return {"query": req.query, "count": len(results), "results": results, "warnings": warnings}


@app.post("/ask")
def ask_endpoint(req: AskRequest):
    return STATE["rag"].answer(req.question)


@app.get("/health")
def health():
    """Liveness: 프로세스 생존만 본다."""
    return {"status": "ok"}


def _check(fn) -> bool:
    try:
        return bool(fn())
    except Exception:
        return False


@app.get("/ready")
def ready(response: Response):
    """Readiness: 실제 의존성(OpenSearch/Milvus/TEI) 연결을 확인한다.

    not-ready 면 HTTP 503 (probe/LB 는 status code 로 판단).
    4개 체크는 서로 독립이라 병렬 실행 — 순차면 장애 시 타임아웃이 합산되어
    (4×3s+) probe 자체가 타임아웃으로 죽는다.
    """
    rag = STATE.get("rag")
    if rag is None:
        response.status_code = 503
        return {"ready": False, "checks": {}}

    probes = {
        "opensearch": lambda: rag.opensearch.client.ping(),
        "milvus": lambda: rag.milvus.client.list_collections() is not None,
        "embedding": lambda: httpx.get(
            f"{rag.embedder.api_url}/health", timeout=3.0
        ).status_code == 200,
    }
    if rag.reranker is not None:
        probes["reranker"] = lambda: httpx.get(
            f"{rag.reranker.api_url}/health", timeout=3.0
        ).status_code == 200

    with ThreadPoolExecutor(max_workers=len(probes)) as pool:
        futures = {name: pool.submit(_check, fn) for name, fn in probes.items()}
        checks: Dict[str, bool] = {name: f.result() for name, f in futures.items()}

    ok = all(checks.values())
    if not ok:
        response.status_code = 503
    return {"ready": ok, "checks": checks}
