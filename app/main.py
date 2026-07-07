"""FastAPI 서비스. 모델은 lifespan 에서 1회 로드(HTTP 클라이언트 생성) 후 재사용.

실행:  uvicorn app.main:app --host 0.0.0.0 --port 8000

- /health : 앱 생존(liveness). 의존성과 무관하게 프로세스가 살아있는지.
- /ready  : 의존성 준비(readiness). OpenSearch/Milvus/TEI 연결을 실제로 확인.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Dict, List

import httpx
from fastapi import FastAPI, Response
from pydantic import BaseModel

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
    doc_id: str
    text: str
    page: int | None = None
    section: str | None = None
    source: str | None = None


class IngestRequest(BaseModel):
    documents: List[Doc]


class AskRequest(BaseModel):
    question: str


@app.post("/ingest")
def ingest_endpoint(req: IngestRequest):
    # exclude_none: page/section/source 미지정(None)이 메타로 섞이지 않게.
    docs = [d.model_dump(exclude_none=True) for d in req.documents]
    n = ingest(docs, embedder=STATE["rag"].embedder)
    return {"ingested_chunks": n}


@app.post("/ask")
def ask_endpoint(req: AskRequest):
    return STATE["rag"].answer(req.question)


@app.get("/health")
def health():
    """Liveness: 프로세스 생존만 본다."""
    return {"status": "ok"}


@app.get("/ready")
def ready(response: Response):
    """Readiness: 실제 의존성(OpenSearch/Milvus/TEI) 연결을 확인한다.

    not-ready 면 HTTP 503 을 반환한다(probe/LB 는 status code 로 판단하므로).
    """
    rag = STATE.get("rag")
    if rag is None:
        response.status_code = 503
        return {"ready": False, "checks": {}}

    checks: Dict[str, bool] = {}
    try:
        checks["opensearch"] = bool(rag.opensearch.client.ping())
    except Exception:
        checks["opensearch"] = False
    try:
        rag.milvus.client.list_collections()
        checks["milvus"] = True
    except Exception:
        checks["milvus"] = False
    try:
        checks["embedding"] = (
            httpx.get(f"{rag.embedder.api_url}/health", timeout=3.0).status_code == 200
        )
    except Exception:
        checks["embedding"] = False
    if rag.reranker is not None:
        try:
            checks["reranker"] = (
                httpx.get(f"{rag.reranker.api_url}/health", timeout=3.0).status_code == 200
            )
        except Exception:
            checks["reranker"] = False

    ok = all(checks.values())
    if not ok:
        response.status_code = 503
    return {"ready": ok, "checks": checks}
