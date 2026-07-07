"""#5 graceful degradation 단위 테스트.

무거운 의존성이 설치돼 있어야 실행된다(없으면 skip). 실제 서버 없이,
__new__ 로 HybridRAG 를 만들고 검색기를 목으로 바꿔 retrieve() 의 폴백을 검증한다.
"""
import pytest

pytest.importorskip("pydantic_settings")
pytest.importorskip("openai")
pytest.importorskip("pymilvus")
pytest.importorskip("opensearchpy")
pytest.importorskip("httpx")

from app.hybrid_search import HybridRAG  # noqa: E402


class _OK:
    def __init__(self, hits):
        self.hits = hits

    def search(self, *a, **k):
        return self.hits


class _Boom:
    def search(self, *a, **k):
        raise RuntimeError("down")


class _BoomReranker:
    def rerank(self, *a, **k):
        raise RuntimeError("rerank down")


class _Embedder:
    def __init__(self, ok):
        self.ok = ok

    def embed_query(self, q):
        if not self.ok:
            raise RuntimeError("tei down")
        return [0.0]


def _rag(embed_ok, milvus, opensearch):
    rag = HybridRAG.__new__(HybridRAG)  # __init__ 우회(실 클라이언트 생성 안 함)
    rag.embedder = _Embedder(embed_ok)
    rag.milvus = milvus
    rag.opensearch = opensearch
    rag.reranker = None
    return rag


def test_vector_down_falls_back_to_bm25():
    rag = _rag(True, _Boom(), _OK([{"chunk_id": "b", "text": "t"}]))
    hits, warns = rag.retrieve("q")
    assert [h["chunk_id"] for h in hits] == ["b"]
    assert any("vector" in w for w in warns)


def test_bm25_down_falls_back_to_vector():
    rag = _rag(True, _OK([{"chunk_id": "v", "text": "t"}]), _Boom())
    hits, warns = rag.retrieve("q")
    assert [h["chunk_id"] for h in hits] == ["v"]
    assert any("bm25" in w for w in warns)


def test_embedding_down_still_returns_bm25():
    rag = _rag(False, _OK([{"chunk_id": "v", "text": "t"}]), _OK([{"chunk_id": "b", "text": "t"}]))
    hits, warns = rag.retrieve("q")
    assert "b" in {h["chunk_id"] for h in hits}  # bm25 살아있음
    assert any("vector" in w for w in warns)


def test_reranker_down_falls_back_to_fused():
    rag = _rag(True, _OK([{"chunk_id": "v", "text": "t"}]), _OK([{"chunk_id": "b", "text": "t"}]))
    rag.reranker = _BoomReranker()  # 리랭커만 죽음
    hits, warns = rag.retrieve("q")
    assert len(hits) >= 1  # 병합 결과로 폴백(전체 실패 아님)
    assert any("rerank" in w for w in warns)
