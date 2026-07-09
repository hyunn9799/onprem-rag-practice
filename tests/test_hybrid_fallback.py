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


class _BoomLLM:
    def generate(self, *a, **k):
        raise RuntimeError("llm down")


def test_retrieve_respects_top_k_param():
    # /search 의 top_k 가 최종 반환 개수를 지배한다(리랭커 off 경로).
    many = [{"chunk_id": str(i), "text": "t"} for i in range(10)]
    rag = _rag(True, _OK(many), _OK([]))
    hits, _ = rag.retrieve("q", top_k=3)
    assert len(hits) == 3


def test_llm_unconfigured_none_still_returns_sources():
    # LLM 미구성(None)이어도 /ask 는 근거+경고 반환 — /search 서비스 부팅과 무관해야 함.
    rag = _rag(True, _OK([{"chunk_id": "v", "text": "본문"}]), _OK([]))
    rag.llm = None
    out = rag.answer("q")
    assert out["sources"] and any("llm" in w for w in out["warnings"])


def test_llm_down_returns_sources_with_warning():
    # LLM 만 죽어도 /ask 는 500 이 아니라 근거(sources)+경고를 반환한다.
    rag = _rag(True, _OK([{"chunk_id": "v", "text": "본문"}]), _OK([{"chunk_id": "b", "text": "본문"}]))
    rag.llm = _BoomLLM()
    out = rag.answer("q")
    assert out["sources"]                       # 검색 결과는 살아서 전달됨
    assert any("llm" in w for w in out["warnings"])
    assert "LLM" in out["answer"]               # 장애 안내 문구
