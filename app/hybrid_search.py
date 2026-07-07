"""하이브리드 RAG 파이프라인.

질문 -> (BM25 + 벡터) 검색 -> RRF 병합 -> 리랭커 정밀 재정렬 ->
상위 chunk 로 프롬프트 구성 -> LLM 답변 생성.

모델(임베딩/리랭커/LLM)은 별도 컨테이너(TEI/vLLM)로 서빙하고, 여기의 embedder/
reranker/llm 은 HTTP 클라이언트다. 그래서 이 객체 생성은 가볍고, 싱글턴으로 한 번만
만들어 재사용한다(main.py lifespan).

검색은 graceful degradation 한다: 한쪽 검색기가 죽어도 나머지로 부분 결과를 내고,
어떤 단계가 실패했는지 warnings 로 함께 반환한다(부분 결과가 정상인 척하지 않게).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from app.config import settings
from app.embeddings.bge_m3_embedder import BGEM3Embedder
from app.llm.ax_light import AXLightLLM
from app.rerank.bge_reranker import BGEReranker
from app.retrievers.fusion import reciprocal_rank_fusion
from app.retrievers.milvus_retriever import MilvusRetriever
from app.retrievers.opensearch_retriever import OpenSearchRetriever

_META_KEYS = ("page", "section", "source")


def _to_source(c: Dict) -> Dict:
    src = {"chunk_id": c["chunk_id"], "text": c["text"]}
    src.update({k: c[k] for k in _META_KEYS if k in c})
    return src


class HybridRAG:
    def __init__(self) -> None:
        self.embedder = BGEM3Embedder()
        self.milvus = MilvusRetriever()
        self.opensearch = OpenSearchRetriever()
        self.reranker = BGEReranker() if settings.use_reranker else None
        self.llm = AXLightLLM()

    def retrieve(self, question: str) -> Tuple[List[Dict], List[str]]:
        warnings: List[str] = []

        # 벡터 경로(임베딩 + Milvus). 실패해도 BM25 로 폴백.
        try:
            q_emb = self.embedder.embed_query(question)
            vector_hits = self.milvus.search(q_emb, settings.vector_top_k)
        except Exception as e:
            vector_hits = []
            warnings.append(f"vector 검색 실패: {e}")

        # 키워드 경로(OpenSearch BM25).
        try:
            bm25_hits = self.opensearch.search(question, settings.bm25_top_k)
        except Exception as e:
            bm25_hits = []
            warnings.append(f"bm25 검색 실패: {e}")

        fused = reciprocal_rank_fusion(
            [vector_hits, bm25_hits], k=settings.rrf_k, top_n=settings.final_top_k * 4
        )
        if self.reranker and fused:
            try:
                fused = self.reranker.rerank(question, fused, top_k=settings.final_top_k)
            except Exception as e:
                # 리랭커만 죽어도 병합 결과로 답한다(리랭크 없이 상위 N).
                warnings.append(f"rerank 실패(폴백): {e}")
                fused = fused[: settings.final_top_k]
        else:
            fused = fused[: settings.final_top_k]
        return fused, warnings

    def answer(self, question: str) -> Dict:
        contexts, warnings = self.retrieve(question)
        if not contexts:
            return {
                "question": question,
                "answer": "제공된 문서에서 확인되지 않습니다.",
                "sources": [],
                "warnings": warnings,
            }
        answer = self.llm.generate(question, [c["text"] for c in contexts])
        return {
            "question": question,
            "answer": answer,
            "sources": [_to_source(c) for c in contexts],
            "warnings": warnings,
        }


if __name__ == "__main__":
    rag = HybridRAG()
    result = rag.answer("지체상금은 하루에 얼마이고 면제되는 경우는?")
    print(result["answer"])
    if result["warnings"]:
        print("\n[warnings]", result["warnings"])
    print("\n--- sources ---")
    for s in result["sources"]:
        print(s["chunk_id"], s.get("page", ""), ":", s["text"][:60])
