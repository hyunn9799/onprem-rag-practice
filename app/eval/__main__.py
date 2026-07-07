"""검색 파이프라인 비교 평가: BM25 / vector / RRF / RRF+rerank 를 같은 정답셋으로 채점.

실행:  make eval   (= uv run python -m app.eval)

전제:
- 스택이 떠 있고(make up 또는 up-search) 문서가 적재돼 있어야 한다(make ingest).
- LLM 은 쓰지 않는다 → vLLM/OpenAI 없이도 동작(검색 지표만).

정답셋(qrels): 기본 data/eval/qrels.jsonl. 한 줄에 하나:
    {"question": "지체상금은 하루에 얼마인가?", "relevant": ["policy-001#0"]}
'#' 로 시작하는 줄은 주석으로 무시한다. EVAL_QRELS 로 경로 변경 가능.

핵심 아이디어: hybrid_search.retrieve() 는 최종 결과 하나만 주므로(단계 관측 불가),
여기서는 컴포넌트(milvus/opensearch/fusion/reranker)를 직접 호출해 단계별 순위를 만든다.
그래야 "리랭커가 nDCG 를 얼마나 올렸나", "하이브리드가 BM25 단독보다 나은가"를 잴 수 있다.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

from app.config import settings
from app.embeddings.bge_m3_embedder import BGEM3Embedder
from app.eval.metrics import hit_at_k, mrr_at_k, ndcg_at_k, recall_at_k
from app.rerank.bge_reranker import BGEReranker
from app.retrievers.fusion import reciprocal_rank_fusion
from app.retrievers.milvus_retriever import MilvusRetriever
from app.retrievers.opensearch_retriever import OpenSearchRetriever

QRELS_PATH = os.getenv("EVAL_QRELS", "data/eval/qrels.jsonl")
DEPTH = 10  # 각 검색기 조회 깊이(top-k)

# 리포트할 지표: (표기, 함수, k)
REPORT = [
    ("Hit@1", hit_at_k, 1),
    ("Hit@3", hit_at_k, 3),
    ("Hit@5", hit_at_k, 5),
    ("MRR@10", mrr_at_k, 10),
    ("nDCG@10", ndcg_at_k, 10),
    ("Recall@5", recall_at_k, 5),
]


def load_qrels(path: str) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"정답셋이 없습니다: {path}\n"
            "  data/eval/qrels.jsonl 형식으로 '질문 → 정답 chunk_id' 를 채워주세요."
        )
    rows: List[Dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"정답셋이 비었습니다: {path}")
    return rows


def rank_by_method(
    question: str,
    embedder: BGEM3Embedder,
    milvus: MilvusRetriever,
    opensearch: OpenSearchRetriever,
    reranker: BGEReranker | None,
) -> Dict[str, List[str]]:
    """한 질문에 대해 method -> 순위(chunk_id 리스트) 를 만든다."""
    q_emb = embedder.embed_query(question)
    vector_hits = milvus.search(q_emb, DEPTH)
    bm25_hits = opensearch.search(question, DEPTH)
    rrf_hits = reciprocal_rank_fusion(
        [vector_hits, bm25_hits], k=settings.rrf_k, top_n=DEPTH
    )

    ranked: Dict[str, List[str]] = {
        "bm25": [h["chunk_id"] for h in bm25_hits],
        "vector": [h["chunk_id"] for h in vector_hits],
        "rrf": [h["chunk_id"] for h in rrf_hits],
    }
    if reranker is not None:
        rr = reranker.rerank(question, rrf_hits, top_k=DEPTH)
        ranked["rrf+rerank"] = [h["chunk_id"] for h in rr]
    return ranked


def print_table(methods: List[str], sums: Dict[str, Dict[str, float]], n: int) -> None:
    labels = [label for (label, _, _) in REPORT]
    print(f"\n정답셋: {n} 질문 ({QRELS_PATH})  |  조회 깊이 DEPTH={DEPTH}\n")
    header = f"{'method':<12}" + "".join(f"{lab:>9}" for lab in labels)
    print(header)
    print("-" * len(header))
    for m in methods:
        cells = "".join(f"{sums[m][lab] / n:>9.3f}" for lab in labels)
        print(f"{m:<12}{cells}")
    print()


def run() -> None:
    qrels = load_qrels(QRELS_PATH)
    n = len(qrels)

    embedder = BGEM3Embedder()
    milvus = MilvusRetriever()
    opensearch = OpenSearchRetriever()
    reranker = BGEReranker() if settings.use_reranker else None

    methods = ["bm25", "vector", "rrf"]
    if reranker is not None:
        methods.append("rrf+rerank")

    sums: Dict[str, Dict[str, float]] = {
        m: {label: 0.0 for (label, _, _) in REPORT} for m in methods
    }

    for row in qrels:
        relevant = set(row["relevant"])
        ranked = rank_by_method(row["question"], embedder, milvus, opensearch, reranker)
        for m in methods:
            for (label, fn, k) in REPORT:
                sums[m][label] += fn(ranked[m], relevant, k)

    print_table(methods, sums, n)


if __name__ == "__main__":
    run()
