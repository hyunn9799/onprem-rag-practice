"""검색 파이프라인 비교 평가: BM25 / vector / RRF / RRF+rerank 를 같은 정답셋으로 채점.

실행:  make eval   (= uv run python -m app.eval)

전제:
- 스택이 떠 있고(make up 또는 up-search) 문서가 적재돼 있어야 한다(make ingest).
- LLM 은 쓰지 않는다 → vLLM/OpenAI 없이도 동작(검색 지표만).

정답셋(qrels): 기본 data/eval/qrels.jsonl (.env 의 EVAL_QRELS 로 변경). 한 줄에 하나:
    {"question": "지체상금은 하루에 얼마인가?", "relevant": ["policy-001#p1"]}
'#' 로 시작하는 줄은 주석으로 무시한다.

핵심 원칙: **프로덕션(hybrid_search.retrieve)과 동일한 깊이로 채점한다.**
검색 깊이/RRF top_n/rerank top_k 를 전부 settings 에서 파생하므로, .env 의
retrieval knob 을 바꾸면 eval 숫자도 그 구성 기준으로 바뀐다 — eval 이
프로덕션이 절대 실행하지 않는 파이프라인을 채점하는 착시를 막는다.
(단계별 결과를 따로 보기 위해 컴포넌트를 직접 호출하는 구조는 유지.)
"""
from __future__ import annotations

import json

from pathlib import Path
from typing import Dict, List

from app.config import settings
from app.embeddings.bge_m3_embedder import BGEM3Embedder
from app.eval.metrics import hit_at_k, mrr_at_k, ndcg_at_k, recall_at_k
from app.rerank.bge_reranker import BGEReranker
from app.retrievers.fusion import reciprocal_rank_fusion
from app.retrievers.milvus_retriever import MilvusRetriever
from app.retrievers.opensearch_retriever import OpenSearchRetriever

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
    q_emb: List[float],
    milvus: MilvusRetriever,
    opensearch: OpenSearchRetriever,
    reranker: BGEReranker | None,
) -> Dict[str, List[str]]:
    """한 질문에 대해 method -> 순위(chunk_id 리스트). 깊이는 프로덕션과 동일."""
    vector_hits = milvus.search(q_emb, settings.vector_top_k)
    bm25_hits = opensearch.search(question, settings.bm25_top_k)
    rrf_hits = reciprocal_rank_fusion(
        [vector_hits, bm25_hits], k=settings.rrf_k, top_n=settings.final_top_k * 4
    )

    ranked: Dict[str, List[str]] = {
        "bm25": [h["chunk_id"] for h in bm25_hits],
        "vector": [h["chunk_id"] for h in vector_hits],
        "rrf": [h["chunk_id"] for h in rrf_hits],
    }
    if reranker is not None:
        rr = reranker.rerank(question, list(rrf_hits), top_k=settings.final_top_k)
        ranked["rrf+rerank"] = [h["chunk_id"] for h in rr]
    return ranked


def print_table(methods: List[str], sums: Dict[str, Dict[str, float]], n: int) -> None:
    labels = [label for (label, _, _) in REPORT]
    print(
        f"\n정답셋: {n} 질문 ({settings.eval_qrels})  |  "
        f"깊이: vector={settings.vector_top_k} bm25={settings.bm25_top_k} "
        f"rrf_top_n={settings.final_top_k * 4} rerank_top_k={settings.final_top_k}\n"
    )
    header = f"{'method':<12}" + "".join(f"{lab:>9}" for lab in labels)
    print(header)
    print("-" * len(header))
    for m in methods:
        cells = "".join(f"{sums[m][lab] / n:>9.3f}" for lab in labels)
        print(f"{m:<12}{cells}")
    print()


def run() -> None:
    qrels = load_qrels(settings.eval_qrels)
    n = len(qrels)

    embedder = BGEM3Embedder()
    milvus = MilvusRetriever()
    opensearch = OpenSearchRetriever()
    reranker = BGEReranker() if settings.use_reranker else None

    # 질문 임베딩은 한 번에 배치로 — 질문당 원격 왕복(N회)을 1회로 줄인다.
    q_embs = embedder.embed_texts([row["question"] for row in qrels])

    methods = ["bm25", "vector", "rrf"]
    if reranker is not None:
        methods.append("rrf+rerank")

    sums: Dict[str, Dict[str, float]] = {
        m: {label: 0.0 for (label, _, _) in REPORT} for m in methods
    }

    for row, q_emb in zip(qrels, q_embs):
        relevant = set(row["relevant"])
        ranked = rank_by_method(row["question"], q_emb, milvus, opensearch, reranker)
        for m in methods:
            for (label, fn, k) in REPORT:
                sums[m][label] += fn(ranked[m], relevant, k)

    print_table(methods, sums, n)


if __name__ == "__main__":
    run()
