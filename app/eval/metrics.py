"""검색 평가 지표 (순위 기반, 외부 의존성/LLM 불필요).

모든 함수는 (ranked, relevant, k) 를 받는다.
- ranked   : 검색이 반환한 chunk_id 리스트 (점수 내림차순)
- relevant : 정답 chunk_id 집합(set)
이진 관련성(정답=1, 그 외=0)을 가정한다.
"""
from __future__ import annotations

import math
from typing import Sequence, Set


def hit_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    """top-k 안에 정답이 하나라도 있으면 1, 없으면 0."""
    return 1.0 if any(cid in relevant for cid in ranked[:k]) else 0.0


def recall_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    """top-k 가 정답 집합의 몇 %를 회수했나."""
    if not relevant:
        return 0.0
    found = sum(1 for cid in ranked[:k] if cid in relevant)
    return found / len(relevant)


def precision_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    """top-k 중 정답 비율."""
    if k <= 0:
        return 0.0
    found = sum(1 for cid in ranked[:k] if cid in relevant)
    return found / k


def mrr_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    """첫 정답의 역순위(1/rank). top-k 안에 없으면 0."""
    for i, cid in enumerate(ranked[:k]):
        if cid in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Set[str], k: int) -> float:
    """정규화 할인 누적 이득. 정답이 상위에 있을수록 높다."""
    dcg = 0.0
    for i, cid in enumerate(ranked[:k]):
        if cid in relevant:
            dcg += 1.0 / math.log2(i + 2)  # rank=i+1 → log2(rank+1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0
