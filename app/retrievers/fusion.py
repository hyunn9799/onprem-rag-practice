"""Reciprocal Rank Fusion (RRF).

BM25 점수와 벡터 유사도는 스케일이 완전히 달라서 그냥 더하면 안 된다.
RRF 는 '점수'가 아니라 '순위'만 사용하므로 이질적인 검색기를 안전하게 합친다.
    score(d) = Σ 1 / (k + rank_i(d))
k 는 보통 60. 낮은 순위의 영향력을 부드럽게 낮춰준다.

병합 시 각 항목의 메타데이터(page/source/text 등)를 그대로 보존한다.
"""
from __future__ import annotations

from typing import Dict, List


def reciprocal_rank_fusion(result_lists: List[List[Dict]], k: int, top_n: int) -> List[Dict]:
    scores: Dict[str, float] = {}
    items: Dict[str, Dict] = {}
    for results in result_lists:
        for rank, item in enumerate(results):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            if cid not in items:
                items[cid] = item  # 첫 등장 항목 보존(메타데이터 유지)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    out: List[Dict] = []
    for cid, s in ranked[:top_n]:
        merged = dict(items[cid])
        # 검색기별 'score' 는 스케일이 달라(BM25 vs 코사인) 병합 후엔 무의미 — 남기면
        # 하류에서 섞어 쓸 위험만 있다. 병합 결과의 점수는 rrf_score 하나다.
        merged.pop("score", None)
        merged["rrf_score"] = s
        out.append(merged)
    return out
