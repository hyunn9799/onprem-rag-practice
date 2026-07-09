"""bge-reranker-v2-m3 재정렬 클라이언트 (TEI 컨테이너 호출).

RRF 로 합친 후보를 TEI 의 /rerank 엔드포인트로 (query, passage) 쌍 채점해 정밀도를 올린다.
모델은 파이썬이 아니라 별도 TEI 컨테이너가 로드한다.
후보 dict 를 in-place 로 갱신해 기존 필드(page 등)를 보존한다.
"""
from __future__ import annotations

from typing import Dict, List

import httpx

from app.config import settings


class BGEReranker:
    def __init__(self, api_url: str | None = None, timeout: float = 60.0) -> None:
        self.api_url = (api_url or settings.reranker_api_url).rstrip("/")
        self.client = httpx.Client(timeout=timeout)

    def rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        if not candidates:
            return []

        resp = self.client.post(
            f"{self.api_url}/rerank",
            json={
                "query": query,
                "texts": [ c["text"] for c in candidates],
                "truncate": True,
            },
        )
        resp.raise_for_status()
        # TEI 응답: [{"index": i, "score": s}, ...] (score 내림차순 정렬)
        ranked = resp.json()

        out: List[Dict] = []
        for item in ranked[:top_k]:
            cand = candidates[item["index"]]
            cand["rerank_score"] = float(item["score"])
            out.append(cand)
        return out
