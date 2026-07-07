"""BGE-M3 dense 임베딩 클라이언트 (TEI 컨테이너 호출).

- 모델을 파이썬 프로세스에 직접 로드하지 않고, 별도 TEI(Text Embeddings Inference)
  컨테이너의 /embed 엔드포인트를 HTTP 로 호출한다.
  → 앱 재배포 시 모델 재로딩 없음, GPU 는 모델 서버로 격리, 앱은 가벼워진다.
- 공개 인터페이스(embed_texts / embed_query)와 차원(1024) 검증은 그대로 유지한다.
  그래서 ingestion.py / hybrid_search.py 는 수정 없이 동작한다.
"""
from __future__ import annotations

from typing import List, Sequence

import httpx

from app.config import settings


class BGEM3Embedder:
    # BGE-M3 base model 의 hidden size(=dense dim)는 1024 고정이다.
    EXPECTED_DIM = 1024

    def __init__(
        self,
        api_url: str | None = None,
        batch_size: int = 32,
        timeout: float = 60.0,
    ) -> None:
        self.dim = settings.embedding_dim
        # config 오타(예: 768)를 조기 진단한다(완전한 가드가 아니라 조기 진단용).
        if self.dim != self.EXPECTED_DIM:
            raise ValueError(
                f"BGE-M3 dense dimension 은 {self.EXPECTED_DIM} 여야 합니다. "
                f"settings.embedding_dim={self.dim}"
            )
        self.api_url = (api_url or settings.embedding_api_url).rstrip("/")
        self.batch_size = batch_size
        self.client = httpx.Client(timeout=timeout)

    def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        resp = self.client.post(
            f"{self.api_url}/embed",
            json={"inputs": batch, "truncate": True},  # truncate: 긴 입력은 서버에서 자름
        )
        resp.raise_for_status()
        vecs: List[List[float]] = resp.json()
        for v in vecs:
            if len(v) != self.dim:
                raise ValueError(
                    f"임베딩 차원 불일치: expected {self.dim}, got {len(v)}"
                )
        return vecs

    def embed_texts(
        self, texts: Sequence[str], batch_size: int | None = None
    ) -> List[List[float]]:
        if not texts:
            return []

        cleaned = [t.strip() for t in texts]
        if any(not t for t in cleaned):
            raise ValueError("Embedding 입력에 빈 문자열이 있습니다.")

        bs = batch_size or self.batch_size
        out: List[List[float]] = []
        for i in range(0, len(cleaned), bs):
            out.extend(self._embed_batch(cleaned[i : i + bs]))
        return out

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text], batch_size=1)[0]
