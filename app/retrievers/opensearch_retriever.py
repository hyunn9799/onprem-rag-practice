"""OpenSearch BM25 키워드 검색기.

핵심 실무 포인트:
- 모든 store 를 동일한 chunk_id 로 색인해야 나중에 RRF 병합이 가능하다.
- 한국어는 standard analyzer 로는 토큰화가 약하다. analysis-nori 플러그인을 깔고
  OPENSEARCH_ANALYZER=nori 로 두면 형태소 기반 BM25 품질이 크게 올라간다.
- 기존 index 의 analyzer 가 설정과 다르면 경고한다(standard↔nori 착시 방지).
  RECREATE_INDEX=1 이면 재생성(재적재 필요).
"""
from __future__ import annotations

from typing import Dict, List

from opensearchpy import OpenSearch, helpers

from app.schema import META_FIELDS as _META_KEYS  # 단일 소스 — 로컬 복제 금지
from app.config import settings


class OpenSearchRetriever:
    def __init__(self) -> None:
        self.index = settings.opensearch_index
        self.client = OpenSearch(
            hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
            http_compress=True,
            use_ssl=False,
            verify_certs=False,
        )

    def _create_index(self) -> None:
        analyzer = settings.opensearch_analyzer
        body = {
            "settings": {"index": {"number_of_shards": 1, "number_of_replicas": 0}},
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "text": {"type": "text", "analyzer": analyzer},
                    "page": {"type": "integer"},
                    "source": {"type": "keyword"},
                }
            },
        }
        self.client.indices.create(index=self.index, body=body)

    def _check_analyzer(self) -> None:
        """기존 index 의 text analyzer 가 설정과 일치하는지 확인.

        검사를 못 하면 조용히 넘기지 않고 경고를 남긴다 — 이 검사가 스킵되면
        stale analyzer 색인으로 실험/서빙이 착시가 되기 때문.
        """
        try:
            mapping = self.client.indices.get_mapping(index=self.index)
        except Exception as e:  # 연결 오류 등 — 검사 불가를 명시적으로 알린다
            print(f"[경고] analyzer 검사를 수행하지 못했습니다(건너뜀): {e}")
            return
        current = (
            mapping.get(self.index, {})
            .get("mappings", {})
            .get("properties", {})
            .get("text", {})
            or {}
        ).get("analyzer", "standard")
        want = settings.opensearch_analyzer
        if current == want:
            return
        if settings.recreate_index:  # .env 의 RECREATE_INDEX=true 로 켠다
            print(f"[재생성] OpenSearch index analyzer '{current}' -> '{want}'")
            self.client.indices.delete(index=self.index)
            self._create_index()
        else:
            print(
                f"[경고] 기존 index analyzer='{current}' != 설정 '{want}'. "
                "기존 index 는 그대로라 검색이 착시가 됩니다. "
                ".env 에 RECREATE_INDEX=true 를 넣고 재적재하거나 make rebuild-bm25 하세요."
            )

    def ensure_index(self) -> None:
        if self.client.indices.exists(self.index):
            self._check_analyzer()
            return
        self._create_index()

    def drop(self) -> None:
        if self.client.indices.exists(self.index):
            self.client.indices.delete(index=self.index)

    def delete_by_doc_id(self, doc_ids: List[str]) -> None:
        """해당 문서들의 기존 청크 전부 삭제(재적재 시 stale 청크 방지)."""
        if not doc_ids or not self.client.indices.exists(self.index):
            return
        self.client.delete_by_query(
            index=self.index,
            body={"query": {"terms": {"doc_id": list(doc_ids)}}},
            refresh=True,
        )

    def index_chunks(self, chunks: List[Dict]) -> None:
        actions = []
        for c in chunks:
            source = {"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "text": c["text"]}
            source.update({k: c[k] for k in _META_KEYS if k in c})
            actions.append(
                {
                    "_index": self.index,
                    "_id": c["chunk_id"],  # chunk_id 를 문서 id 로 → 재적재 시 자동 upsert
                    "_source": source,
                }
            )
        helpers.bulk(self.client, actions, refresh=True)

    def search(self, query: str, top_k: int) -> List[Dict]:
        res = self.client.search(
            index=self.index,
            body={"size": top_k, "query": {"match": {"text": query}}},
        )
        results: List[Dict] = []
        for h in res["hits"]["hits"]:
            src = h["_source"]
            item = {
                "chunk_id": src["chunk_id"],
                "doc_id": src.get("doc_id"),
                "text": src["text"],
                "score": h["_score"],
            }
            item.update({k: src[k] for k in _META_KEYS if k in src})
            results.append(item)
        return results
