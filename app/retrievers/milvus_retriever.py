"""Milvus dense 벡터 검색기 (pymilvus 2.6 MilvusClient API).

- dim 은 BGE-M3 에 맞춰 1024.
- BGE-M3 dense 는 코사인 유사도가 표준 → metric_type=COSINE, index=HNSW.
- chunk_id 를 PK 로 써서 OpenSearch 와 동일 키로 맞춘다(RRF 병합 조건).
- 출처 메타데이터(META_FIELDS: page/source)는 meta(VARCHAR, JSON 문자열)로 저장·반환한다.
  → 스키마 변경이므로 기존 컬렉션은 make reindex 로 재구축해야 반영된다.
"""
from __future__ import annotations

import json
from typing import Dict, List

from pymilvus import DataType, MilvusClient

from app.schema import META_FIELDS as _META_KEYS  # 단일 소스 — 로컬 복제 금지
from app.config import settings


def _dump_meta(chunk: Dict) -> str:
    return json.dumps({k: chunk[k] for k in _META_KEYS if k in chunk}, ensure_ascii=False)


def _load_meta(raw: str | None) -> Dict:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


class MilvusRetriever:
    def __init__(self) -> None:
        self.collection = settings.milvus_collection
        self.client = MilvusClient(uri=settings.milvus_uri)

    def ensure_collection(self) -> None:
        if self.client.has_collection(self.collection):
            return
        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True, max_length=128)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=128)
        # max_length 는 '글자'가 아니라 'UTF-8 바이트' 기준 — 한글은 글자당 3바이트라
        # 4천자 페이지가 1.2만 바이트를 넘는다. 페이지 청킹이므로 최대치(65535)로.
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("meta", DataType.VARCHAR, max_length=2048)  # JSON: META_FIELDS(page/source)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=settings.embedding_dim)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        self.client.create_collection(
            collection_name=self.collection, schema=schema, index_params=index_params
        )

    def drop(self) -> None:
        if self.client.has_collection(self.collection):
            self.client.drop_collection(self.collection)

    def delete_by_doc_id(self, doc_ids: List[str]) -> None:
        """해당 문서들의 기존 청크 전부 삭제(재적재 시 stale 청크 방지)."""
        if not doc_ids or not self.client.has_collection(self.collection):
            return
        expr = f"doc_id in {json.dumps(list(doc_ids), ensure_ascii=False)}"
        self.client.delete(collection_name=self.collection, filter=expr)

    def upsert(self, chunks: List[Dict], embeddings: List[List[float]]) -> None:
        # 개수 불일치 시 zip 이 조용히 잘려 store 간 데이터 불일치가 생긴다 → 먼저 막는다.
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks({len(chunks)}) 와 embeddings({len(embeddings)}) 개수가 다릅니다."
            )
        rows = [
            {
                "chunk_id": c["chunk_id"],
                "doc_id": c["doc_id"],
                "text": c["text"],
                "meta": _dump_meta(c),
                "embedding": emb,
            }
            for c, emb in zip(chunks, embeddings)
        ]
        self.client.upsert(collection_name=self.collection, data=rows)

    def search(self, query_embedding: List[float], top_k: int) -> List[Dict]:
        res = self.client.search(
            collection_name=self.collection,
            data=[query_embedding],
            anns_field="embedding",
            limit=top_k,
            search_params={"metric_type": "COSINE", "params": {"ef": 128}},
            output_fields=["chunk_id", "doc_id", "text", "meta"],
        )
        results: List[Dict] = []
        for h in res[0]:
            e = h["entity"]
            item = {
                "chunk_id": e["chunk_id"],
                "doc_id": e.get("doc_id"),
                "text": e["text"],
                "score": h["distance"],
            }
            item.update(_load_meta(e.get("meta")))
            results.append(item)
        return results
