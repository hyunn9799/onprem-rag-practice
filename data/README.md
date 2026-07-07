# data/ — 문서 데이터 레이아웃

RAG 파이프라인의 데이터 흐름:

```
data/raw/*.pdf        원본 PDF (효성 매거진 등)        ← 여기에 파일을 넣는다
   │  (1번 담당: PDF 파서)
   ▼
data/processed/*.jsonl  파싱 결과 (텍스트+메타데이터)   ← 적재가 읽는 입력
   │  (2번 담당: make ingest)
   ▼
Milvus + OpenSearch   벡터/BM25 인덱스
```

## 폴더

| 경로 | 용도 | git |
|---|---|---|
| `raw/` | 원본 PDF 를 넣는 곳 | 추적 안 함(사내문서·대용량) |
| `processed/` | 파싱된 문서 JSONL | 추적 안 함 |
| `eval/qrels.jsonl` | 검색 평가 정답셋 | 추적함 |

> PDF 원본과 파싱 결과는 `.gitignore` 로 커밋되지 않는다. 폴더 자체는 `.gitkeep` 로 유지된다.

## `processed/*.jsonl` 계약 (1번 → 2번 인터페이스)

**페이지 단위 청킹(요구사항)** 이라 파서는 **페이지당 한 줄**을 출력한다.

```json
{"doc_id": "2023_..._vol.3_kor", "text": "해당 페이지 텍스트", "page": 12, "source": "2023_..._vol.3_kor.pdf"}
```

- **필수**: `doc_id`(문서 id, 파일 stem), `text`(페이지 본문), `page`(1-base).
- **선택**: `section`, `source`(파일명) — 출처 표시용.
- 적재(page 전략)에서 이 한 줄이 그대로 **1 청크**가 된다: `chunk_id = "{doc_id}#p{page}"`
  (Milvus PK = OpenSearch _id). 페이지가 BGE-M3 8192 토큰에 들어가는 게 실측 확인됨.
- 전략은 `CHUNK_STRATEGY=page|char` 로 바꿀 수 있다(기본 page).

## 적재 실행

```bash
make ingest    # data/processed/*.jsonl 을 읽어 청크→임베딩→저장.
               # processed 가 비어 있으면 샘플 문서로 대체 적재(스모크 테스트용).
```
