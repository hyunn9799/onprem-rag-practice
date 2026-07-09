# data/ — 문서 데이터 레이아웃

RAG 파이프라인의 데이터 흐름:

```
data/raw/*.pdf        원본 PDF (효성 매거진 등)
   │  (1번 담당: app/parse_odl.py — ODL+OCR. 1번 환경에서 실행)
   ▼
data/processed/hyosung_chunks.json  pre-chunked 청크 배열  ← 적재가 읽는 입력
   │  (2번 담당: make ingest — chunk_id 있으면 재청킹 없이 적재)
   ▼
Milvus + OpenSearch   벡터/BM25 인덱스
```

> 자체 pdfplumber 파서(app/parsing/, `make parse-legacy`)는 학습/백업용이다.
> id 체계가 팀 파서와 달라 **두 산출물을 processed 에 같이 두면 안 된다.**

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
- **선택**: `source`(문서 제목) — 출처 표시용.
- 적재(page 전략)에서 이 한 줄이 그대로 **1 청크**가 된다: `chunk_id = "{doc_id}#p{page}"`
  (Milvus PK = OpenSearch _id). 페이지가 BGE-M3 8192 토큰에 들어가는 게 실측 확인됨.
- 청킹 전략/크기는 파서(1번) 소관 — 적재는 받은 청크를 검증·저장만 한다.

### pre-chunked 입력 (팀 파서 계약)

레코드에 **`chunk_id` 가 이미 있으면 재청킹 없이 그대로 적재**한다(청킹/ID 소유권은
발급자에게). 팀 파서의 `hyosung_chunks.json`(레코드 배열, `*.json`)을 processed 에
넣으면 바로 소비된다:

```json
{"chunk_id": "hyosung_vol4_p005", "doc_id": "hyosung_vol4",
 "source": "효성중공업 전력기술 매거진 Vol.4 (2024)", "page": 5, "text": "..."}
```

⚠️ pre-chunked 로 갈아타면 chunk_id 형식이 바뀌므로 `data/eval/qrels.jsonl` 의
정답 id 도 함께 마이그레이션해야 한다(`hyosung-2024-vol4#p5` → `hyosung_vol4_p005`).

## 적재 실행

```bash
make ingest         # data/processed/*.jsonl 을 읽어 청크→임베딩→저장.
                    # processed 가 비어 있으면 실패한다(샘플 오적재 방지 가드).
make ingest-sample  # 개발용 스모크 테스트 — 명시적으로만 샘플 문서 적재.
```
