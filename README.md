# 온프레미스 하이브리드 RAG 실습 (Lightning AI T4)

BGE-M3 + A.X-4.0-Light + Milvus + OpenSearch 로 구성한 **하이브리드(BM25+벡터) RAG** 실무형 레포.
DB(OpenSearch/Milvus)와 모델 서버(TEI 임베딩·리랭커, vLLM LLM)를 **모두 Docker 로 서빙**하고,
파이썬 앱은 그것들을 HTTP 로 호출하는 **가벼운 오케스트레이터**다. `make` 로 실행을 규격화했다.

---

## 1. 공유한 설계 검증 결과

| 항목 | 판정 | 근거 |
|---|---|---|
| Milvus/OpenSearch 를 Docker 로 | ✅ 맞음 | 둘 다 상태 저장 서버. Milvus standalone 은 `milvus-standalone/minio/etcd` 3컨테이너, 포트 19530 |
| 모델은 처음엔 Python 직접 로드 | ✅ 맞음 | CUDA/vLLM/bnb 문제를 초반에 몰아 터뜨리지 않는 좋은 학습 순서 |
| BGE-M3 dense = 1024차원 | ✅ 정확 | Milvus collection `dim=1024` 와 일치시켜야 함 |
| A.X-4.0-Light = 7B, 16,384 컨텍스트 | ✅ 정확 | T4 에서는 bf16 대신 fp16/4bit 가 현실적 → 4bit 방향 맞음 |
| T4 16GB, INT4 지원 | ✅ 정확 | 7B 4bit + RAG 컨텍스트 여유 확보 가능 |

→ **방향은 거의 그대로 유효하다.** 아래는 실무 기준 보강점.

## 2. 실무 관점 보강점 6가지 (이 레포에 반영됨)

1. **RRF 병합 추가** — BM25 점수와 코사인 유사도는 스케일이 달라 단순 합산 불가.
   `app/retrievers/fusion.py` 의 Reciprocal Rank Fusion 으로 순위 기반 병합. (원래 계획에 빠져 있던 핵심 조각)
2. **공유 chunk_id 키** — Milvus PK 와 OpenSearch `_id` 를 `doc_id#순번` 으로 통일.
   그래야 두 검색 결과를 RRF 로 합칠 수 있고, 재적재 시 upsert 되어 중복이 안 쌓인다.
3. **한국어 nori 분석기** — OpenSearch 기본 `standard` analyzer 는 한국어 토큰화가 약하다.
   `Dockerfile.opensearch` 로 `analysis-nori` 설치 후 `OPENSEARCH_ANALYZER=nori` 로 BM25 품질↑.
4. **리랭커 단계 구현** — 계획에 언급만 됐던 재정렬을 `bge-reranker-v2-m3` 로 실제 구현.
   T4 여유 없으면 `USE_RERANKER=false` 로 끄면 됨.
5. **모델 서빙 분리** — 임베딩/리랭커/LLM 을 앱 프로세스에 직접 로드하지 않고 별도 컨테이너
   (TEI x2 + vLLM)로 띄운 뒤 HTTP 호출. 앱 재배포 시 모델 재로딩이 없고 GPU 를 모델 서버로 격리한다.
   단, 단일 T4(16GB)엔 셋 다 fp16 으로 못 올라가므로 LLM 은 OpenAI 로 빼거나(권장) 4bit 로 돌린다(6절).
6. **버전 고정** — `:3`/`:latest` 대신 OpenSearch `3.1.0`, Milvus `v2.6.x`, pip 패키지 전부 핀 고정.

> 참고: Milvus 2.5+ 는 내장 BM25(full-text) 를 지원하므로 OpenSearch 없이 Milvus 하나로도 하이브리드가 가능하다.
> 다만 "키워드 엔진 + 벡터 DB 분리" 는 실제 사내 검색에서 매우 흔한 구성이라 학습 가치가 있어 두 엔진 구성을 유지했다.

---

## 3. 실행 순서

모든 명령은 `Makefile` 로 규격화되어 있다. (`make help` 로 전체 목록)
> Windows 는 git bash / WSL 에서 `make` 실행. `make` 가 없으면 각 target 의 명령을 직접 복사해 쓴다.
> **T4 에서 처음 띄운다면 → [`docs/t4-startup-checklist.md`](docs/t4-startup-checklist.md)** (GPU 패스스루·메모리 경로·스모크 테스트).

```bash
# 0) 준비
make sync                 # uv 로 의존성 설치 (앱은 HTTP 클라이언트만 — 모델 라이브러리 없음)
cp .env.example .env      # 필요시 LLM_PROVIDER / OPENAI_API_KEY 등 수정

# 1) 컨테이너 기동 — ★권장: 분리 모드★ (상태=로컬, 연산=GPU)
make up-db                # 노트북: DB(OpenSearch/Milvus)만
make up-models-gpu        # T4(Lightning): TEI 임베딩+리랭커만 GPU 로
#   → Lightning 에서 포트 8080/8081 공개 후 .env 의 *_API_URL 을 원격 URL 로
# 올인원 대안:
make up-cpu               #   노트북에 전부(CPU) — RAM 12GB+ 필요, 부족 시 TEI OOM
make up-search            #   T4 에 DB+TEI 전부
make up                   #   T4 전부 + vLLM (메모리 빠듯 — 6절)

# 2) 상태 확인
make health               # OpenSearch / Milvus / TEI x2 / vLLM 헬스 체크
make ps                   # 컨테이너 상태

# 3) ★적재 먼저★ (이게 없으면 검색 0건)
make ingest               # 샘플 문서 → 임베딩 → Milvus/OpenSearch 저장

# 4) 실행 (둘 중 하나)
make query                # 단발 질의: 검색 -> RRF -> 리랭크 -> 답변
make serve                # FastAPI 서버 (POST /ingest, /ask) — http://localhost:8000
#   POST /ingest  {"documents":[{"doc_id":"d1","text":"..."}]}
#   POST /ask     {"question":"..."}
```

**서비스 구성 / 포트**

| 계층 | 서비스 | 포트 | 비고 |
|---|---|---|---|
| 검색 DB | OpenSearch (BM25) | 9200 | nori 쓰려면 `Dockerfile.opensearch` (4·6절) |
| 벡터 DB | Milvus standalone | 19530 | etcd/minio 동반 |
| 모델 | TEI — BGE-M3 임베딩 | 8080 | CPU 기본 / GPU 는 `docker-compose.gpu.yml` |
| 모델 | TEI — bge-reranker | 8081 | CPU 기본 / GPU 오버레이 |
| 모델 | vLLM — A.X-4.0-Light | 8001 | GPU 전용. `LLM_PROVIDER=openai` 면 불필요 |
| 앱 | FastAPI | 8000 | |

**LLM 전환(로컬 ↔ OpenAI)** — `.env` 만 바꾸면 된다. vLLM 이 OpenAI 호환 API 라 코드는 그대로.
```bash
LLM_PROVIDER=local                 # 기본: 로컬 vLLM (http://localhost:8001/v1)
# LLM_PROVIDER=openai              # 전환: OpenAI
# OPENAI_API_KEY=sk-...            #   ↳ 키 필수. GPU 는 검색모델만, LLM 은 API 로 뺌
```

## 4. 데이터 흐름

```
질문
 └─ TEI(BGE-M3) 임베딩 ─▶ Milvus 벡터 검색(top 20) ┐
 └─ 원문 그대로       ─▶ OpenSearch BM25(top 20)  ┴─▶ RRF 병합 ─▶ TEI(reranker) 재정렬(top 5) ─▶ vLLM/OpenAI 답변
```
(임베딩·리랭크·생성은 모두 별도 컨테이너에 HTTP 호출. 앱은 검색 오케스트레이션과 병합만 담당.)

## 5. 폴더 구조

```
onprem-rag-practice/
├── docker-compose.yml              # 통합 스택(기본 CPU): DB + TEI x2(models) + vLLM(llm)
├── docker-compose.gpu.yml          # GPU 오버레이: TEI 를 T4 이미지 + GPU 예약으로
├── docker-compose.opensearch.yml   # (구) OpenSearch 단독 — 참고용
├── Dockerfile.opensearch           # nori 분석기 포함 커스텀 이미지(선택)
├── Makefile                        # up / up-search / ingest / serve / query / health ...
├── docs/t4-startup-checklist.md    # T4 실기동 체크리스트
├── .env.example
├── requirements.txt                # pip 미러 (uv 는 pyproject.toml 사용)
└── app/
    ├── config.py                   # pydantic-settings 중앙 설정 (서비스 URL 포함)
    ├── parsing/pdf_parser.py       # PDF → data/processed JSONL (1번, 현재 스텁)
    ├── embeddings/bge_m3_embedder.py  # TEI /embed HTTP 클라이언트
    ├── retrievers/
    │   ├── opensearch_retriever.py # BM25
    │   ├── milvus_retriever.py     # 벡터
    │   └── fusion.py               # RRF
    ├── rerank/bge_reranker.py      # TEI /rerank HTTP 클라이언트
    ├── llm/ax_light.py             # OpenAI 호환 클라이언트 (vLLM ↔ OpenAI)
    ├── ingestion.py                # 두 store 동시 적재
    ├── hybrid_search.py            # 전체 파이프라인
    ├── eval/                       # 검색 비교 평가 (Hit@k/MRR/nDCG)
    │   ├── metrics.py              #   순위 기반 지표(LLM 불필요)
    │   └── __main__.py             #   BM25/vector/RRF/+rerank 비교 러너
    ├── ingestion.py                # ↑ 위. data/processed/*.jsonl 을 읽어 적재
    └── main.py                     # FastAPI
data/                              # 문서 데이터 (data/README.md 참고)
├── raw/                           #   원본 PDF 를 넣는 곳 (git 추적 X)
├── processed/                     #   파싱 결과 JSONL (1번 산출물, git 추적 X)
└── eval/qrels.jsonl              #   평가 정답셋 (질문 → 정답 chunk_id)
```

## 6. T4 운영 주의

- **메모리 현실(중요)**: TEI(bge-m3)~3GB + TEI(reranker)~3GB + vLLM(7B fp16)~14GB ≈ 20GB > T4 16GB.
  단일 T4 라면 둘 중 하나로 간다:
  - **(권장) LLM 만 OpenAI 로** — `.env` 에서 `LLM_PROVIDER=openai` + `OPENAI_API_KEY`.
    GPU 는 임베딩/리랭커(~6GB)만 쓰고 LLM 은 API 로 뺀다.
  - **(전부 로컬) vLLM 4bit** — `docker-compose.yml` 의 vllm `command` 에서
    `--quantization=bitsandbytes` / `--load-format=bitsandbytes` 두 줄 주석 해제. 메모리 빠듯함 감수.
- **TEI 이미지 태그**: T4 는 Turing 이라 `text-embeddings-inference:turing-1.5` 태그를 쓴다(다른 GPU 는 태그 다름).
- DB 컨테이너에 GPU 예약을 넣지 말 것. GPU 는 모델 서버(TEI/vLLM) 전용. GPU 없는 개발 PC 는 `make up-db` 로 DB 만.
- OpenSearch heap 은 `-Xms512m -Xmx512m` 로 시작, 문서 늘면 1g 로.
- 게이티드 모델은 `.env` 의 `HF_TOKEN` 으로 다운로드한다(모델 캐시는 `hf-cache` 볼륨에 유지).

## 7. 검색 평가 (`make eval`)

검색 품질을 **숫자로** 비교한다. LLM 없이 순위 지표만 쓰므로 `up-search` 로도 돌아간다.

```bash
make ingest      # 정답셋과 같은 문서가 적재돼 있어야 함
make eval        # BM25 / vector / RRF / RRF+rerank 를 같은 정답셋으로 채점
```

- **정답셋**: `data/eval/qrels.jsonl` — 한 줄에 `{"question": "...", "relevant": ["docid#0"]}`.
  실제 문서를 적재한 뒤, 답이 들어있는 chunk_id 를 정답으로 20~30개 채운다.
- **지표**: Hit@1/3/5, MRR@10, nDCG@10, Recall@5 (`app/eval/metrics.py`).
- **핵심 실험**: `USE_RERANKER` on/off, `standard` vs `nori` 분석기를 바꿔가며 지표 변화를 비교 →
  "리랭커/하이브리드/형태소 분석이 검색을 얼마나 올렸나"가 이 프로젝트의 대표 숫자다.

출력 예:
```
method        Hit@1  Hit@3  Hit@5  MRR@10  nDCG@10  Recall@5
bm25          ...
vector        ...
rrf           ...
rrf+rerank    ...
```
