# 실행 가이드 (재시작 루틴 + 팀원 온보딩)

> `make` 가 없는 환경(Windows git bash 기본)은 각 명령 옆의 **원본 명령**을 그대로 복사해 쓰면 된다.

---

## 0. 아키텍처 한 장 요약

```
[내 노트북]                                [Lightning T4 GPU]
 파이썬 앱 (검색 오케스트레이션)   ──HTTPS──▶  TEI 임베딩 (BGE-M3, :8080)
 OpenSearch :9200 (BM25)          ◀──벡터──   TEI 리랭커 (:8081)
 Milvus :19530 (벡터 DB)
 └ 데이터 저장은 전부 로컬 Docker 볼륨          └ 무거운 모델 연산만 GPU
```

- **왜 분리?** 노트북 RAM(16GB, Docker 11GB 한도)에 모델 2개를 올리면 OOM. 데이터(상태)는 로컬, 모델(연산)은 GPU.
- LLM(답변 생성)은 별도: `.env` 의 `OPENAI_API_KEY` (ingest/eval 에는 필요 없음).

---

## 1. 전체 껐다 켰을 때 — 재시작 루틴

### ① 노트북: DB 켜기

```bash
# Docker Desktop 먼저 실행 (트레이 아이콘 확인)
cd onprem-rag-practice
docker compose up -d                    # = make up-db  (DB 4개: opensearch, milvus, etcd, minio)

# 확인 (둘 다 200 이면 OK — OpenSearch 는 부팅에 ~30초)
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:9200
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:9091/healthz
```

### ② T4: 모델 켜기 (Lightning 스튜디오 시작 후)

```bash
cd onprem-rag-practice && git pull
make up-models-gpu
# 원본: docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile models up -d tei-embedding tei-reranker

docker logs -f tei-embedding            # "Ready" 뜰 때까지 (모델 캐시 있으면 ~1분)
docker logs tei-reranker --tail 3       # 역시 "Ready"
nvidia-smi                              # 두 프로세스 합계 ~2.6GB 정상
```

### ③ 포트 공개 + .env 갱신 ★스튜디오 재시작마다 필수★

1. Lightning UI 우측 → **Port 플러그인** → `8080`, `8081` 추가 (Auth 끄기=public)
2. 생성된 URL 을 노트북 `.env` 에 반영:

```bash
EMBEDDING_API_URL=https://8080-<스튜디오ID>.cloudspaces.litng.ai
RERANKER_API_URL=https://8081-<스튜디오ID>.cloudspaces.litng.ai
```

> ⚠️ 스튜디오를 껐다 켜면 **URL 이 바뀔 수 있다.** 연결 안 되면 제일 먼저 이걸 의심.

### ④ 노트북: 연결 확인 → 테스트

```bash
# 원격 모델 살아있나 (200 이면 OK)
curl -s -o /dev/null -w "%{http_code}\n" $EMBEDDING_API_URL/health   # 또는 URL 직접 입력

# 진짜 왕복 테스트 (한글은 curl 로 보내면 Windows 인코딩에 깨지므로 파이썬으로!)
uv run python -c "from app.embeddings.bge_m3_embedder import BGEM3Embedder; print('dim:', len(BGEM3Embedder().embed_query('테스트')))"
# → dim: 1024 나오면 전체 연결 완료
```

### ⑤ 데이터 & 평가

```bash
# 적재는 "처음 1회" 또는 "스키마/청킹 변경 후"만. 데이터는 Docker 볼륨에 남아있다.
uv run python -m app.ingestion          # = make ingest   (적재: 73청크)
uv run python -m app.eval               # = make eval     (BM25/vector/RRF/+rerank 비교표)

# 스키마 바꿨거나 store 불일치 의심 때 (drop 후 재구축):
uv run python -c "from app.ingestion import reindex; print(reindex())"   # = make reindex
```

### 끝낼 때

```bash
docker compose down                     # 노트북 (볼륨 유지 — 데이터 안 날아감)
# T4: Lightning 스튜디오 정지 (과금 방지)
```

---

## 2. 팀원 온보딩 — 레포 받고 처음 할 일

### 사전 설치 (1회)

| 도구 | 용도 | 설치 |
|---|---|---|
| **Docker Desktop** | DB(+모델) 컨테이너 | docker.com — 설치 후 WSL2 백엔드 확인 |
| **uv** | 파이썬 패키지/실행 | `winget install astral-sh.uv` 또는 curl 스크립트 |
| git | 코드 | 이미 있을 것 |

> **모델을 직접 설치할 필요 없음!** BGE-M3 등 모델은 TEI 컨테이너가 첫 실행 때 HuggingFace 에서 자동 다운로드한다(~2.3GB/개, 볼륨에 캐시됨). 파이썬 쪽엔 torch 같은 무거운 것도 없다 — 앱은 HTTP 클라이언트만.

### 셋업 순서

```bash
# 1) 코드 + 의존성
git clone <레포 URL> && cd onprem-rag-practice
uv sync                                  # 가벼움 (모델 라이브러리 없음)

# 2) 설정
cp .env.example .env                     # 그리고 아래 "모델 연결" 선택에 맞게 편집

# 3) ★PDF 는 git 에 없다★ (사내문서라 .gitignore 됨)
#    공유받은 효성 PDF 4개를 data/raw/ 에 넣기

# 4) DB 켜기
docker compose up -d

# 5) 파싱 → 적재
uv run python -m app.parsing.pdf_parser   # PDF → data/processed/*.jsonl (페이지당 1줄)
uv run python -m app.ingestion            # 임베딩(원격/로컬) → Milvus + OpenSearch

# 6) 동작 확인
uv run python -m app.eval                 # 검색 품질 비교표가 뜨면 성공
uv run --no-project --with pytest python -m pytest -q   # 단위 테스트 (스택 없이도 OK)
```

### 모델 연결 — 3가지 중 택1 (`.env` 만 다름)

| 선택 | 방법 | .env 설정 |
|---|---|---|
| **A. 공용 T4 URL 받기 (제일 쉬움)** | GPU 담당자가 띄운 URL 을 공유받음 | `EMBEDDING_API_URL`/`RERANKER_API_URL` 에 그 URL |
| B. 자기 노트북 CPU | `docker compose --profile models up -d` (RAM 12GB+ 필요, 리랭커까지는 OOM 위험 → `USE_RERANKER=false` 권장) | localhost URL 그대로 |
| C. 자기 GPU 있음 | `make up-models-gpu` | localhost URL 그대로 |

### LLM (답변 생성) — 검색/평가에는 불필요

`/ask` 로 실제 답변까지 보려면 `.env` 에 `LLM_PROVIDER=openai` + `OPENAI_API_KEY=sk-...`.
검색 개발/평가(eval)만 하면 키 없어도 된다.

---

## 3. 자주 터지는 것 (FAQ)

| 증상 | 원인/해결 |
|---|---|
| TEI 컨테이너 exit 137 | **메모리 부족(OOM)**. 노트북이면 `%UserProfile%\.wslconfig` 에 `[wsl2] memory=11GB` + `wsl --shutdown`. 그래도 안 되면 모델은 원격(택1 A)으로 |
| 원격 URL 이 안 열림/404 | 스튜디오 재시작으로 URL 변경됨 → Port 플러그인에서 새 URL 확인해 .env 갱신. 포트 공개 직후엔 몇 초 걸릴 수 있음 |
| curl 로 한글 보내면 400 "invalid unicode" | Windows 셸 인코딩(cp949) 문제. 서버 정상 — **한글 테스트는 파이썬으로** (위 ④의 dim 테스트) |
| `make: command not found` | git bash 에 make 없음 → 각 명령의 원본(위에 병기)을 복사 실행 |
| 검색 결과 0건 | 적재 안 함 → `uv run python -m app.ingestion`. `data/processed` 가 비면 파싱부터 |
| `data/processed 에 문서가 없습니다` 에러 | 정상 동작(샘플 오적재 방지 가드). PDF 넣고 `make parse` 먼저. 급하면 `INGEST_ALLOW_SAMPLE=1` |
| Milvus "length exceeds max length" | 스키마 구버전 → `reindex` 로 재구축 (text 필드는 65535 바이트로 수정돼 있음) |
| 첫 모델 기동이 5~10분 | 정상 (HF 다운로드). 두 번째부터는 볼륨 캐시로 ~1분 |

---

## 4. 명령어 치트시트

| 하고 싶은 것 | make | 원본 명령 |
|---|---|---|
| DB만 켜기 (노트북) | `make up-db` | `docker compose up -d` |
| 모델만 GPU 로 (T4) | `make up-models-gpu` | `docker compose -f docker-compose.yml -f docker-compose.gpu.yml --profile models up -d tei-embedding tei-reranker` |
| 전부 CPU 로 (RAM 12GB+) | `make up-cpu` | `docker compose --profile models up -d` |
| PDF 파싱 | `make parse` | `uv run python -m app.parsing.pdf_parser` |
| 적재 | `make ingest` | `uv run python -m app.ingestion` |
| 재구축 | `make reindex` | `uv run python -c "from app.ingestion import reindex; print(reindex())"` |
| 검색 평가 | `make eval` | `uv run python -m app.eval` |
| 단발 질의 | `make query` | `uv run python -m app.hybrid_search` |
| API 서버 | `make serve` | `uv run uvicorn app.main:app --host 0.0.0.0 --port 8000` |
| 단위 테스트 | `make test` | `uv run --no-project --with pytest python -m pytest -q` |
| 전부 끄기 | `make down` | `docker compose --profile models --profile llm down` |
