# T4 기동 체크리스트 (`make up`)

Lightning AI T4(16GB) 환경에서 스택을 실제로 띄울 때의 순서·검증·함정 모음.
핵심 관문은 딱 두 개다: **① GPU 가 도커 컨테이너 안에서 보이는가**, **② 16GB 에 뭘 올릴 것인가**.

---

## 0. 결론부터 — 어떤 경로로 갈지 먼저 정한다

세 모델을 fp16 으로 다 올리면 `~20GB > 16GB` 라 **단일 T4 에선 안 된다.** 기동 전에 택1:

| 경로 | 명령 | GPU 사용 | LLM | 추천 |
|---|---|---|---|---|
| **A. 검색만 로컬 + LLM=OpenAI** | `make up-search` | TEI 2개 ~6GB | OpenAI API | ★ 권장 |
| **B. 전부 로컬 (vLLM 4bit)** | `make up` (+4bit 설정) | ~15GB 빠듯 | vLLM | 도전용 |

- **A** 를 쓸 거면 `.env` 에 `LLM_PROVIDER=openai` + `OPENAI_API_KEY=sk-...` 를 먼저 넣는다. `make up-search` 는 vLLM 을 아예 안 띄운다.
- **B** 를 쓸 거면 `docker-compose.yml` 의 `vllm.command` 에서 `--quantization=bitsandbytes` / `--load-format=bitsandbytes` 두 줄 주석을 해제한다.

---

## 1. Pre-flight — 호스트 점검 (기동 전, 한 번)

| # | 확인 | 명령 | 통과 기준 |
|---|---|---|---|
| 1 | GPU 인식 | `nvidia-smi` | Tesla T4 / 15360MiB 표시 |
| 2 | **도커 GPU 패스스루** ★ | `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi` | 컨테이너 안에서 T4 보임 |
| 3 | compose v2 | `docker compose version` | v2.x |
| 4 | uv | `uv --version` | 버전 출력 |
| 5 | 디스크 여유 | `df -h .` | **50GB+** 여유 (모델·이미지 캐시) |
| 6 | .env 준비 | `cp .env.example .env` | 경로 A 면 OPENAI_API_KEY 기입 |
| 7 | HF 토큰(선택) | `.env` 의 `HF_TOKEN=` | 게이티드 모델 다운로드용 |

> **2번이 최대 관문.** Lightning Studio 자체가 컨테이너라 중첩 도커에서 `--gpus` 가 막힐 수 있다.
> 여기서 실패하면 → 아래 [트러블슈팅 T1](#7-흔한-실패--해결) 참고. (안 되면 모델을 도커 대신 호스트 프로세스로 띄우는 대안)

---

## 2. 의존성 설치

```bash
make sync            # uv sync — 앱은 HTTP 클라이언트만 설치(가벼움)
```
통과 기준: 에러 없이 완료. (`torch/transformers` 같은 무거운 다운로드가 **없어야** 정상 — 그건 컨테이너에 있음)

---

## 3. 기동 (경로 A 기준)

```bash
make up-search       # DB(OpenSearch/Milvus) + TEI 임베딩 + TEI 리랭커
make ps              # 컨테이너 상태 확인
```

기대 컨테이너: `opensearch-dev`, `milvus-standalone`, `milvus-etcd`, `milvus-minio`, `tei-embedding`, `tei-reranker`.
(경로 B 는 `make up` — 여기에 `vllm` 추가)

---

## 4. 모델 다운로드 대기 (첫 기동만 오래 걸림)

TEI/vLLM 는 뜨자마자 HuggingFace 에서 모델을 받는다. **`ps` 가 running 이어도 아직 준비 안 됐을 수 있다.** 로그로 "준비 완료"를 확인한다.

```bash
make logs                                             # 전체 로그 (또는 아래로 개별)
docker compose --profile models logs -f tei-embedding # "Ready" / "Starting HTTP server" 나올 때까지
docker compose --profile models logs -f tei-reranker
# 경로 B:
docker compose --profile llm logs -f vllm             # "Application startup complete" / "Uvicorn running"
```

BGE-M3(~2.3GB), reranker(~2.3GB) 다운로드라 네트워크에 따라 수 분. 재기동 시엔 `hf-cache` 볼륨에 남아 빠르다.

---

## 5. 헬스 & 스모크 테스트 (엔드포인트 직접 타격)

```bash
make health          # 5개 서비스 일괄 체크
```

개별로 실제 응답까지 확인 (여기까지 통과하면 파이프라인 준비 완료):

```bash
# OpenSearch
curl -s localhost:9200 | head

# Milvus
curl -s localhost:9091/healthz            # "OK"

# TEI 임베딩 — 1024개 float 배열이 나와야 함
curl -s localhost:8080/embed -X POST -H 'Content-Type: application/json' \
  -d '{"inputs":["지체상금은 하루에 얼마인가"]}' | python -c "import sys,json;v=json.load(sys.stdin)[0];print('dim=',len(v))"
#  → dim= 1024

# TEI 리랭커 — index/score 배열
curl -s localhost:8081/rerank -X POST -H 'Content-Type: application/json' \
  -d '{"query":"지체상금","texts":["지체상금은 지연 1일당 0.1%","하자보수 보증기간은 2년"]}'

# vLLM (경로 B만) — 모델 목록에 A.X 보이면 OK
curl -s localhost:8001/v1/models
```

---

## 6. 파이프라인 실행

```bash
make ingest          # ★적재 먼저★ (안 하면 검색 0건)
make query           # 단발 질의 테스트
# 또는
make serve           # API 서버 → http://localhost:8000
```

`make ingest` 통과 기준: `ingested chunks: N` (N>0) 출력.
경로 A 면 이 시점에 `.env` 의 `LLM_PROVIDER=openai` 로 OpenAI 가 답변 생성.

---

## 7. 흔한 실패 & 해결

| ID | 증상 | 원인 | 해결 |
|---|---|---|---|
| **T1** | `could not select device driver "nvidia"` | nvidia-container-toolkit 미설정 / 중첩 도커 GPU 불가 | `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`. Lightning 에서 안 되면 모델을 호스트 프로세스로(도커 밖) 실행하거나 경로 A 로 |
| **T2** | vLLM `CUDA out of memory` | 7B fp16 이 T4 초과 | 4bit 주석 해제 / `--gpu-memory-utilization` 낮춤 / **경로 A(OpenAI)로 전환** |
| **T3** | vLLM `bfloat16 ... not supported` | T4(Turing)는 bf16 미지원 | `--dtype=float16` (이미 설정됨) 확인 |
| **T4** | TEI `unsupported CUDA arch` / 즉시 크래시 | 이미지 태그 불일치 | 태그가 `text-embeddings-inference:turing-1.5` 인지 확인 (T4=Turing) |
| **T5** | HF `401/403 gated` | 모델 접근 권한/토큰 | `.env` 에 `HF_TOKEN` 넣고 HF 에서 모델 승인 |
| **T6** | 첫 요청 타임아웃 / connection refused | 모델 아직 로딩 중 | 4절 로그로 "Ready" 확인 후 재시도 |
| **T7** | Milvus `unhealthy` | etcd/minio 가 늦게 뜸 | `docker compose logs milvus-standalone`, 잠시 후 재확인 |
| **T8** | 포트 충돌 (8000) | 앱(8000) vs vLLM(8001) 혼동 | 앱=8000, vLLM=8001 확인 |
| **T9** | 디스크 풀 | 모델/이미지 캐시 수십 GB | `docker system df`, 불필요 이미지 정리 |

---

## 8. 종료 / 정리

```bash
make down                        # 컨테이너 정지·삭제 (볼륨은 유지)
docker volume ls | grep onprem   # 데이터/모델 캐시 볼륨 확인
# 완전 초기화(주의: 적재 데이터·모델캐시 삭제):
# docker compose --profile models --profile llm down -v
```

---

### 한 장 요약 (경로 A, 처음부터)
```bash
nvidia-smi && docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi  # 관문
# .env: LLM_PROVIDER=openai + OPENAI_API_KEY 기입
make sync
make up-search        # DB + TEI 2개
# (4절 로그로 TEI "Ready" 확인)
make health
make ingest
make query
```
