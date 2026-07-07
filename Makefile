# 온프렘 하이브리드 RAG — 명령 규격화.
# 사용: make up / make ingest / make serve / make query / make down
# (Windows 는 git bash 또는 WSL 에서 make 실행. make 없으면 recipe 의 명령을 직접 복사 실행.)

COMPOSE = docker compose
GPU     = -f docker-compose.yml -f docker-compose.gpu.yml   # GPU 오버레이
MODELS  = --profile models                                  # DB + TEI
ALL     = --profile models --profile llm                    # + vLLM

.PHONY: help up up-search up-cpu up-db up-models-gpu down logs ps health sync ingest ingest-sample reindex serve query eval test

help:                ## 명령 목록
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-13s\033[0m %s\n", $$1, $$2}'

up-cpu:              ## 노트북 올인원: DB + TEI(CPU). RAM 12GB+ 필요 — 부족하면 분리 모드
	$(COMPOSE) $(MODELS) up -d

# ── 분리 모드(권장): 노트북=DB만(up-db), T4=모델만(up-models-gpu), .env 로 연결 ──
up-models-gpu:       ## ★T4 전용★ TEI 임베딩+리랭커만 GPU 로 (DB 없음)
	$(COMPOSE) $(GPU) $(MODELS) up -d tei-embedding tei-reranker

up-search:           ## T4: DB + TEI(GPU). vLLM 제외, LLM 은 OpenAI
	$(COMPOSE) $(GPU) $(MODELS) up -d

up:                  ## T4 전부: DB + TEI(GPU) + vLLM (메모리 빠듯 — README 6절)
	$(COMPOSE) $(GPU) $(ALL) up -d

up-db:               ## DB 계층(OpenSearch/Milvus)만
	$(COMPOSE) up -d

down:                ## 전체 컨테이너 정지/삭제
	$(COMPOSE) $(ALL) down

logs:                ## 전체 로그 팔로우
	$(COMPOSE) $(ALL) logs -f

ps:                  ## 컨테이너 상태
	$(COMPOSE) $(ALL) ps

health:              ## 서비스 헬스 체크
	@echo "OpenSearch:" && curl -s http://localhost:9200 >/dev/null && echo "  OK" || echo "  DOWN"
	@echo "Milvus:"     && curl -s http://localhost:9091/healthz >/dev/null && echo "  OK" || echo "  DOWN"
	@echo "TEI embed:"  && curl -s http://localhost:8080/health >/dev/null && echo "  OK" || echo "  (gpu 프로파일 미기동?)"
	@echo "TEI rerank:" && curl -s http://localhost:8081/health >/dev/null && echo "  OK" || echo "  (gpu 프로파일 미기동?)"
	@echo "vLLM:"       && curl -s http://localhost:8001/health >/dev/null && echo "  OK" || echo "  (LLM_PROVIDER=openai 면 불필요)"

sync:                ## 파이썬 의존성 설치
	uv sync

parse:               ## PDF(data/raw) → data/processed JSONL (파서 구현 후 사용)
	uv run python -m app.parsing.pdf_parser

ingest:              ## 문서 적재 (data/processed → 임베딩 → Milvus/OpenSearch). 검색 전 필수
	uv run python -m app.ingestion

ingest-sample:       ## 샘플 문서로 적재(개발용, 실문서 없을 때만 명시적으로)
	INGEST_ALLOW_SAMPLE=1 uv run python -m app.ingestion

reindex:             ## 두 store drop 후 data/processed 로 재구축(저장소 불일치/스키마 변경 후)
	uv run python -c "from app.ingestion import reindex; print('reindexed chunks:', reindex())"

serve:               ## FastAPI 서버 (POST /ingest, /ask)
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000

query:               ## 단발 질의 파이프라인 테스트
	uv run python -m app.hybrid_search

eval:                ## 검색 비교 평가 (BM25/vector/RRF/+rerank, Hit@k·MRR·nDCG). ingest 선행
	uv run python -m app.eval

test:                ## 순수 로직 단위 테스트 (스택 불필요). 무거운 의존성 테스트는 자동 skip
	uv run --no-project --with pytest python -m pytest -q
