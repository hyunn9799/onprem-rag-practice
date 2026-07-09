"""중앙 설정. pydantic-settings 로 .env 를 타입 안전하게 로드한다.

실무 포인트: 접속 정보/하이퍼파라미터를 코드에 하드코딩하지 않고 한 곳에서 관리.
어디서든 `from app.config import settings` 로 재사용한다.
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenSearch
    opensearch_host: str = "localhost"
    opensearch_port: int = 9200
    opensearch_index: str = "rag_chunks"
    opensearch_analyzer: str = "nori"      # 한국어 형태소 분석(기본). 이미지에 analysis-nori 필요
    recreate_index: bool = False           # True: analyzer 불일치 시 index 재생성(재적재 필요)

    # Milvus
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "rag_chunks"

    # Models — 파이썬에 직접 로드하지 않고 별도 컨테이너(TEI/vLLM)로 서빙. 앱은 HTTP 호출.
    embedding_model: str = "BAAI/bge-m3"          # TEI 컨테이너가 로드(참고용)
    embedding_dim: int = 1024
    embedding_api_url: str = "http://localhost:8080"   # TEI: BGE-M3 dense
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_api_url: str = "http://localhost:8081"    # TEI: cross-encoder rerank
    use_reranker: bool = True

    # LLM — vLLM 은 OpenAI 호환 API 라 로컬/OpenAI 를 같은 클라이언트로 호출한다.
    #   llm_provider=local  → vLLM(로컬, 기본)
    #   llm_provider=openai → OpenAI (openai_api_key 필요)
    llm_provider: str = "local"                        # "local" | "openai"
    llm_model: str = "skt/A.X-4.0-Light"
    llm_base_url: str = "http://localhost:8001/v1"     # vLLM OpenAI 호환 엔드포인트
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    # Retrieval knobs
    bm25_top_k: int = 20
    vector_top_k: int = 20
    rrf_k: int = 60
    final_top_k: int = 5

    # Ingest / Eval
    ingest_allow_sample: bool = False          # True: processed 비었을 때 샘플 적재 허용
    eval_qrels: str = "data/eval/qrels.jsonl"  # 평가 정답셋 경로


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
