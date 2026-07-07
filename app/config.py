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
    opensearch_analyzer: str = "standard"  # "nori" 로 바꾸면 한국어 형태소 분석

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

    # Chunking
    chunk_strategy: str = "page"   # "page"(1페이지=1청크) | "char"(문자 슬라이딩)
    chunk_size: int = 500          # char 전략일 때만
    chunk_overlap: int = 80        # char 전략일 때만 (size > overlap)

    # Retrieval knobs
    bm25_top_k: int = 20
    vector_top_k: int = 20
    rrf_k: int = 60
    final_top_k: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
