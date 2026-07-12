from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5434/ragkit"

    llm_model: str = "openai:gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    jwt_secret: str = "change-me"
    jwt_ttl_hours: int = 12

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    reranker: str = "none"
    cohere_api_key: str = ""

    # Retrieval knobs — see docs/adr/0002 for how these defaults were chosen.
    retrieval_arm_size: int = 20
    retrieval_final_k: int = 6
    rrf_k: int = 60

    max_input_chars: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()
