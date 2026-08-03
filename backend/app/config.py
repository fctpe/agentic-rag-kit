from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://rag:rag@localhost:5434/ragkit"

    llm_model: str = "openai:gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    # No usable default: an empty/placeholder secret is rejected at startup so a
    # forgeable "change-me" JWT signing key can never ship (see main.lifespan).
    jwt_secret: str = ""
    jwt_ttl_hours: int = 12

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # Unset endpoint = tracing off, no exporter, no background thread. Mirrored
    # into the process environment at startup so the OTLP exporter's own
    # resolution (path suffix, header parsing) applies — see app/observability.py.
    otel_exporter_otlp_endpoint: str = ""
    otel_exporter_otlp_headers: str = ""

    log_level: str = "INFO"

    # USD per million tokens for llm_model. Left at 0 the spans carry token
    # counts and no cost: a price table baked into the repo goes stale without
    # anyone noticing, and a cost of 0.0 reads as "free" rather than "unpriced".
    llm_input_price_per_mtok: float = 0.0
    llm_output_price_per_mtok: float = 0.0

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
