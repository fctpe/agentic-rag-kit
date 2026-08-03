from langchain_openai import OpenAIEmbeddings

from app.config import get_settings
from app.observability import GEN_AI_OPERATION_NAME, GEN_AI_REQUEST_MODEL, tracer

_embedder: OpenAIEmbeddings | None = None

# The same hung-call failure ADR 0005 closed for chat, on the arm it missed. The
# OpenAI client ships with no timeout, and this call sits inside hybrid_search,
# inside the tool node, inside the request — holding its AsyncSession open for
# as long as the provider keeps the socket.
EMBEDDING_TIMEOUT_SECONDS = 60


def get_embedder() -> OpenAIEmbeddings:
    global _embedder
    if _embedder is None:
        settings = get_settings()
        _embedder = OpenAIEmbeddings(
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
            timeout=EMBEDDING_TIMEOUT_SECONDS,
            # Send plain strings, not pre-tokenized arrays: chunks are already
            # bounded (<=700 tokens) and some endpoints reject token arrays.
            check_embedding_ctx_length=False,
        )
    return _embedder


async def embed_texts(texts: list[str], batch_size: int = 128) -> list[list[float]]:
    embedder = get_embedder()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        vectors.extend(await embedder.aembed_documents(batch))
    return vectors


async def embed_query(text: str) -> list[float]:
    # The vector arm's only separately timeable step — the arms themselves are
    # one SQL statement. No token attributes: LangChain's embeddings interface
    # returns vectors and drops the provider's usage block.
    settings = get_settings()
    with tracer.start_as_current_span(
        f"embeddings {settings.embedding_model}",
        attributes={
            GEN_AI_OPERATION_NAME: "embeddings",
            GEN_AI_REQUEST_MODEL: settings.embedding_model,
        },
    ):
        return await get_embedder().aembed_query(text)
