from langchain_openai import OpenAIEmbeddings

from app.config import get_settings

_embedder: OpenAIEmbeddings | None = None


def get_embedder() -> OpenAIEmbeddings:
    global _embedder
    if _embedder is None:
        settings = get_settings()
        _embedder = OpenAIEmbeddings(
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
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
    return await get_embedder().aembed_query(text)
