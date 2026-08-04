"""The corpus half of the committed embeddings: one vector per fixture chunk.

`app/embedding_cache.py` holds the format, the key and the fail-closed reader.
This module holds the three things that are specific to the corpus: where the
file lives, what command rewrites it, and — the load-bearing one —
`embedded_text`, the single definition of the string that gets embedded.

`embedded_text` is here rather than inline in the pipeline because the cache and
the ingest have to agree on it byte for byte. If the pipeline joined prefix and
content with "\\n\\n" and the regeneration script used "\\n", every key would
miss and the ingest would refuse — which is at least loud. The dangerous
direction is subtler: any future edit that changes the join in one place and not
the other. One function, two callers, no second definition.

Only the contextual corpus is committed. `--no-contextual` embeds a different
string per chunk and therefore describes a different corpus — one no eval number
in this repository was measured on. It is available on `--source network`, and
refused on `--source fixture`, rather than doubling this file to hold vectors
for a corpus nobody scores.
"""

from __future__ import annotations

from pathlib import Path

from app.embedding_cache import (
    EmbeddingCache,
    cached_embeddings,
    embedding_key,
)
from app.embedding_cache import (
    load_cache as _load_cache,
)
from app.ingestion.chunker import UnitChunk
from app.ingestion.eurlex import FIXTURE_DIR

#: Committed alongside the text and the prefixes it was computed from, because
#: all three together are the corpus.
CACHE_PATH = FIXTURE_DIR / "chunk_embeddings.json"

#: Named in every failure message. One embedding call per chunk, so it costs
#: money and is deliberately not part of `make refresh-fixtures`.
REGENERATE_COMMAND = "make embedding-cache"


def embedded_text(prefix: str, content: str) -> str:
    """The exact string sent to the embedding endpoint for one chunk."""
    return f"{prefix}\n\n{content}"


def chunk_label(chunk: UnitChunk) -> str:
    """How a chunk is named in a cache-miss message."""
    return f"{chunk.ref} chunk {chunk.idx}"


def chunk_embedding_key(model: str, dimensions: int, prefix: str, chunk: UnitChunk) -> str:
    return embedding_key(model, dimensions, embedded_text(prefix, chunk.content))


def load_chunk_cache(path: Path | None = None) -> EmbeddingCache:
    return _load_cache(path or CACHE_PATH, REGENERATE_COMMAND)


def cached_chunk_vectors(
    regulation: str,
    chunks: list[UnitChunk],
    prefixes: list[str],
    cache: EmbeddingCache,
    model: str,
    dimensions: int,
) -> list[list[float]]:
    """The committed vector for every chunk, or raise naming the ones missing."""
    return cached_embeddings(
        scope=f"{regulation} chunk",
        labels=[chunk_label(chunk) for chunk in chunks],
        texts=[
            embedded_text(prefix, chunk.content)
            for prefix, chunk in zip(prefixes, chunks, strict=True)
        ],
        cache=cache,
        model=model,
        dimensions=dimensions,
        regenerate_command=REGENERATE_COMMAND,
    )
