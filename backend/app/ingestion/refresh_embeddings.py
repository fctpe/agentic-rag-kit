"""Regenerate `data/fixtures/chunk_embeddings.json` — one embedding call per chunk.

**This costs money.** It embeds every chunk of the committed corpus (284 today)
with the configured embedding model, which is why it is a separate command and
not part of `make refresh-fixtures`: refreshing the text is free and stays free.

    make embedding-cache     # or: uv run python -m app.ingestion.refresh_embeddings

Run it after `make refresh-fixtures` **and after `make prefix-cache`**, in that
order. The embedded string is prefix + content, so a regenerated prefix changes
the string that is embedded; generating vectors first would commit vectors for
text that no longer exists. `tests/test_embedding_cache.py` fails offline the
moment the two disagree, and `--source fixture` refuses to ingest.

Every regulation is regenerated, always, and the file is written only once both
of them succeed. A half-written cache would ingest as a half-cached corpus, and
there is no flag to write one: `cached_chunk_vectors` refuses to assemble a
corpus whose vectors come partly from a file and partly from a fresh API call,
which is exactly the mixture this file exists to prevent.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date
from typing import Any

from app.config import get_settings
from app.embedding_cache import build_entry, write_cache
from app.ingestion.chunk_embeddings import (
    CACHE_PATH,
    REGENERATE_COMMAND,
    chunk_embedding_key,
    embedded_text,
)
from app.ingestion.chunker import UnitChunk, chunk_unit
from app.ingestion.contextual import deterministic_prefix
from app.ingestion.eurlex import REGULATIONS, load_fixture
from app.ingestion.prefix_cache import cached_prefixes
from app.ingestion.prefix_cache import load_cache as load_prefix_cache
from app.retrieval.embedder import embed_texts

NOTE = (
    "Embedding vectors for the committed fixture corpus, generated once so that ingesting "
    "data/fixtures/ twice produces the same vectors and not merely the same text. The "
    "embedding endpoint does not return bit-identical vectors for identical input — measured "
    "at 99-141 of 284 rows differing on every pair of ingests. Keyed by a SHA-256 over the "
    "embedding model, the dimension count and the exact embedded string (context prefix + "
    "content), so an edited chunk or a changed model misses rather than reusing a vector from "
    f"another text or another embedding space. Regenerate with `{REGENERATE_COMMAND}`."
)


async def _generate(regulation: str, model: str, dimensions: int) -> list[dict[str, Any]]:
    title = str(REGULATIONS[regulation]["title"])
    chunks: list[UnitChunk] = [
        chunk for unit in load_fixture(regulation) for chunk in chunk_unit(unit)
    ]

    # The committed corpus is the contextual one, so the committed vectors are
    # of the contextual text. Reading the prefixes rather than regenerating them
    # is what makes this script deterministic given its inputs: run it twice and
    # only the vectors move, never the strings they were computed from.
    semantic = cached_prefixes(regulation, title, chunks, load_prefix_cache())
    prefixes = [
        f"{deterministic_prefix(title, chunk)} {sentence}"
        for chunk, sentence in zip(chunks, semantic, strict=True)
    ]

    texts = [
        embedded_text(prefix, chunk.content) for prefix, chunk in zip(prefixes, chunks, strict=True)
    ]
    print(f"[{regulation}] {len(texts)} chunks -> {len(texts)} embedding calls", flush=True)

    vectors = await embed_texts(texts)

    # A vector of the wrong width, or of all zeros, is a failed call that
    # returned successfully. Committing it would give the chunk a cache "hit"
    # that ranks against nothing.
    for chunk, vector in zip(chunks, vectors, strict=True):
        if len(vector) != dimensions:
            raise RuntimeError(
                f"[{regulation}] {chunk.ref} chunk {chunk.idx} came back with {len(vector)} "
                f"components, expected {dimensions}. Refusing to commit a cache with a "
                f"malformed vector in it."
            )
        if not any(vector):
            raise RuntimeError(
                f"[{regulation}] {chunk.ref} chunk {chunk.idx} came back as an all-zero "
                f"vector. Refusing to commit a cache with holes in it."
            )

    return [
        build_entry(
            chunk_embedding_key(model, dimensions, prefix, chunk),
            vector,
            regulation=regulation,
            ref=chunk.ref,
            idx=chunk.idx,
        )
        for chunk, prefix, vector in zip(chunks, prefixes, vectors, strict=True)
    ]


async def _run(generated: str) -> int:
    settings = get_settings()
    entries: list[dict[str, Any]] = []
    for regulation in REGULATIONS:
        try:
            entries.extend(
                await _generate(regulation, settings.embedding_model, settings.embedding_dimensions)
            )
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 1

    path = write_cache(
        CACHE_PATH,
        entries,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        generated=generated,
        generated_by="app.ingestion.refresh_embeddings",
        note=NOTE,
    )
    print(f"wrote {path}: {len(entries)} vectors")
    print("Re-ingest (`make ingest-fixture`) and re-run the eval suites: the embedded corpus")
    print("has changed, so every committed retrieval and RAGAS number now describes an old one.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generated",
        default=date.today().isoformat(),
        help="date stamped into the cache (default: today)",
    )
    args = parser.parse_args(argv)

    print(
        f"Regenerating {CACHE_PATH} — one embedding call per chunk. This costs money.", flush=True
    )
    return asyncio.run(_run(args.generated))


if __name__ == "__main__":
    raise SystemExit(main())
