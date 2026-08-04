"""Ingestion CLI: EUR-Lex (or the committed corpus) -> units -> chunks ->
embeddings -> Postgres.

A unit is one citable subdivision: an article ("Art. 6") or an annex
("Annex III"). Both are chunked the same way and land in the same table.

    uv run python -m app.ingestion.pipeline --regulations ai_act gdpr
    uv run python -m app.ingestion.pipeline --source fixture                 # offline
    uv run python -m app.ingestion.pipeline --no-contextual --max-units 10   # fast smoke

Re-running replaces a regulation's document and chunks atomically, so the
pipeline is idempotent. `--source fixture` is also *reproducible*: it reads its
contextual prefixes from `data/fixtures/context_prefixes.json` rather than
generating them, so two ingests of the committed corpus produce byte-identical
chunks. `--source network` still generates them, and does not.
"""

import argparse
import asyncio
import sys

from sqlalchemy import delete, select

from app.config import get_settings
from app.db import dispose_engine, get_session_factory
from app.ingestion.chunker import UnitChunk, chunk_unit
from app.ingestion.contextual import deterministic_prefix, llm_prefixes
from app.ingestion.eurlex import REGULATIONS, Annex, fetch_html, load_fixture, parse_units
from app.ingestion.prefix_cache import cached_prefixes, load_cache
from app.models.tables import Chunk, Document
from app.retrieval.embedder import embed_texts


async def resolve_prefixes(
    regulation: str,
    regulation_title: str,
    chunks: list[UnitChunk],
    contextual: bool,
    source: str,
    model_name: str,
) -> list[str]:
    """The full context prefix for each chunk, and where the semantic half comes from.

    The deterministic half is computed here on every run. The semantic sentence
    is *read* for `--source fixture` and *generated* for `--source network`, and
    that split is the whole point: the model call is not reproducible, so the
    committed corpus must not depend on one. A fixture ingest whose cache does
    not cover it fails closed rather than falling back to either the model or
    the deterministic prefix — see `app/ingestion/prefix_cache.py`.
    """
    prefixes = [deterministic_prefix(regulation_title, chunk) for chunk in chunks]
    if not contextual:
        return prefixes

    if source == "fixture":
        print(f"[{regulation}] reading committed context prefixes", file=sys.stderr)
        semantic = cached_prefixes(regulation, regulation_title, chunks, load_cache())
    else:
        # A fresh EUR-Lex parse has no committed prefixes by definition: the
        # text may differ from the fixture, so its cache keys would miss anyway.
        # This arm is therefore irreproducible on purpose, and a corpus ingested
        # through it should not be the one a committed eval number describes.
        print(f"[{regulation}] generating contextual prefixes…", file=sys.stderr)
        semantic = await llm_prefixes(regulation_title, chunks, model_name)

    return [f"{det} {sem}" for det, sem in zip(prefixes, semantic, strict=True)]


async def ingest_regulation(
    regulation: str,
    contextual: bool,
    max_units: int | None,
    source: str,
) -> tuple[int, int]:
    meta = REGULATIONS[regulation]
    settings = get_settings()

    if source == "fixture":
        print(f"[{regulation}] reading committed corpus", file=sys.stderr)
        units = load_fixture(regulation)
    else:
        print(f"[{regulation}] fetching {meta['fetch_url']}", file=sys.stderr)
        html = fetch_html(meta["fetch_url"])
        units = parse_units(html)
    if max_units:
        units = units[:max_units]
    if not units:
        raise RuntimeError(f"No units parsed for {regulation} — EUR-Lex markup may have changed.")

    chunks: list[UnitChunk] = []
    for unit in units:
        chunks.extend(chunk_unit(unit))
    # Annexes are counted separately because zero of them is a real answer for
    # one regulation (the GDPR has none) and a parser regression for the other.
    annexes = sum(1 for unit in units if isinstance(unit, Annex))
    print(
        f"[{regulation}] {len(units) - annexes} articles + {annexes} annexes "
        f"-> {len(chunks)} chunks",
        file=sys.stderr,
    )

    prefixes = await resolve_prefixes(
        regulation, meta["title"], chunks, contextual, source, settings.llm_model
    )

    print(f"[{regulation}] embedding {len(chunks)} chunks…", file=sys.stderr)
    vectors = await embed_texts(
        [f"{prefix}\n\n{chunk.content}" for prefix, chunk in zip(prefixes, chunks, strict=True)]
    )

    factory = get_session_factory()
    async with factory() as session:
        existing = await session.scalar(select(Document).where(Document.regulation == regulation))
        if existing:
            await session.execute(delete(Chunk).where(Chunk.document_id == existing.id))
            await session.delete(existing)
            await session.flush()

        document = Document(
            regulation=regulation,
            title=meta["title"],
            source_url=meta["url"],
        )
        session.add(document)
        await session.flush()

        session.add_all(
            Chunk(
                document_id=document.id,
                # Column name predates annexes; it holds any unit ref (ADR 0003).
                article_ref=chunk.ref,
                heading=chunk.heading,
                idx=chunk.idx,
                content=chunk.content,
                context_prefix=prefix,
                embedding=vector,
            )
            for chunk, prefix, vector in zip(chunks, prefixes, vectors, strict=True)
        )
        await session.commit()

    return len(units), len(chunks)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regulations", nargs="+", choices=list(REGULATIONS), default=list(REGULATIONS)
    )
    parser.add_argument(
        "--contextual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Add a context sentence per chunk. Read from data/fixtures/context_prefixes.json "
            "with --source fixture (free, reproducible, fails closed if it does not cover the "
            "corpus); generated with one model call per chunk with --source network"
        ),
    )
    parser.add_argument(
        "--max-units",
        type=int,
        default=None,
        help="Ingest only the first N units (articles come first) — for smoke runs",
    )
    parser.add_argument(
        "--source",
        choices=["network", "fixture"],
        default="network",
        help=(
            "network: fetch and parse live EUR-Lex (default). "
            "fixture: read the corpus committed under data/fixtures/"
        ),
    )
    args = parser.parse_args()

    try:
        for regulation in args.regulations:
            units, chunks = await ingest_regulation(
                regulation, args.contextual, args.max_units, args.source
            )
            print(f"[{regulation}] done: {units} units, {chunks} chunks")
    finally:
        await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
