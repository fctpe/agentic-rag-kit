"""Ingestion CLI: EUR-Lex -> articles -> chunks -> embeddings -> Postgres.

    uv run python -m app.ingestion.pipeline --regulations ai_act gdpr
    uv run python -m app.ingestion.pipeline --no-contextual --max-articles 10   # fast smoke

Re-running replaces a regulation's document and chunks atomically, so the
pipeline is idempotent.
"""

import argparse
import asyncio
import sys

from sqlalchemy import delete, select

from app.config import get_settings
from app.db import dispose_engine, get_session_factory
from app.ingestion.chunker import ArticleChunk, chunk_article
from app.ingestion.contextual import deterministic_prefix, llm_prefixes
from app.ingestion.eurlex import REGULATIONS, fetch_html, parse_articles
from app.models.tables import Chunk, Document
from app.retrieval.embedder import embed_texts


async def ingest_regulation(
    regulation: str,
    contextual: bool,
    max_articles: int | None,
) -> tuple[int, int]:
    meta = REGULATIONS[regulation]
    settings = get_settings()

    print(f"[{regulation}] fetching {meta['url']}", file=sys.stderr)
    html = fetch_html(meta["url"])
    articles = parse_articles(html)
    if max_articles:
        articles = articles[:max_articles]
    if not articles:
        raise RuntimeError(
            f"No articles parsed for {regulation} — EUR-Lex markup may have changed."
        )

    chunks: list[ArticleChunk] = []
    for article in articles:
        chunks.extend(chunk_article(article))
    print(f"[{regulation}] {len(articles)} articles -> {len(chunks)} chunks", file=sys.stderr)

    prefixes = [deterministic_prefix(meta["title"], chunk) for chunk in chunks]
    if contextual:
        print(f"[{regulation}] generating contextual prefixes…", file=sys.stderr)
        semantic = await llm_prefixes(meta["title"], chunks, settings.llm_model)
        prefixes = [f"{det} {sem}" for det, sem in zip(prefixes, semantic, strict=True)]

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
                article_ref=chunk.article_ref,
                heading=chunk.heading,
                idx=chunk.idx,
                content=chunk.content,
                context_prefix=prefix,
                embedding=vector,
            )
            for chunk, prefix, vector in zip(chunks, prefixes, vectors, strict=True)
        )
        await session.commit()

    return len(articles), len(chunks)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regulations", nargs="+", choices=list(REGULATIONS), default=list(REGULATIONS)
    )
    parser.add_argument(
        "--contextual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add an LLM-written context sentence per chunk (one model call per chunk)",
    )
    parser.add_argument("--max-articles", type=int, default=None)
    args = parser.parse_args()

    try:
        for regulation in args.regulations:
            articles, chunks = await ingest_regulation(
                regulation, args.contextual, args.max_articles
            )
            print(f"[{regulation}] done: {articles} articles, {chunks} chunks")
    finally:
        await dispose_engine()


if __name__ == "__main__":
    asyncio.run(main())
