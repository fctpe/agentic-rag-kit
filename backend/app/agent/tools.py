"""Retrieval tools bound to a DB session via factory.

Tools only retrieve — analysis and drafting stay in the agent LLM, so every
piece of regulation text the model sees flows through one audited path. The
collector accumulates every chunk retrieved during a run; citations shown to
the user come from it, never from the model's memory.
"""

from typing import Any

from langchain_core.tools import BaseTool, tool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tables import Chunk, Document
from app.retrieval.citations import citation_url
from app.retrieval.hybrid import RetrievedChunk, hybrid_search

_CITATION_KEYS = (
    "index",
    "regulation",
    "document",
    "article",
    "heading",
    "url",
    "snippet",
    "score",
)


def sources_to_citations(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """UI citation dicts (no full content) from persisted retrieved_sources."""
    return [{key: source[key] for key in _CITATION_KEYS} for source in sources]


def sources_to_excerpts(sources: list[dict[str, Any]], max_chars: int = 4000) -> str:
    """Full-content excerpts for the grounding check. Chunks are bounded at
    ~700 tokens, so the window covers the whole chunk — judging against a
    truncated excerpt flags correct claims (list items past the cutoff) as
    unsupported."""
    return "\n".join(
        f"[{source['index']}] {source['article']}: {source['content'][:max_chars]}"
        for source in sources
    )


class CitationCollector:
    """Assigns stable 1-based source ids across all tool calls in one run."""

    def __init__(self) -> None:
        self._chunks: dict[str, RetrievedChunk] = {}
        self._index: dict[str, int] = {}

    def _register(self, chunk: RetrievedChunk) -> int:
        if chunk.chunk_id not in self._index:
            self._index[chunk.chunk_id] = len(self._index) + 1
            self._chunks[chunk.chunk_id] = chunk
        return self._index[chunk.chunk_id]

    def numbered(self, chunks: list[RetrievedChunk]) -> str:
        blocks = []
        for chunk in chunks:
            index = self._register(chunk)
            blocks.append(
                f'<source id="{index}" regulation="{chunk.regulation}" ref="{chunk.article_ref}">\n'
                f"{chunk.content}\n</source>"
            )
        return "\n\n".join(blocks)

    def export_sources(self) -> list[dict[str, Any]]:
        """JSON-serializable snapshot of everything retrieved so far, written
        into graph state so it survives checkpointing (and a resumed
        approval, which never re-runs the tools node)."""
        ordered = sorted(self._chunks.values(), key=lambda chunk: self._index[chunk.chunk_id])
        return [
            {
                "index": self._index[chunk.chunk_id],
                "regulation": chunk.regulation,
                "document": chunk.document_title,
                "article": chunk.article_ref,
                "heading": chunk.heading,
                "url": citation_url(chunk.regulation, chunk.article_ref),
                "snippet": chunk.content[:300],
                "content": chunk.content,
                "score": round(chunk.score, 5),
            }
            for chunk in ordered
        ]


def build_tools(session: AsyncSession, collector: CitationCollector) -> list[BaseTool]:
    @tool
    async def search_corpus(query: str, regulation: str | None = None) -> str:
        """Search the EU AI Act and GDPR corpus. Optionally restrict to one
        regulation: "ai_act" or "gdpr". Returns numbered source excerpts —
        cite them as [id] in your answer."""
        if regulation not in (None, "ai_act", "gdpr"):
            return 'Invalid regulation filter — use "ai_act", "gdpr", or omit it.'
        chunks = await hybrid_search(session, query, regulation=regulation)
        if not chunks:
            return "No matching provisions found. Try different terms or drop the filter."
        return collector.numbered(chunks)

    @tool
    async def read_article(regulation: str, article_number: str) -> str:
        """Read ONE article in full (all its chunks, in order). Use after
        search_corpus identifies the controlling article, especially before
        enumerating list-type content (prohibitions, obligations, rights) —
        top-k search may return only part of a long article. regulation is
        "ai_act" or "gdpr"; article_number like "5" or "22"."""
        if regulation not in ("ai_act", "gdpr"):
            return 'Invalid regulation — use "ai_act" or "gdpr".'
        rows = await session.execute(
            select(Chunk, Document)
            .join(Document, Document.id == Chunk.document_id)
            .where(Document.regulation == regulation)
            .where(Chunk.article_ref == f"Art. {article_number.strip()}")
            .order_by(Chunk.idx)
        )
        pairs = rows.all()
        if not pairs:
            return f"No Art. {article_number} found in {regulation}."
        chunks = [
            RetrievedChunk(
                chunk_id=str(chunk.id),
                regulation=document.regulation,
                document_title=document.title,
                source_url=document.source_url,
                article_ref=chunk.article_ref,
                heading=chunk.heading,
                content=chunk.content,
                score=1.0,
                vector_rank=None,
                text_rank=None,
            )
            for chunk, document in pairs
        ]
        return collector.numbered(chunks)

    @tool
    async def compare_regulations(topic: str) -> str:
        """Retrieve what the AI Act AND the GDPR each say about a topic, side
        by side. Use for overlap/conflict questions (e.g. automated
        decision-making, data governance, transparency duties)."""
        ai_act = await hybrid_search(session, topic, regulation="ai_act", final_k=4)
        gdpr = await hybrid_search(session, topic, regulation="gdpr", final_k=4)
        if not ai_act and not gdpr:
            return "Neither regulation returned matches for this topic."
        parts = []
        if ai_act:
            parts.append(f"AI ACT sources:\n{collector.numbered(ai_act)}")
        if gdpr:
            parts.append(f"GDPR sources:\n{collector.numbered(gdpr)}")
        return "\n\n".join(parts)

    return [search_corpus, read_article, compare_regulations]
