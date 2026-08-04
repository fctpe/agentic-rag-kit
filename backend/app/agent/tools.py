"""Retrieval tools bound to a DB session via factory.

Tools only retrieve — analysis and drafting stay in the agent LLM, so every
piece of regulation text the model sees flows through one audited path. The
collector accumulates every chunk retrieved during a run; citations shown to
the user come from it, never from the model's memory.
"""

from collections.abc import Sequence
from typing import Any

from langchain_core.tools import BaseTool, tool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ingestion.eurlex import unit_ref
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


def _source_dict(chunk: RetrievedChunk, index: int) -> dict[str, Any]:
    return {
        "index": index,
        # Not in _CITATION_KEYS, so it never reaches the UI payload: it exists
        # so a later turn on the same thread can recognise a chunk it has
        # already numbered instead of numbering it twice.
        "chunk_id": chunk.chunk_id,
        "regulation": chunk.regulation,
        "document": chunk.document_title,
        "article": chunk.article_ref,
        "heading": chunk.heading,
        "url": citation_url(chunk.regulation, chunk.article_ref),
        "snippet": chunk.content[:300],
        "content": chunk.content,
        "score": round(chunk.score, 5),
    }


class CitationCollector:
    """Assigns 1-based source ids that are stable for the whole THREAD.

    Per-run numbering was the older behaviour and it was wrong in a way nothing
    could see: the collector was rebuilt on every HTTP request, so turn 2 handed
    out [1] again, while the model still had turn 1's answer — with turn 1's [1]
    — in its message history. Ask it to summarise both answers and it reuses the
    earlier markers against the later source list, and every one of them renders
    as a working button pointing at a different regulation. It failed silently,
    with grounded=true, because the grounding prompt is explicitly told to ignore
    numbering.

    Seeding from the checkpointed `retrieved_sources` fixes it at the source of
    the numbers: an id, once handed out on a thread, means that chunk forever.
    The channel then grows monotonically across a thread, which is the intended
    trade — sources_to_excerpts already bounds what the verifier sees.
    """

    def __init__(self, seed: Sequence[dict[str, Any]] | None = None) -> None:
        self._sources: dict[str, dict[str, Any]] = {}
        self._index: dict[str, int] = {}
        self._next_index = 1
        self.seed(seed or [])

    def seed(self, sources: Sequence[dict[str, Any]]) -> None:
        """Adopt the ids a previous turn on this thread already handed out."""
        for source in sources:
            index = source.get("index")
            if not isinstance(index, int):
                continue
            # Checkpoints written before chunk_id was exported still have to
            # keep their numbers: key them by index so the id is occupied and
            # can never be re-issued to a different chunk. They simply never
            # de-duplicate against a re-retrieval, which costs a duplicate card,
            # not a wrong link.
            key = str(source.get("chunk_id") or f"index:{index}")
            self._index[key] = index
            self._sources[key] = dict(source)
        self._next_index = max(self._index.values(), default=0) + 1

    def _register(self, chunk: RetrievedChunk) -> int:
        if chunk.chunk_id not in self._index:
            index = self._next_index
            self._next_index += 1
            self._index[chunk.chunk_id] = index
            self._sources[chunk.chunk_id] = _source_dict(chunk, index)
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
        """JSON-serializable snapshot of everything retrieved on this thread so
        far, written into graph state so it survives checkpointing (and a
        resumed approval, which never re-runs the tools node)."""
        return sorted(self._sources.values(), key=lambda source: source["index"])


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
        """Read ONE article or annex in full (all its chunks, in order). Use
        after search_corpus identifies the controlling text, especially before
        enumerating list-type content (prohibitions, obligations, rights, the
        high-risk use cases in Annex III) — top-k search may return only part
        of a long article or annex. regulation is "ai_act" or "gdpr";
        article_number like "5" or "22" for an article, "Annex III" for an
        annex."""
        if regulation not in ("ai_act", "gdpr"):
            return 'Invalid regulation — use "ai_act" or "gdpr".'
        # Annexes are the enumerations most likely to be asked for whole
        # ("which use cases are high-risk?"), and search returns fragments of
        # them exactly as it does for a long article.
        ref = unit_ref(article_number)
        rows = await session.execute(
            select(Chunk, Document)
            .join(Document, Document.id == Chunk.document_id)
            .where(Document.regulation == regulation)
            .where(Chunk.article_ref == ref)
            .order_by(Chunk.idx)
        )
        pairs = rows.all()
        if not pairs:
            return f"No {ref} found in {regulation}."
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
