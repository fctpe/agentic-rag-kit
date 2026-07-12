"""Retrieval tools bound to a DB session via factory.

Tools only retrieve — analysis and drafting stay in the agent LLM, so every
piece of regulation text the model sees flows through one audited path. The
collector accumulates every chunk retrieved during a run; citations shown to
the user come from it, never from the model's memory.
"""

from typing import Any

from langchain_core.tools import BaseTool, tool
from sqlalchemy.ext.asyncio import AsyncSession

from app.retrieval.hybrid import RetrievedChunk, hybrid_search, to_citation_dicts


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

    def citations(self) -> list[dict[str, Any]]:
        ordered = sorted(self._chunks.values(), key=lambda chunk: self._index[chunk.chunk_id])
        return to_citation_dicts(ordered)

    def source_excerpts(self, max_chars: int = 900) -> str:
        """Fuller excerpts for the grounding check — the 300-char UI snippet
        is too little context to judge faithfulness against."""
        ordered = sorted(self._chunks.values(), key=lambda chunk: self._index[chunk.chunk_id])
        return "\n".join(
            f"[{self._index[chunk.chunk_id]}] {chunk.article_ref}: {chunk.content[:max_chars]}"
            for chunk in ordered
        )


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

    return [search_corpus, compare_regulations]
