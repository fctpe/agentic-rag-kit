"""Hybrid retrieval: pgvector cosine + Postgres full-text, fused with RRF.

Legal queries split into two shapes — lexical ("Article 6(1)(f)") and
semantic ("can I process data without consent") — so both arms always run
and Reciprocal Rank Fusion (k=60) merges them without score calibration.
Both arms over-fetch (default 20) before fusion keeps the final k.

Every ordering here is **total**, and that is load-bearing rather than tidy.
RRF scores are sums of `1/(k + rank)` over small integers, so exact ties are not
a rare accident — they are arithmetic. Golden question A07 produced two chunks
with bit-identical fused scores (AI Act Art. 4 at vector rank 1 and Art. 66 at
text rank 1, `0x1.0c9714fbcda3bp-6` both), and with `ORDER BY f.score DESC`
alone the winner was chosen by the query plan: `final_k=2` returned Art. 4 as
rank 1, `final_k=3` returned Art. 66, `final_k=6` Art. 66, `final_k=8` Art. 4.
That is a production defect — the citation the user sees depends on how many
results were asked for — and it was worth 0.0132 of hybrid MRR across ingests,
the only committed retrieval number still moving after the corpus was pinned.

The tiebreak is `(regulation, article_ref, idx)`: unique per chunk (checked in
`tests/test_retrieval_total_order.py`) and derived from the fixture, so it is
the same on every ingest — unlike `chunks.id`, which is a fresh uuid4 each time
and would be a total order that still moved. `COLLATE "C"` because the default
collation is an environment property and a tiebreak that depends on the ICU
version is not reproducible across machines.

It costs nothing: the window functions already force a full sort of the
filtered set (verified with EXPLAIN), so the extra sort keys ride along.
"""

from dataclasses import dataclass
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.observability import (
    GEN_AI_OPERATION_NAME,
    RETRIEVAL_ARM_SIZE,
    RETRIEVAL_FINAL_K,
    RETRIEVAL_FROM_TEXT_ARM,
    RETRIEVAL_FROM_VECTOR_ARM,
    RETRIEVAL_REGULATION,
    RETRIEVAL_RETURNED,
    tracer,
)
from app.retrieval.embedder import embed_query

HYBRID_SQL = text(
    """
WITH params AS (
    SELECT CAST(:regulation AS text) AS regulation
),
vector_arm AS (
    SELECT c.id,
           row_number() OVER (
               ORDER BY c.embedding <=> CAST(:qvec AS vector),
                        d.regulation COLLATE "C", c.article_ref COLLATE "C", c.idx
           ) AS rank
    FROM chunks c
    JOIN documents d ON d.id = c.document_id, params p
    WHERE (p.regulation IS NULL OR d.regulation = p.regulation)
    ORDER BY c.embedding <=> CAST(:qvec AS vector),
             d.regulation COLLATE "C", c.article_ref COLLATE "C", c.idx
    LIMIT :arm_size
),
text_arm AS (
    -- Deliberately AND-semantics (plainto_tsquery): the arm stays silent on
    -- most natural-language questions and fires with high precision on
    -- lexical/citation queries ("data protection impact assessment").
    -- OR-semantics over stemmed lexemes was measured and rejected — generic
    -- legal vocabulary matches everywhere and the noise leaks through RRF
    -- (hybrid MRR fell 0.891 -> 0.772 on the golden set; see the README
    -- Results section for the committed hybrid/vector/text ablation numbers).
    --
    -- ts_rank_cd ties are common (many chunks share a rank of 0.1), so this
    -- arm needs the total order at least as much as fusion does.
    SELECT c.id,
           row_number() OVER (
               ORDER BY ts_rank_cd(c.tsv, query) DESC,
                        d.regulation COLLATE "C", c.article_ref COLLATE "C", c.idx
           ) AS rank
    FROM chunks c
    JOIN documents d ON d.id = c.document_id,
    plainto_tsquery('english', CAST(:query AS text)) query, params p
    WHERE c.tsv @@ query
      AND (p.regulation IS NULL OR d.regulation = p.regulation)
    ORDER BY ts_rank_cd(c.tsv, query) DESC,
             d.regulation COLLATE "C", c.article_ref COLLATE "C", c.idx
    LIMIT :arm_size
),
fused AS (
    SELECT COALESCE(v.id, t.id) AS id,
           COALESCE(1.0 / (:rrf_k + v.rank), 0) + COALESCE(1.0 / (:rrf_k + t.rank), 0) AS score,
           v.rank AS vector_rank,
           t.rank AS text_rank
    FROM vector_arm v
    FULL OUTER JOIN text_arm t USING (id)
)
SELECT c.id, c.article_ref, c.heading, c.idx, c.content,
       d.regulation, d.title AS document_title, d.source_url,
       f.score, f.vector_rank, f.text_rank
FROM fused f
JOIN chunks c ON c.id = f.id
JOIN documents d ON d.id = c.document_id
ORDER BY f.score DESC, d.regulation COLLATE "C", c.article_ref COLLATE "C", c.idx
LIMIT :final_k
"""
)


@dataclass
class RetrievedChunk:
    chunk_id: str
    regulation: str
    document_title: str
    source_url: str
    article_ref: str
    heading: str
    content: str
    score: float
    vector_rank: int | None
    text_rank: int | None


def _to_vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{value:.7f}" for value in vector) + "]"


async def hybrid_search(
    session: AsyncSession,
    query: str,
    regulation: str | None = None,
    final_k: int | None = None,
    query_vector: list[float] | None = None,
) -> list[RetrievedChunk]:
    """Retrieve for `query`. Pass `query_vector` to supply the embedding.

    The application never passes it — a user's question has to be embedded when
    it arrives. `evals/run_retrieval_eval.py` does, from the vectors committed
    under `evals/query_embeddings.json`, because the embedding endpoint returns
    a slightly different vector for the same string on different calls and a
    committed eval number should be a function of the corpus and the question
    set, not of which draw the API happened to return. See that file for the
    measurement of how large the difference is and what it moves.
    """
    settings = get_settings()
    with tracer.start_as_current_span(
        "retrieval hybrid",
        attributes={
            GEN_AI_OPERATION_NAME: "retrieval",
            RETRIEVAL_ARM_SIZE: settings.retrieval_arm_size,
            RETRIEVAL_FINAL_K: final_k or settings.retrieval_final_k,
            RETRIEVAL_REGULATION: regulation or "all",
        },
    ) as span:
        if query_vector is None:
            query_vector = await embed_query(query)

        # pgvector >= 0.8: relax the index scan so metadata filters cannot
        # silently empty the vector arm (the classic overfiltering failure).
        await session.execute(text("SET hnsw.iterative_scan = relaxed_order"))

        rows = await session.execute(
            HYBRID_SQL,
            {
                "qvec": _to_vector_literal(query_vector),
                "query": query,
                "regulation": regulation,
                "arm_size": settings.retrieval_arm_size,
                "rrf_k": settings.rrf_k,
                "final_k": final_k or settings.retrieval_final_k,
            },
        )
        results: list[RetrievedChunk] = []
        for row in rows.mappings():
            results.append(
                RetrievedChunk(
                    chunk_id=str(row["id"]),
                    regulation=row["regulation"],
                    document_title=row["document_title"],
                    source_url=row["source_url"],
                    article_ref=row["article_ref"],
                    heading=row["heading"],
                    content=row["content"],
                    score=float(row["score"]),
                    vector_rank=row["vector_rank"],
                    text_rank=row["text_rank"],
                )
            )
        # Per-arm contribution is only observable after fusion — the arms are
        # one statement. These are the numbers that say how often the
        # AND-semantics text arm stays silent.
        span.set_attribute(RETRIEVAL_RETURNED, len(results))
        span.set_attribute(
            RETRIEVAL_FROM_VECTOR_ARM, sum(1 for chunk in results if chunk.vector_rank is not None)
        )
        span.set_attribute(
            RETRIEVAL_FROM_TEXT_ARM, sum(1 for chunk in results if chunk.text_rank is not None)
        )
        return results


def to_citation_dicts(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    from app.retrieval.citations import citation_url

    return [
        {
            "index": index,
            "regulation": chunk.regulation,
            "document": chunk.document_title,
            "article": chunk.article_ref,
            "heading": chunk.heading,
            "url": citation_url(chunk.regulation, chunk.article_ref),
            "snippet": chunk.content[:300],
            "score": round(chunk.score, 5),
        }
        for index, chunk in enumerate(chunks, start=1)
    ]
