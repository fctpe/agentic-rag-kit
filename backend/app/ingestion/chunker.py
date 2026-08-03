"""Structure-aware chunking for statutory text.

Chunks never cross unit boundaries — a chunk that mixes Article 5 with
Article 6 produces citations that point at the wrong law. Within a unit,
numbered paragraphs are packed whole; only a single oversized paragraph is
ever split on sentence boundaries.

Annexes go through the same packing as articles, deliberately. Annex III is
~19 chunks' worth of text, and the one thing it must not become is a single
blob: as one unit it outranks and crowds out the articles that cite it, which
is exactly the failure the article-container fix removed.
"""

import re
from dataclasses import dataclass

import tiktoken

from app.ingestion.eurlex import Unit

_encoder = tiktoken.get_encoding("cl100k_base")

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.;:])\s+")


def count_tokens(text: str) -> int:
    return len(_encoder.encode(text))


@dataclass
class UnitChunk:
    #: The ref of the unit this chunk came from — "Art. 6" or "Annex III".
    #: Persisted as `chunks.article_ref`, which predates annexes and keeps its
    #: name so existing rows, deep links and eval baselines stay readable.
    ref: str
    heading: str
    idx: int
    content: str
    token_count: int


def _split_oversized(paragraph: str, max_tokens: int) -> list[str]:
    sentences = _SENTENCE_BOUNDARY.split(paragraph)
    parts: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        if current and current_tokens + sentence_tokens > max_tokens:
            parts.append(" ".join(current))
            current = []
            current_tokens = 0
        current.append(sentence)
        current_tokens += sentence_tokens
    if current:
        parts.append(" ".join(current))
    return parts


def chunk_unit(unit: Unit, max_tokens: int = 700) -> list[UnitChunk]:
    pieces: list[str] = []
    for paragraph in unit.paragraphs:
        if count_tokens(paragraph) > max_tokens:
            pieces.extend(_split_oversized(paragraph, max_tokens))
        else:
            pieces.append(paragraph)

    chunks: list[UnitChunk] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> None:
        if not current:
            return
        content = "\n\n".join(current)
        chunks.append(
            UnitChunk(
                ref=unit.ref,
                heading=unit.heading,
                idx=len(chunks),
                content=content,
                token_count=count_tokens(content),
            )
        )

    for piece in pieces:
        piece_tokens = count_tokens(piece)
        if current and current_tokens + piece_tokens > max_tokens:
            flush()
            current = []
            current_tokens = 0
        current.append(piece)
        current_tokens += piece_tokens
    flush()

    return chunks
