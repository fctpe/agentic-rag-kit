"""Structure-aware chunking for statutory text.

Chunks never cross article boundaries — a chunk that mixes Article 5 with
Article 6 produces citations that point at the wrong law. Within an article,
numbered paragraphs are packed whole; only a single oversized paragraph is
ever split on sentence boundaries.
"""

import re
from dataclasses import dataclass

import tiktoken

from app.ingestion.eurlex import Article

_encoder = tiktoken.get_encoding("cl100k_base")

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.;:])\s+")


def count_tokens(text: str) -> int:
    return len(_encoder.encode(text))


@dataclass
class ArticleChunk:
    article_ref: str
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


def chunk_article(article: Article, max_tokens: int = 700) -> list[ArticleChunk]:
    pieces: list[str] = []
    for paragraph in article.paragraphs:
        if count_tokens(paragraph) > max_tokens:
            pieces.extend(_split_oversized(paragraph, max_tokens))
        else:
            pieces.append(paragraph)

    chunks: list[ArticleChunk] = []
    current: list[str] = []
    current_tokens = 0

    def flush() -> None:
        if not current:
            return
        content = "\n\n".join(current)
        chunks.append(
            ArticleChunk(
                article_ref=article.ref,
                heading=article.heading,
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
