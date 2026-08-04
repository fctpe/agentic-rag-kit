"""Committed query vectors for the golden set — the other half of a reproducible number.

A committed corpus does not make a retrieval number reproducible on its own,
because the *query* is embedded too, and that call is non-deterministic in the
same way the corpus one is. Measured over three independent passes across the
38 in-scope golden questions, compared at float64 before pgvector's float4 cast:
2, 3 and 4 questions of 38 differed between passes; the largest cosine distance
between two embeddings of the same question text was 6.21e-07, the largest
single-component delta 1.83e-04. Embedding the same string batched and singly
also differs, so batch shape is one of the inputs.

What that is worth, in the quantity that actually decides ranking — the
perturbation to the query-chunk distance |Δd(q,c)| over 38 questions x 284
chunks: mean 1.31e-06, p95 2.60e-06, max 7.88e-05. That is 22x smaller in the
mean and 19x smaller at the maximum than the corpus-side drift the committed
`chunk_embeddings.json` removes, and holding the corpus fixed while swapping
the query pass gave hit@6 1.0 / MRR 0.9189 / recall 0.9101 all three times,
with zero questions changing top-6 order.

So it is small, it is real, and "small" is not the standard `evals/thresholds.yaml`
sizes retrieval floors against. The floors are tight because a retrieval number
is supposed to be a function of the corpus and the question set; leaving one of
those two inputs free would keep that sentence false, just less false than
before. Both are pinned.

**The application still embeds live, as it must** — a user's question arrives as
text. This file pins the *eval*, and `--query-embeddings live` runs the eval the
way production runs, which is the check that pinning did not change what is
being measured.

    make embedding-cache      # regenerates this file and the corpus one together

Format, key and fail-closed behaviour are `app/embedding_cache.py`; the key
covers the embedding model and dimension count, so a model change misses every
entry rather than pairing a question with a vector from another space.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date
from pathlib import Path
from typing import Any

import yaml

# `app` resolves because everything that runs this file runs from `backend/`
# (`cd backend && uv run … ../evals/…`, and pytest from the same directory).
# `app.embedding_cache` imports nothing but the standard library, so this is
# cheap even for a caller that only wants CACHE_PATH.
from app.embedding_cache import EmbeddingCache, cached_embeddings, load_cache

EVALS_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = EVALS_DIR / "golden_questions.yaml"

#: Committed next to the questions it describes.
CACHE_PATH = EVALS_DIR / "query_embeddings.json"

#: Named in every failure message.
REGENERATE_COMMAND = "make embedding-cache"

NOTE = (
    "Embedding vectors for every question in golden_questions.yaml, generated once so that a "
    "committed retrieval number is a function of the corpus and the question set alone. The "
    "embedding endpoint returns a slightly different vector for the same string on different "
    "calls. Keyed by a SHA-256 over the embedding model, the dimension count and the exact "
    "question text, so an edited question or a changed model misses rather than reusing a "
    "vector written for other text. The application embeds live; this pins the eval. "
    f"Regenerate with `{REGENERATE_COMMAND}`."
)


def load_golden_questions() -> list[dict[str, Any]]:
    """Every question in the golden set, out-of-scope probes included.

    All of them, not just the ones the retrieval eval scores: `--questions`
    takes arbitrary ids, and a cache that covers "the ones we usually run" is a
    cache that fails closed on the day someone runs a different subset.
    """
    data = yaml.safe_load(GOLDEN_PATH.read_text())
    return list(data["questions"])


def load_query_cache(path: Path | None = None) -> EmbeddingCache:
    return load_cache(path or CACHE_PATH, REGENERATE_COMMAND)


def cached_query_vectors(
    ids: list[str],
    questions: list[str],
    cache: EmbeddingCache,
    model: str,
    dimensions: int,
) -> list[list[float]]:
    return cached_embeddings(
        scope="golden question",
        labels=ids,
        texts=questions,
        cache=cache,
        model=model,
        dimensions=dimensions,
        regenerate_command=REGENERATE_COMMAND,
    )


async def _run(generated: str) -> int:
    from app.config import get_settings
    from app.embedding_cache import build_entry, write_cache
    from app.embedding_cache import embedding_key as key_of
    from app.retrieval.embedder import embed_query

    settings = get_settings()
    model = settings.embedding_model
    dimensions = settings.embedding_dimensions

    questions = load_golden_questions()
    texts = [q["question"] for q in questions]
    print(f"[golden] {len(texts)} questions -> {len(texts)} embedding calls", flush=True)

    # One call per question, because that is what /chat does. `embed_texts`
    # would send them as a single batch, and the batch shape is one of the
    # inputs the provider's vector depends on: the same question embedded
    # batched versus alone differed in 32 of 38 cases here, an order of
    # magnitude further apart than two runs of the same shape. Pinning the
    # batched vector would pin a query the product never issues, so the eval
    # would be measuring a path nothing else takes.
    vectors = [await embed_query(text) for text in texts]
    for question, vector in zip(questions, vectors, strict=True):
        if len(vector) != dimensions or not any(vector):
            print(
                f"[golden] {question['id']} came back with {len(vector)} components"
                f"{' (all zero)' if not any(vector) else ''}, expected {dimensions} "
                f"non-zero. Refusing to commit a cache with holes in it.",
                file=sys.stderr,
            )
            return 1

    entries = [
        build_entry(key_of(model, dimensions, question["question"]), vector, id=question["id"])
        for question, vector in zip(questions, vectors, strict=True)
    ]
    path = write_cache(
        CACHE_PATH,
        entries,
        model=model,
        dimensions=dimensions,
        generated=generated,
        generated_by="evals.query_embeddings",
        note=NOTE,
    )
    print(f"wrote {path}: {len(entries)} vectors")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generated",
        default=date.today().isoformat(),
        help="date stamped into the cache (default: today)",
    )
    args = parser.parse_args(argv)
    print(f"Regenerating {CACHE_PATH} — one embedding call per question. This costs money.")
    return asyncio.run(_run(args.generated))


if __name__ == "__main__":
    raise SystemExit(main())
