"""Digest the ingested corpus — the text **and** the vectors.

    make corpus-digest          # print them
    make corpus-digest-check    # compare against the committed pair, exit 1 on drift

Prints one SHA-256 over `context_prefix`/`content` and one over
`chunks.embedding`, both taken in `(regulation, article_ref, idx)` order. Ingest
twice and compare: identical digests are what "the fixture ingest is
reproducible" means, and either digest alone is not enough to say it.

`--check` does the comparison for the reader, against
`data/fixtures/corpus_digest.json` — the pair every committed eval number was
measured on. "Reproduce it yourself and eyeball two hashes" is a reproduction
step nobody performs; a non-zero exit is one CI performs on every push.

That is not a hypothetical. The previous round committed the contextual
prefixes, verified the *text* digest twice, and shipped the claim that two
ingests produce byte-identical chunks. They did not: the text digest was stable
and the vector digest changed on every ingest, in a third to a half of the
rows, because the embedding endpoint does not return the same vector twice. The
column retrieval ranks on was the one nothing was hashing.

The vector is read as `embedding::text`, which is Postgres rendering the float4
values it actually stores. Re-encoding the column through the client would hash
a copy and could hide a cast.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from sqlalchemy import text

from app.db import dispose_engine, get_session_factory

CORPUS_SQL = text(
    """
SELECT d.regulation, c.article_ref, c.idx, c.context_prefix, c.content,
       c.embedding::text AS embedding_text
FROM chunks c
JOIN documents d ON d.id = c.document_id
ORDER BY d.regulation COLLATE "C", c.article_ref COLLATE "C", c.idx
"""
)


async def corpus_digests() -> tuple[int, str, str]:
    """(row count, text digest, vector digest) for the corpus in the database."""
    factory = get_session_factory()
    text_hash = hashlib.sha256()
    vector_hash = hashlib.sha256()
    rows = 0
    try:
        async with factory() as session:
            result = await session.execute(CORPUS_SQL)
            for row in result.mappings():
                rows += 1
                key = f"{row['regulation']}\x1f{row['article_ref']}\x1f{row['idx']}"
                # The key is hashed into both digests: two chunks swapping their
                # text would otherwise leave the concatenation unchanged.
                text_hash.update(
                    f"{key}\x1f{row['context_prefix']}\x00{row['content']}\x1e".encode()
                )
                vector = row["embedding_text"]
                if vector is None:
                    raise RuntimeError(
                        f"{key} has no embedding. A corpus with a null vector is not a "
                        f"corpus this digest can describe."
                    )
                vector_hash.update(f"{key}\x1f{vector}\x1e".encode())
    finally:
        await dispose_engine()
    if rows == 0:
        raise RuntimeError("No chunks in the database — run `make ingest-fixture` first.")
    return rows, text_hash.hexdigest(), vector_hash.hexdigest()


COMMITTED_DIGEST = (
    Path(__file__).resolve().parents[3] / "data" / "fixtures" / "corpus_digest.json"
)


def load_committed(path: Path = COMMITTED_DIGEST) -> dict:
    """The digest pair every committed eval number was measured against."""
    return json.loads(path.read_text(encoding="utf-8"))


def compare(
    measured: tuple[int, str, str], committed: dict | None = None
) -> list[str]:
    """Field-by-field mismatches between a measured corpus and the committed one.

    Returns an empty list when they agree. Every field is reported, not just the
    first: a chunk-count change and a vector change have different causes, and
    seeing one is not seeing the other.
    """
    expected = load_committed() if committed is None else committed
    rows, text_digest, vector_digest = measured
    fields = (("chunks", rows), ("text", text_digest), ("vector", vector_digest))
    return [
        f"{name}: measured {value!r}, committed {expected[name]!r}"
        for name, value in fields
        if value != expected[name]
    ]


async def _run(check: bool) -> int:
    measured = await corpus_digests()
    rows, text_digest, vector_digest = measured
    print(f"chunks: {rows}")
    print(f"text:   {text_digest}")
    print(f"vector: {vector_digest}")
    if not check:
        return 0
    mismatches = compare(measured)
    if mismatches:
        print(f"\nMISMATCH against {COMMITTED_DIGEST.name}:")
        for line in mismatches:
            print(f"  {line}")
        print(
            "\nThis corpus is not the one evals/results/ describes. Either the "
            "fixtures changed — in which case re-run the suites, re-promote, and "
            "update data/fixtures/corpus_digest.json — or the ingest is not "
            "reading the committed prefixes and vectors."
        )
        return 1
    print(f"\nmatches committed {COMMITTED_DIGEST.name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare against data/fixtures/corpus_digest.json and exit 1 on drift",
    )
    args = parser.parse_args(argv)
    return asyncio.run(_run(args.check))


if __name__ == "__main__":
    raise SystemExit(main())
