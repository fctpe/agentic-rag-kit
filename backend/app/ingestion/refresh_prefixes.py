"""Regenerate `data/fixtures/context_prefixes.json` — one model call per chunk.

**This costs money.** It calls the configured chat model once for every chunk of
the committed corpus (284 today), which is why it is a separate command and not
part of `make refresh-fixtures`: refreshing the text is free and should stay
free to run.

    make prefix-cache        # or: uv run python -m app.ingestion.refresh_prefixes

Run it after `make refresh-fixtures`, or after any change to `CONTEXT_PROMPT` or
to the chunker. Nothing here guesses when that is needed —
`tests/test_prefix_cache.py` fails when the committed cache stops covering the
committed fixture, and `--source fixture` refuses to ingest.

Every regulation is regenerated, always, and the file is written only once all
of them succeed. A partial cache would ingest as a partial corpus, and there is
no flag to write one: it would be the same "some chunks contextual, some not"
corpus that `cached_prefixes` refuses to assemble at ingest time.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date

from app.config import get_settings
from app.ingestion.chunker import UnitChunk, chunk_unit
from app.ingestion.contextual import llm_prefixes
from app.ingestion.eurlex import REGULATIONS, load_fixture
from app.ingestion.prefix_cache import CACHE_PATH, build_entry, write_cache


async def _generate(regulation: str, model_name: str) -> list[dict]:
    title = REGULATIONS[regulation]["title"]
    chunks: list[UnitChunk] = [
        chunk for unit in load_fixture(regulation) for chunk in chunk_unit(unit)
    ]
    print(f"[{regulation}] {len(chunks)} chunks -> {len(chunks)} model calls", flush=True)

    prefixes = await llm_prefixes(title, chunks, model_name)

    # An empty sentence is a failed call that returned successfully. Writing it
    # would commit a chunk whose "cached prefix" is nothing at all, and the
    # ingest would take it: the cache hit is by key, not by usefulness.
    blank = [
        chunk.ref for chunk, prefix in zip(chunks, prefixes, strict=True) if not prefix.strip()
    ]
    if blank:
        raise RuntimeError(
            f"[{regulation}] the model returned an empty prefix for {len(blank)} chunk(s), "
            f"first {blank[0]}. Refusing to commit a cache with holes in it."
        )

    return [
        build_entry(regulation, title, chunk, prefix)
        for chunk, prefix in zip(chunks, prefixes, strict=True)
    ]


async def _run(generated: str) -> int:
    settings = get_settings()
    entries: list[dict] = []
    for regulation in REGULATIONS:
        try:
            entries.extend(await _generate(regulation, settings.llm_model))
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 1

    path = write_cache(entries, model=settings.llm_model, generated=generated)
    print(f"wrote {path}: {len(entries)} prefixes")
    print("Re-ingest (`make ingest-fixture`) and re-run the eval suites: the embedded corpus")
    print("has changed, so every committed retrieval and RAGAS number now describes an old one.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generated",
        default=date.today().isoformat(),
        help="date stamped into the cache (default: today)",
    )
    args = parser.parse_args(argv)

    print(f"Regenerating {CACHE_PATH} — one model call per chunk. This costs money.", flush=True)
    return asyncio.run(_run(args.generated))


if __name__ == "__main__":
    raise SystemExit(main())
