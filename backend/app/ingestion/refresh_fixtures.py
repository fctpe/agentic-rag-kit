"""Regenerate the committed corpus under `data/fixtures/` from live EUR-Lex.

The fixtures are what CI and the scored eval workflow ingest, so they are the
corpus every committed number was measured against. Until this module existed
they could only be produced by a throwaway script, which meant the corpus in the
repository was not reproducible from the repository — a reviewer could read the
parser and read the fixture and had no way to check that one had produced the
other.

Run it after any change to the parser:

    make refresh-fixtures        # or: uv run python -m app.ingestion.refresh_fixtures

It writes only after both regulations parse. A partial refresh would leave the
AI Act on a new parser and the GDPR on an old one, and every corpus-wide count
would then describe a corpus that never existed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date

from app.ingestion.chunker import chunk_unit
from app.ingestion.eurlex import (
    FIXTURE_DIR,
    REGULATIONS,
    Annex,
    Article,
    FetchError,
    ParseError,
    Unit,
    fetch_html,
    parse_units,
)


def _document(regulation: str, units: list[Unit], retrieved: str) -> dict:
    meta = REGULATIONS[regulation]

    def entry(unit: Unit) -> dict:
        return {
            "number": unit.number,
            "heading": unit.heading,
            "paragraphs": list(unit.paragraphs),
        }

    return {
        "regulation": regulation,
        "title": meta["title"],
        "celex": meta["celex"],
        # Where a reader verifies the text, not where it was fetched from.
        # Both are recorded: a fixture that names only one of them cannot be
        # audited, and they are not interchangeable.
        "source_url": meta["url"],
        "fetched_from": meta["fetch_url"],
        "retrieved": retrieved,
        "parsed_by": "app.ingestion.eurlex.parse_units",
        "articles": [entry(u) for u in units if isinstance(u, Article)],
        # Written even when empty. `load_fixture` rejects a file with no
        # "annexes" key, because a missing key means the file predates annex
        # ingestion, which is not the same fact as a regulation having none.
        "annexes": [entry(u) for u in units if isinstance(u, Annex)],
    }


def _fetch_with_retries(url: str, attempts: int) -> str:
    """Fetch, retrying only the transient refusal EUR-Lex actually produces.

    EUR-Lex answers 202 with an empty body under bot protection, and it clears
    within a minute or two — `FetchError` is raised for exactly that shape and
    for nothing else, so retrying it is not a blanket swallow of failures. An
    HTTP error still propagates on the first attempt, and running out of
    attempts raises the last `FetchError` rather than returning a short document.
    """
    delay = 20.0
    for attempt in range(1, attempts + 1):
        try:
            return fetch_html(url, timeout=180.0)
        except FetchError:
            if attempt == attempts:
                raise
            print(
                f"  attempt {attempt}/{attempts} refused (202, empty body); "
                f"retrying in {delay:.0f}s",
                flush=True,
            )
            time.sleep(delay)
            delay *= 1.5
    raise AssertionError("unreachable")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrieved",
        default=date.today().isoformat(),
        help="date stamped into the fixture (default: today)",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=6,
        help="fetch attempts per regulation before giving up (default: 6)",
    )
    args = parser.parse_args(argv)

    parsed: dict[str, list[Unit]] = {}
    for regulation, meta in REGULATIONS.items():
        print(f"[{regulation}] fetching {meta['fetch_url']}", flush=True)
        try:
            html = _fetch_with_retries(meta["fetch_url"], args.attempts)
        except FetchError as exc:
            # A fetch failure is not a parser failure, and the distinction is
            # the whole reason FetchError exists. Say which one happened.
            print(f"[{regulation}] {exc}", file=sys.stderr)
            return 2
        try:
            units = parse_units(html)
        except ParseError as exc:
            print(f"[{regulation}] parse failed: {exc}", file=sys.stderr)
            return 1
        parsed[regulation] = units

        articles = sum(1 for u in units if isinstance(u, Article))
        annexes = sum(1 for u in units if isinstance(u, Annex))
        chunks = sum(len(chunk_unit(u)) for u in units)
        print(f"[{regulation}] {articles} articles + {annexes} annexes -> {chunks} chunks")

    # Both parsed, so neither file can be left describing a different parser
    # than the other.
    for regulation, units in parsed.items():
        path = FIXTURE_DIR / f"{regulation}.json"
        payload = _document(regulation, units, args.retrieved)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(f"wrote {path}")

    print("\nRe-run the eval suites: the corpus these fixtures feed has changed.")
    print(
        "The committed context prefixes are keyed by chunk content, so every chunk whose text "
        "moved now misses. `make ingest-fixture` will refuse until you run `make prefix-cache` "
        "(one model call per chunk — that is why it is not run from here)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
