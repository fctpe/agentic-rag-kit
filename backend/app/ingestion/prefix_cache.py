"""Committed contextual prefixes for the fixture corpus.

`contextual.llm_prefixes` calls a model once per chunk at ingest time.
`temperature=0` is not determinism — there is no seed, and no provider
guarantees one — so two ingests of the *same* committed fixture produced
different prefixes, different embeddings and different retrieval numbers.
Measured on this corpus: hybrid MRR 0.888 on one ingest and 0.914 on the next,
same fixture, same code, same SQL. That is wider than the floors in
`evals/thresholds.yaml`, whose reasoning ("retrieval is deterministic… any real
movement is a real change") only holds if the corpus is fixed. It also broke
the repository's central promise: a reviewer could not reproduce a committed
retrieval number, because they could not reproduce the corpus.

So the model output is generated once and committed next to the text it
describes, in `data/fixtures/context_prefixes.json`. The fixture ingest path
reads it and **fails closed** on a miss (`pipeline.resolve_prefixes`); the live
path still generates, because a fresh EUR-Lex parse has no cache by definition.

Entries are keyed by the *whole model call*: a SHA-256 over the rendered prompt
plus the untruncated chunk content. Keying on `(ref, idx)` would let an edited
fixture silently reuse a sentence written for different text — the exact
staleness this file exists to make impossible. The prompt is in the key too, so
rewording `CONTEXT_PROMPT` invalidates every entry rather than pairing new
prompt semantics with old answers. Content is hashed past the 4,000-character
prompt truncation for the same reason.

Only the model's sentence is stored. The deterministic half of the prefix
(regulation, ref, heading) is computed from the chunk on every ingest and is
already reproducible.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.ingestion.chunker import UnitChunk
from app.ingestion.contextual import context_prompt
from app.ingestion.eurlex import FIXTURE_DIR

#: Committed alongside the corpus it describes, because it is part of the corpus:
#: the fixture text and these sentences are embedded together.
CACHE_PATH = FIXTURE_DIR / "context_prefixes.json"

#: Named in every failure message. One model call per chunk, so it costs money
#: and is deliberately not part of `make refresh-fixtures`.
REGENERATE_COMMAND = "make prefix-cache"

#: How many missing chunks a failure message names before it summarises. Enough
#: to identify a single edited unit; short enough that a wholesale invalidation
#: (a reworded prompt, a refreshed fixture) does not print 284 lines.
_MAX_NAMED_MISSES = 10


class PrefixCacheError(RuntimeError):
    """The committed prefixes do not cover the corpus being ingested."""


def chunk_key(regulation_title: str, chunk: UnitChunk) -> str:
    """SHA-256 identifying the model call this prefix was the answer to.

    Covers the rendered prompt (which carries the regulation title, the ref, the
    heading and the first 4,000 characters of content) and the full content, so
    any edit to the chunk text produces a different key and therefore a miss.
    """
    payload = "\x1f".join([context_prompt(regulation_title, chunk), chunk.content])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_entry(regulation: str, regulation_title: str, chunk: UnitChunk, prefix: str) -> dict:
    """One cache record. `ref`/`idx` are for humans reading the diff; `key` is the index."""
    return {
        "regulation": regulation,
        "ref": chunk.ref,
        "idx": chunk.idx,
        "key": chunk_key(regulation_title, chunk),
        "prefix": prefix,
    }


def load_cache(path: Path | None = None) -> dict[str, str]:
    """Read the committed cache as `key -> prefix`.

    A missing or malformed file raises rather than returning an empty mapping:
    an empty cache and a complete one differ only in whether the ingest is
    reproducible, and silently ingesting an unreproducible corpus is the failure
    being removed.
    """
    path = path or CACHE_PATH
    if not path.is_file():
        raise PrefixCacheError(
            f"No committed context prefixes at {path}. The fixture ingest reads its "
            f"prefixes from this file instead of calling the model, so that the same "
            f"fixture always produces the same corpus. Regenerate it with "
            f"`{REGENERATE_COMMAND}`, or ingest from live EUR-Lex "
            f"(`--source network`), which generates them."
        )

    try:
        document: Any = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise PrefixCacheError(f"{path} is not valid JSON: {exc}") from exc

    entries = document.get("entries") if isinstance(document, dict) else None
    if not isinstance(entries, list) or not entries:
        raise PrefixCacheError(
            f"{path} has no 'entries' list. Regenerate it with `{REGENERATE_COMMAND}`."
        )

    cache: dict[str, str] = {}
    for entry in entries:
        key = entry.get("key")
        prefix = entry.get("prefix")
        if not isinstance(key, str) or not isinstance(prefix, str) or not prefix.strip():
            raise PrefixCacheError(
                f"{path} contains an entry without a usable key/prefix pair: {entry!r}. "
                f"Regenerate it with `{REGENERATE_COMMAND}`."
            )
        cache[key] = prefix
    return cache


def cached_prefixes(
    regulation: str,
    regulation_title: str,
    chunks: list[UnitChunk],
    cache: dict[str, str],
) -> list[str]:
    """The committed sentence for every chunk, or raise naming the ones missing.

    There is deliberately no fallback. Calling the model on a miss reintroduces
    the non-determinism the cache removes, and falling back to the deterministic
    prefix would embed a corpus that is part contextual and part not — an
    unlabelled third corpus, and worse than one that refuses to ingest.
    """
    keys = [chunk_key(regulation_title, chunk) for chunk in chunks]
    missing = [(chunk, key) for chunk, key in zip(chunks, keys, strict=True) if key not in cache]
    if missing:
        named = ", ".join(
            f"{chunk.ref} chunk {chunk.idx} ({key[:12]}…)"
            for chunk, key in missing[:_MAX_NAMED_MISSES]
        )
        rest = len(missing) - _MAX_NAMED_MISSES
        if rest > 0:
            named += f", and {rest} more"
        raise PrefixCacheError(
            f"{len(missing)} of {len(chunks)} {regulation} chunks have no committed context "
            f"prefix in {CACHE_PATH.name}: {named}. Entries are keyed by chunk content, so a "
            "chunk whose text changed — or one a fixture refresh added — no longer matches "
            f"its entry and counts as missing. Regenerate with `{REGENERATE_COMMAND}` (one "
            "model call per chunk; it costs money, which is why it is not part of "
            "`make refresh-fixtures`). Refusing to generate the missing prefixes on the fly "
            "or to fall back to the deterministic prefix: either one makes this ingest "
            "irreproducible, which is the whole reason the cache exists."
        )
    return [cache[key] for key in keys]


def write_cache(entries: list[dict], model: str, generated: str, path: Path | None = None) -> Path:
    """Write the cache file. Callers assemble every entry first — see `refresh_prefixes`."""
    path = path or CACHE_PATH
    payload = {
        "generated": generated,
        "model": model,
        "generated_by": "app.ingestion.refresh_prefixes",
        "note": (
            "Contextual prefixes for the committed fixture corpus, generated once so that "
            "ingesting data/fixtures/ twice produces the same chunks. Keyed by a SHA-256 over "
            "the rendered prompt and the full chunk content; an edited chunk therefore misses "
            "rather than reusing a sentence written for different text. Regenerate with "
            f"`{REGENERATE_COMMAND}`."
        ),
        "entries": entries,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return path
