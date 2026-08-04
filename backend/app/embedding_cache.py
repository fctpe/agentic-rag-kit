"""Committed embedding vectors, for the two inputs a committed number depends on.

`prefix_cache` made the chunk *text* reproduce. It did not make the corpus
reproduce, because the column retrieval actually ranks on is `chunks.embedding`,
and the embedding endpoint does not return bit-identical vectors for identical
input. Measured over ten pairs of ingests of the byte-identical committed
fixture (chunk text digest identical in all five ingests, so the vectors were
the only variable): **99 to 141 of 284 rows differed bit-exactly on every pair**.
A differing row is not one rounded component — for one pair the 113 differing
rows differed in 933 to 1511 of their 1536 components. The magnitude is small
(largest single cosine deviation 1.07e-04, against a smallest real gap between
two distinct chunks of 3.73e-02), but "small" is not "reproducible", and
`evals/thresholds.yaml` sizes retrieval floors on reproducible.

So the vectors are generated once and committed, exactly as the prefixes were.

Storage format
--------------
One base64 string per vector, little-endian **float32**, inside a JSON envelope.

*float32 because that is what survives.* `chunks.embedding` is pgvector's
`vector` type, which is float4. Committing float64 would commit 8 bytes of which
Postgres keeps 4, and the round trip would no longer be exact — the committed
file has to hold the value the database will hold, or the guarantee is about the
wrong number.

*base64 rather than the two obvious alternatives.* A JSON array of decimal
floats is ~6 MB for the same data and no more reviewable: nobody reads 436,224
numbers, and a repr round trip through decimal is an extra way to lose a bit. A
side-car binary blob is smaller still, but it diffs as one opaque object, so a
regenerated cache tells a reviewer nothing about *what* changed. One base64
string per entry keeps the file a diffable list keyed by `ref`/`idx`: `git diff`
names the chunks whose vectors moved. The cost is 33% over raw — see
`size_note`, which every written file states about itself.

Keying
------
SHA-256 over `model \x1f dimensions \x1f text`, where `text` is the exact string
that was sent to the embedding endpoint (prefix and content, joined the way the
ingest joins them). Both halves matter:

- **text**, because a positional key would let an edited chunk reuse the vector
  of its previous text — stale, ranks cleanly, invisible in every count;
- **model and dimensions**, because `text-embedding-3-small` and
  `-3-large` embed the same string into different spaces, and the same model at
  1536 and at 512 dimensions likewise. A model change must invalidate the cache
  rather than silently pair new text with vectors from another space.

A miss fails closed. There is deliberately no fallback to the API: calling it is
the non-determinism being removed.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import sys
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: How many missing items a failure message names before it summarises. Enough
#: to identify a single edited chunk; short enough that a wholesale invalidation
#: (a new embedding model, a refreshed fixture) does not print 284 lines.
_MAX_NAMED_MISSES = 10

_FLOAT32_BYTES = 4

#: Written into every file and checked on load. Names the byte layout, so a file
#: produced by some other encoder cannot be read as if it were this one.
DTYPE = "float32-le-base64"


class EmbeddingCacheError(RuntimeError):
    """The committed vectors do not cover the texts being embedded."""


def embedding_key(model: str, dimensions: int, text: str) -> str:
    """SHA-256 identifying the embedding call this vector was the answer to."""
    payload = "\x1f".join([model, str(dimensions), text])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def encode_vector(vector: Sequence[float]) -> str:
    """float32 little-endian, base64. Lossy by design — see the module docstring."""
    packed = array("f", vector)
    if sys.byteorder == "big":
        packed.byteswap()
    return base64.b64encode(packed.tobytes()).decode("ascii")


def decode_vector(blob: str, dimensions: int) -> list[float]:
    try:
        raw = base64.b64decode(blob, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise EmbeddingCacheError(f"vector is not valid base64: {exc}") from exc
    if len(raw) != dimensions * _FLOAT32_BYTES:
        raise EmbeddingCacheError(
            f"vector is {len(raw)} bytes, expected {dimensions * _FLOAT32_BYTES} "
            f"({dimensions} x float32). The file's declared dimensions and its "
            f"vectors disagree, so one of them is from another run."
        )
    packed = array("f")
    packed.frombytes(raw)
    if sys.byteorder == "big":
        packed.byteswap()
    return packed.tolist()


@dataclass(frozen=True)
class EmbeddingCache:
    """A committed cache file, indexed by key."""

    path: Path
    model: str
    dimensions: int
    #: key -> base64 vector. Kept encoded; decoded on lookup.
    vectors: dict[str, str]

    def __len__(self) -> int:
        return len(self.vectors)

    def __contains__(self, key: object) -> bool:
        return key in self.vectors

    def get(self, key: str) -> list[float]:
        return decode_vector(self.vectors[key], self.dimensions)


def load_cache(path: Path, regenerate_command: str) -> EmbeddingCache:
    """Read a committed cache file, or raise naming the command that writes it.

    A missing or malformed file raises rather than returning an empty cache: an
    empty cache and a complete one differ only in whether the ingest is
    reproducible, and silently embedding an unreproducible corpus is the failure
    this module removes.
    """
    if not path.is_file():
        raise EmbeddingCacheError(
            f"No committed embeddings at {path}. This path reads its vectors from "
            f"this file instead of calling the embedding endpoint, because the "
            f"endpoint does not return the same vector twice for the same text. "
            f"Regenerate it with `{regenerate_command}`."
        )

    try:
        document: Any = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise EmbeddingCacheError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise EmbeddingCacheError(f"{path} is not an embedding cache object.")

    dtype = document.get("dtype")
    if dtype != DTYPE:
        raise EmbeddingCacheError(
            f"{path} declares dtype {dtype!r}, expected {DTYPE!r}. Refusing to read "
            f"vectors whose byte layout this decoder did not write. Regenerate with "
            f"`{regenerate_command}`."
        )

    model = document.get("model")
    dimensions = document.get("dimensions")
    if not isinstance(model, str) or not model or not isinstance(dimensions, int):
        raise EmbeddingCacheError(
            f"{path} does not state the model and dimensions its vectors were "
            f"generated with, so nothing can check they match the configured ones. "
            f"Regenerate with `{regenerate_command}`."
        )

    entries = document.get("entries")
    if not isinstance(entries, list) or not entries:
        raise EmbeddingCacheError(
            f"{path} has no 'entries' list. Regenerate it with `{regenerate_command}`."
        )

    vectors: dict[str, str] = {}
    for entry in entries:
        key = entry.get("key") if isinstance(entry, dict) else None
        blob = entry.get("vector") if isinstance(entry, dict) else None
        if not isinstance(key, str) or not isinstance(blob, str) or not blob:
            raise EmbeddingCacheError(
                f"{path} contains an entry without a usable key/vector pair: {entry!r}. "
                f"Regenerate it with `{regenerate_command}`."
            )
        # Decode eagerly: a truncated vector is a corrupt file, and finding that
        # out at ingest time — after the API-free promise has been made — is
        # worse than finding it out on load.
        decode_vector(blob, dimensions)
        vectors[key] = blob

    return EmbeddingCache(path=path, model=model, dimensions=dimensions, vectors=vectors)


def cached_embeddings(
    scope: str,
    labels: Sequence[str],
    texts: Sequence[str],
    cache: EmbeddingCache,
    model: str,
    dimensions: int,
    regenerate_command: str,
) -> list[list[float]]:
    """The committed vector for every text, or raise naming what is missing.

    `model`/`dimensions` are the *configured* ones, not the file's: the key is
    built from them, so a model change misses every entry instead of pairing
    text with vectors from another embedding space. The message says so when
    that is what happened, because "284 of 284 missing" on its own reads like a
    corrupt file.
    """
    keys = [embedding_key(model, dimensions, text) for text in texts]
    missing = [label for label, key in zip(labels, keys, strict=True) if key not in cache]
    if not missing:
        return [cache.get(key) for key in keys]

    if (model, dimensions) != (cache.model, cache.dimensions):
        reason = (
            f"The committed vectors were generated with {cache.model} at "
            f"{cache.dimensions} dimensions; this run is configured for {model} at "
            f"{dimensions}. Those are different embedding spaces, so every key "
            f"misses — which is the intended behaviour, not a corrupt file. "
        )
    else:
        reason = (
            "Entries are keyed by the exact embedded text, so a chunk whose text "
            "changed — or one a fixture refresh added — no longer matches its entry "
            "and counts as missing. "
        )

    named = ", ".join(missing[:_MAX_NAMED_MISSES])
    rest = len(missing) - _MAX_NAMED_MISSES
    if rest > 0:
        named += f", and {rest} more"
    raise EmbeddingCacheError(
        f"{len(missing)} of {len(texts)} {scope} texts have no committed embedding in "
        f"{cache.path.name}: {named}. {reason}"
        f"Regenerate with `{regenerate_command}` (one embedding call per text; it costs "
        f"money, which is why it is not part of `make refresh-fixtures`). Refusing to "
        f"embed the missing texts on the fly: the endpoint returns a different vector "
        f"for the same text on different calls, which is the whole reason this file "
        f"exists."
    )


def build_entry(key: str, vector: Sequence[float], **provenance: Any) -> dict[str, Any]:
    """One cache record. `provenance` fields are for humans reading the diff."""
    return {**provenance, "key": key, "vector": encode_vector(vector)}


def write_cache(
    path: Path,
    entries: list[dict[str, Any]],
    model: str,
    dimensions: int,
    generated: str,
    generated_by: str,
    note: str,
) -> Path:
    """Write a cache file. Callers assemble every entry first — all or nothing."""
    raw_bytes = len(entries) * dimensions * _FLOAT32_BYTES
    payload = {
        "generated": generated,
        "model": model,
        "dimensions": dimensions,
        "dtype": DTYPE,
        "generated_by": generated_by,
        "vectors": len(entries),
        # The file states its own size, because the reason this format was
        # chosen over a JSON array of floats is a size trade-off, and a trade-off
        # nobody can see the numbers of gets re-litigated on vibes.
        "size_note": (
            f"{len(entries)} vectors x {dimensions} float32 = "
            f"{raw_bytes / 1_048_576:.2f} MiB raw, "
            f"{raw_bytes * 4 / 3 / 1_048_576:.2f} MiB as base64 text"
        ),
        "note": note,
        "entries": entries,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return path
