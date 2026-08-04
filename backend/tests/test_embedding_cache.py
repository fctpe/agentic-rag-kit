"""The committed corpus has to reproduce — the vectors, not only the text.

The previous round committed the contextual prefixes, checked a SHA-256 over
`chunks.context_prefix` twice, and published the claim that two ingests of the
committed fixture produce byte-identical chunks. The check was true and the
claim was false: `chunks.embedding` — the column the vector arm actually ranks
on — differed in 99 to 141 of 284 rows on every pair of ingests, because the
embedding endpoint does not return bit-identical vectors for identical input.
Nothing in `backend/tests/` looked at that column at all.

Measured here on this corpus while writing these tests: embedding the same 284
strings through the API and comparing against the committed vectors at float32
gives **141 of 284 rows different**, largest single-component delta 3.235e-03.
That is the number the cache removes.

So the vectors are generated once and committed, and these tests hold the four
properties that makes true, all offline and free:

1. the committed cache covers the committed corpus exactly, in both directions
   — no chunk without a vector, no vector without a chunk;
2. a miss fails closed, naming the chunks and the regeneration command, and does
   not fall back to the endpoint;
3. a changed embedding model or dimension count invalidates every entry, rather
   than pairing this corpus with vectors from another embedding space;
4. `--source fixture` never reaches the embedding API.

Every one is paired with the negative control that proves the assertion can
fail. That pairing is not ceremony here: the check this file replaces passed
for two ingests while the property it named was false.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import get_settings
from app.embedding_cache import (
    DTYPE,
    EmbeddingCacheError,
    build_entry,
    decode_vector,
    embedding_key,
    encode_vector,
    load_cache,
    write_cache,
)
from app.ingestion.chunk_embeddings import (
    CACHE_PATH,
    cached_chunk_vectors,
    chunk_embedding_key,
    embedded_text,
    load_chunk_cache,
)
from app.ingestion.chunker import UnitChunk, chunk_unit
from app.ingestion.contextual import deterministic_prefix
from app.ingestion.eurlex import REGULATIONS, load_fixture
from app.ingestion.pipeline import resolve_embeddings
from app.ingestion.prefix_cache import cached_prefixes
from app.ingestion.prefix_cache import load_cache as load_prefix_cache

#: Kept in step with tests/test_prefix_cache.py::EXPECTED_CHUNKS and
#: tests/test_corpus_fixture.py::EXPECTED, which pin the same corpus from the
#: prefix and the text side. All three move together or one of them fails.
EXPECTED_CHUNKS = {"ai_act": 167, "gdpr": 117}

MODEL = get_settings().embedding_model
DIMENSIONS = get_settings().embedding_dimensions


def title(regulation: str) -> str:
    return str(REGULATIONS[regulation]["title"])


def corpus_chunks(regulation: str) -> list[UnitChunk]:
    return [chunk for unit in load_fixture(regulation) for chunk in chunk_unit(unit)]


def corpus_prefixes(regulation: str, chunks: list[UnitChunk]) -> list[str]:
    """The full prefixes a fixture ingest would build — deterministic + committed."""
    semantic = cached_prefixes(regulation, title(regulation), chunks, load_prefix_cache())
    return [
        f"{deterministic_prefix(title(regulation), chunk)} {sentence}"
        for chunk, sentence in zip(chunks, semantic, strict=True)
    ]


@pytest.fixture(scope="module")
def committed_cache():
    return load_chunk_cache()


# --- 1. coverage, both directions -------------------------------------------


@pytest.mark.parametrize("regulation", sorted(EXPECTED_CHUNKS))
def test_committed_vectors_cover_every_chunk_of_the_committed_corpus(
    regulation: str, committed_cache
) -> None:
    """The test that catches a fixture or a prefix cache regenerated without vectors.

    `make refresh-fixtures` is free, `make prefix-cache` and `make
    embedding-cache` are not, so the three will drift. This is where that
    surfaces — on every push, offline — rather than in an ingest someone runs
    against a database.
    """
    chunks = corpus_chunks(regulation)
    assert len(chunks) == EXPECTED_CHUNKS[regulation]

    vectors = cached_chunk_vectors(
        regulation, chunks, corpus_prefixes(regulation, chunks), committed_cache, MODEL, DIMENSIONS
    )

    assert len(vectors) == len(chunks)
    for vector in vectors:
        assert len(vector) == DIMENSIONS
        # An all-zero vector is a failed call that returned successfully: it
        # would be a cache hit that ranks against nothing.
        assert any(vector)


def test_the_cache_holds_the_corpus_and_nothing_else(committed_cache) -> None:
    # Coverage alone passes on a cache carrying vectors for chunks that no
    # longer exist — the residue of a fixture that shrank, or of a prefix
    # rewrite that left the old strings behind. Set equality catches that
    # direction too, and it is free.
    corpus_keys = set()
    for regulation in EXPECTED_CHUNKS:
        chunks = corpus_chunks(regulation)
        for chunk, prefix in zip(chunks, corpus_prefixes(regulation, chunks), strict=True):
            corpus_keys.add(chunk_embedding_key(MODEL, DIMENSIONS, prefix, chunk))

    assert len(corpus_keys) == sum(EXPECTED_CHUNKS.values()) == 284
    assert set(committed_cache.vectors) == corpus_keys

    # Negative control: the assertion above is not vacuous — one extra entry
    # breaks it, so a cache with residue would be caught.
    with_residue = dict(committed_cache.vectors)
    with_residue["0" * 64] = next(iter(committed_cache.vectors.values()))
    assert set(with_residue) != corpus_keys


def test_the_cache_and_the_prefix_cache_describe_the_same_corpus(committed_cache) -> None:
    """The two committed artefacts are keyed off each other, in that order.

    The embedded string is prefix + content, so regenerating the prefixes
    changes every embedding key. Running `make embedding-cache` before `make
    prefix-cache` would commit vectors for text that no longer exists — this is
    the assertion that catches it.
    """
    for regulation in EXPECTED_CHUNKS:
        chunks = corpus_chunks(regulation)
        prefixes = corpus_prefixes(regulation, chunks)
        assert all(
            chunk_embedding_key(MODEL, DIMENSIONS, prefix, chunk) in committed_cache
            for chunk, prefix in zip(chunks, prefixes, strict=True)
        )

        # Negative control: the same chunks with the *deterministic* prefix only
        # — the string a `--no-contextual` ingest would embed — miss every
        # entry, which is why that flag is refused on --source fixture.
        bare = [deterministic_prefix(title(regulation), chunk) for chunk in chunks]
        assert not any(
            chunk_embedding_key(MODEL, DIMENSIONS, prefix, chunk) in committed_cache
            for chunk, prefix in zip(chunks, bare, strict=True)
        )


# --- 2. a miss fails closed --------------------------------------------------


def test_a_missing_entry_fails_closed_naming_the_chunks_and_the_command(committed_cache) -> None:
    chunks = corpus_chunks("gdpr")
    prefixes = corpus_prefixes("gdpr", chunks)
    dropped_key = chunk_embedding_key(MODEL, DIMENSIONS, prefixes[7], chunks[7])
    incomplete = type(committed_cache)(
        path=committed_cache.path,
        model=committed_cache.model,
        dimensions=committed_cache.dimensions,
        vectors={k: v for k, v in committed_cache.vectors.items() if k != dropped_key},
    )
    assert len(incomplete) == len(committed_cache) - 1

    with pytest.raises(EmbeddingCacheError) as excinfo:
        cached_chunk_vectors("gdpr", chunks, prefixes, incomplete, MODEL, DIMENSIONS)

    message = str(excinfo.value)
    assert chunks[7].ref in message
    assert f"chunk {chunks[7].idx}" in message
    assert "make embedding-cache" in message
    assert f"1 of {len(chunks)} gdpr chunk texts" in message
    # Fail closed means fail closed. The one repair that would look tidy —
    # embedding the missing chunk on the fly — is the defect, so it is not
    # offered, not hinted at, and not reachable through any flag.
    assert "Refusing to embed the missing texts on the fly" in message

    # Negative control: the only difference from the call above is the removed
    # entry. On the complete cache the same chunks resolve, so the raise is the
    # miss and not the call shape.
    assert len(
        cached_chunk_vectors("gdpr", chunks, prefixes, committed_cache, MODEL, DIMENSIONS)
    ) == len(chunks)


def test_a_missing_cache_file_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(EmbeddingCacheError) as excinfo:
        load_chunk_cache(tmp_path / "chunk_embeddings.json")
    assert "make embedding-cache" in str(excinfo.value)

    # Negative control: the same call against the committed file returns a
    # populated cache, so the raise is about absence, not about the path type.
    assert len(load_chunk_cache(CACHE_PATH)) == 284


def test_an_entry_whose_text_changed_counts_as_a_miss_not_a_hit(committed_cache) -> None:
    """The reason the key is a text hash and not `(ref, idx)`.

    Under a positional key an edited chunk would silently keep the vector of its
    previous text — stale, ranks cleanly, invisible in every count this
    repository keeps.
    """
    chunks = corpus_chunks("ai_act")
    prefixes = corpus_prefixes("ai_act", chunks)
    edited = UnitChunk(
        ref=chunks[3].ref,
        heading=chunks[3].heading,
        idx=chunks[3].idx,
        content=chunks[3].content + "\n\nAmended by a later regulation.",
        token_count=chunks[3].token_count,
    )
    assert (edited.ref, edited.idx) == (chunks[3].ref, chunks[3].idx)

    with pytest.raises(EmbeddingCacheError) as excinfo:
        cached_chunk_vectors("ai_act", [edited], [prefixes[3]], committed_cache, MODEL, DIMENSIONS)
    assert edited.ref in str(excinfo.value)

    # Negative control: the unedited chunk — same ref, same idx, same prefix —
    # hits. So the miss is the text change, not the hand-built chunk.
    assert cached_chunk_vectors(
        "ai_act", [chunks[3]], [prefixes[3]], committed_cache, MODEL, DIMENSIONS
    )


def test_an_entry_whose_prefix_changed_counts_as_a_miss(committed_cache) -> None:
    # The prefix is embedded with the content, so a regenerated context sentence
    # changes the vector. `make prefix-cache` without `make embedding-cache`
    # therefore has to fail rather than reuse.
    chunks = corpus_chunks("gdpr")
    prefixes = corpus_prefixes("gdpr", chunks)
    with pytest.raises(EmbeddingCacheError, match="make embedding-cache"):
        cached_chunk_vectors(
            "gdpr", [chunks[0]], [prefixes[0] + " Rewritten."], committed_cache, MODEL, DIMENSIONS
        )

    # Negative control: the unmodified prefix hits.
    assert cached_chunk_vectors(
        "gdpr", [chunks[0]], [prefixes[0]], committed_cache, MODEL, DIMENSIONS
    )


# --- 3. a changed model invalidates ------------------------------------------


def test_the_key_covers_the_embedding_model_and_the_dimension_count() -> None:
    text = "Regulation (EU) 2016/679, Art. 5. Principles.\n\nPersonal data shall be…"
    small = embedding_key("text-embedding-3-small", 1536, text)
    assert small != embedding_key("text-embedding-3-large", 1536, text)
    assert small != embedding_key("text-embedding-3-small", 512, text)

    # Negative control: identical inputs give the identical key, so the three
    # comparisons above are about the model and the width and not about the
    # hash being unstable.
    assert small == embedding_key("text-embedding-3-small", 1536, text)


def test_a_changed_model_misses_every_entry_and_says_why(committed_cache) -> None:
    """A model change must invalidate the cache, not silently pair spaces.

    `text-embedding-3-small` and `-3-large` embed the same string into different
    geometries. A cache keyed on text alone would hit on every chunk and hand
    the ingest vectors that mean nothing in the configured model's space —
    an ingest that succeeds, a corpus that ranks like noise, and no error
    anywhere.
    """
    chunks = corpus_chunks("gdpr")[:3]
    prefixes = corpus_prefixes("gdpr", corpus_chunks("gdpr"))[:3]

    with pytest.raises(EmbeddingCacheError) as excinfo:
        cached_chunk_vectors(
            "gdpr", chunks, prefixes, committed_cache, "text-embedding-3-large", DIMENSIONS
        )

    message = str(excinfo.value)
    assert "3 of 3 gdpr chunk texts" in message
    # The message distinguishes "wrong model" from "corrupt file": 3 of 3
    # missing otherwise reads like the latter and sends the reader hunting.
    assert "different embedding spaces" in message
    assert committed_cache.model in message and "text-embedding-3-large" in message

    # Negative control: same chunks, same cache, configured model — hits.
    assert cached_chunk_vectors("gdpr", chunks, prefixes, committed_cache, MODEL, DIMENSIONS)


def test_a_changed_dimension_count_misses_every_entry(committed_cache) -> None:
    chunks = corpus_chunks("gdpr")[:3]
    prefixes = corpus_prefixes("gdpr", corpus_chunks("gdpr"))[:3]
    with pytest.raises(EmbeddingCacheError, match="different embedding spaces"):
        cached_chunk_vectors("gdpr", chunks, prefixes, committed_cache, MODEL, 512)

    # Negative control: the configured width hits.
    assert cached_chunk_vectors("gdpr", chunks, prefixes, committed_cache, MODEL, DIMENSIONS)


def test_the_committed_file_states_the_model_it_was_generated_with(committed_cache) -> None:
    # Not decoration: `cached_embeddings` reads it to tell a model change apart
    # from a corrupt file, and a reviewer reads it to know what the numbers are
    # numbers of.
    assert committed_cache.model == MODEL
    assert committed_cache.dimensions == DIMENSIONS


# --- 4. the fixture path never calls the API ---------------------------------


async def test_the_fixture_path_reads_the_cache_and_never_calls_the_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guarantee in one assertion: `--source fixture` embeds nothing.

    Embedding on a miss would be the tidy-looking fix and is the defect itself
    — it is what made the vectors of one fixture disagree across ingests.
    """

    async def forbidden(*args: object, **kwargs: object) -> list[list[float]]:
        raise AssertionError("the fixture ingest path called the embedding endpoint")

    monkeypatch.setattr("app.ingestion.pipeline.embed_texts", forbidden)
    chunks = corpus_chunks("gdpr")[:5]
    prefixes = corpus_prefixes("gdpr", corpus_chunks("gdpr"))[:5]

    vectors = await resolve_embeddings("gdpr", chunks, prefixes, "fixture", MODEL, DIMENSIONS)

    assert len(vectors) == len(chunks)
    assert all(len(vector) == DIMENSIONS for vector in vectors)

    # Negative control: the stub is reachable — the network path hits it — so
    # the fixture path passing is a difference in behaviour and not a
    # monkeypatch that missed its target.
    with pytest.raises(AssertionError, match="called the embedding endpoint"):
        await resolve_embeddings("gdpr", chunks, prefixes, "network", MODEL, DIMENSIONS)


async def test_the_fixture_path_fails_closed_when_the_cache_does_not_cover_the_corpus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # End to end through the ingest helper, not just the lookup: a cache that
    # covers the other regulation must stop the ingest rather than fall through
    # to the endpoint.
    chunks = corpus_chunks("gdpr")
    prefixes = corpus_prefixes("gdpr", chunks)
    other = corpus_chunks("ai_act")[0]
    other_prefix = corpus_prefixes("ai_act", corpus_chunks("ai_act"))[0]

    path = tmp_path / "chunk_embeddings.json"
    write_cache(
        path,
        [
            build_entry(
                chunk_embedding_key(MODEL, DIMENSIONS, other_prefix, other),
                [0.5] * DIMENSIONS,
                regulation="ai_act",
                ref=other.ref,
                idx=other.idx,
            )
        ],
        model=MODEL,
        dimensions=DIMENSIONS,
        generated="2026-08-04",
        generated_by="tests",
        note="thin cache",
    )
    monkeypatch.setattr("app.ingestion.chunk_embeddings.CACHE_PATH", path)

    async def forbidden(*args: object, **kwargs: object) -> list[list[float]]:
        raise AssertionError("the fixture ingest path called the embedding endpoint")

    monkeypatch.setattr("app.ingestion.pipeline.embed_texts", forbidden)

    with pytest.raises(EmbeddingCacheError, match="make embedding-cache"):
        await resolve_embeddings("gdpr", chunks, prefixes, "fixture", MODEL, DIMENSIONS)

    # Negative control: restore the committed cache and the identical call
    # succeeds, so the failure is coverage and not the monkeypatch.
    monkeypatch.setattr("app.ingestion.chunk_embeddings.CACHE_PATH", CACHE_PATH)
    assert await resolve_embeddings("gdpr", chunks, prefixes, "fixture", MODEL, DIMENSIONS)


# --- the format itself -------------------------------------------------------


def test_a_vector_round_trips_bit_exactly_at_float32(committed_cache) -> None:
    """float32 in, the same float32 out — the property the whole file rests on.

    If the codec were lossy the committed vector and the ingested one would
    differ, and the guarantee would be false again in a new way.
    """
    original = committed_cache.get(next(iter(committed_cache.vectors)))
    assert decode_vector(encode_vector(original), DIMENSIONS) == original

    # Negative control: perturbing one component by 1e-3 — the size of the
    # largest drift a fresh API pass produces — survives the round trip as a
    # difference. So the equality above is a measurement, not a tautology about
    # two values that were rounded into each other.
    perturbed = list(original)
    perturbed[0] += 1e-3
    assert decode_vector(encode_vector(perturbed), DIMENSIONS) != original


def test_a_truncated_vector_fails_closed() -> None:
    blob = encode_vector([0.25] * 8)
    assert decode_vector(blob, 8) == [0.25] * 8
    with pytest.raises(EmbeddingCacheError, match="expected"):
        decode_vector(blob, 9)


def test_a_file_written_for_a_chunk_round_trips(tmp_path: Path) -> None:
    # Guards the writer against the reader: `refresh_embeddings` builds entries
    # and `load_cache` indexes them, and nothing else checks they agree.
    chunk = corpus_chunks("gdpr")[0]
    prefix = corpus_prefixes("gdpr", corpus_chunks("gdpr"))[0]
    vector = [0.125] * DIMENSIONS
    path = tmp_path / "chunk_embeddings.json"
    write_cache(
        path,
        [build_entry(chunk_embedding_key(MODEL, DIMENSIONS, prefix, chunk), vector)],
        model=MODEL,
        dimensions=DIMENSIONS,
        generated="2026-08-04",
        generated_by="tests",
        note="round trip",
    )
    assert cached_chunk_vectors(
        "gdpr", [chunk], [prefix], load_chunk_cache(path), MODEL, DIMENSIONS
    ) == [vector]

    # Negative control: a file whose declared byte layout is not the one this
    # decoder writes is refused rather than read as if it were.
    document = json.loads(path.read_text())
    document["dtype"] = "float64-be-hex"
    path.write_text(json.dumps(document))
    with pytest.raises(EmbeddingCacheError, match="byte layout"):
        load_chunk_cache(path)


def test_the_committed_file_states_its_own_size() -> None:
    # The format is a size trade-off against a JSON array of decimal floats. A
    # trade-off whose numbers are not written down gets re-litigated on vibes.
    document = json.loads(CACHE_PATH.read_text())
    assert document["dtype"] == DTYPE
    assert document["vectors"] == 284
    assert "MiB raw" in document["size_note"] and "base64" in document["size_note"]


def test_embedded_text_is_defined_once(committed_cache) -> None:
    """The ingest and the cache must join prefix and content identically.

    Two definitions of this string is the way this cache silently stops
    covering the corpus while every count still reads 284.
    """
    chunk = corpus_chunks("gdpr")[0]
    prefix = corpus_prefixes("gdpr", corpus_chunks("gdpr"))[0]
    assert embedding_key(MODEL, DIMENSIONS, embedded_text(prefix, chunk.content)) in committed_cache

    # Negative control: any other join is a miss, which is what would happen if
    # one of the two callers drifted.
    assert embedding_key(MODEL, DIMENSIONS, f"{prefix}\n{chunk.content}") not in committed_cache


def test_an_entry_without_a_usable_vector_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "chunk_embeddings.json"
    write_cache(
        path,
        [build_entry("a" * 64, [0.5] * DIMENSIONS)],
        model=MODEL,
        dimensions=DIMENSIONS,
        generated="2026-08-04",
        generated_by="tests",
        note="",
    )
    assert len(load_cache(path, "make embedding-cache")) == 1

    document = json.loads(path.read_text())
    document["entries"][0]["vector"] = ""
    path.write_text(json.dumps(document))
    with pytest.raises(EmbeddingCacheError, match="usable key/vector pair"):
        load_cache(path, "make embedding-cache")
