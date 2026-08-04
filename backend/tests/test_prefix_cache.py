"""The committed corpus has to be reproducible, prefixes included.

`--source fixture` used to call the model once per chunk to write the contextual
half of each prefix. `temperature=0` is not determinism, so the same committed
fixture produced a different corpus on every ingest — measured here as hybrid
MRR 0.888 on one ingest and 0.914 on the next, which is wider than the floors
`evals/thresholds.yaml` sets on the strength of retrieval being deterministic.

The prefixes are now generated once and committed. These tests hold the three
properties that makes true, all offline and free:

1. the committed cache covers the committed fixture, so a fixture refreshed
   without its prefixes fails here rather than at ingest time;
2. a miss raises, names the chunks and names the regeneration command — it does
   not fall back to the model (irreproducible) or to the deterministic prefix
   (a silently half-contextual corpus);
3. the key covers chunk *content*, so an edited fixture cannot reuse a sentence
   written for different text.

Every one is paired with the negative control that proves the assertion can
fail: a check that only ever passes proves nothing about the property it names.
"""

from __future__ import annotations

import json

import pytest

from app.ingestion.chunker import UnitChunk, chunk_unit
from app.ingestion.contextual import deterministic_prefix
from app.ingestion.eurlex import REGULATIONS, load_fixture
from app.ingestion.pipeline import resolve_prefixes
from app.ingestion.prefix_cache import (
    CACHE_PATH,
    PrefixCacheError,
    build_entry,
    cached_prefixes,
    chunk_key,
    load_cache,
    write_cache,
)

#: Kept in step with tests/test_corpus_fixture.py::EXPECTED, which pins the same
#: corpus from the text side. Both have to move together or one of them fails.
EXPECTED_CHUNKS = {"ai_act": 167, "gdpr": 117}


def corpus_chunks(regulation: str) -> list[UnitChunk]:
    return [chunk for unit in load_fixture(regulation) for chunk in chunk_unit(unit)]


def title(regulation: str) -> str:
    return str(REGULATIONS[regulation]["title"])


@pytest.fixture(scope="module")
def committed_cache() -> dict[str, str]:
    return load_cache()


@pytest.mark.parametrize("regulation", sorted(EXPECTED_CHUNKS))
def test_committed_cache_covers_every_chunk_of_the_committed_fixture(
    regulation: str, committed_cache: dict[str, str]
) -> None:
    """The test that catches a fixture regenerated without its prefixes.

    `make refresh-fixtures` is free and `make prefix-cache` is not, so the two
    will drift; this is where that drift surfaces, on every push, rather than in
    an ingest that someone runs with a provider key.
    """
    chunks = corpus_chunks(regulation)
    assert len(chunks) == EXPECTED_CHUNKS[regulation]

    prefixes = cached_prefixes(regulation, title(regulation), chunks, committed_cache)

    assert len(prefixes) == len(chunks)
    assert all(prefix.strip() for prefix in prefixes)


def test_the_cache_holds_the_corpus_and_nothing_else(committed_cache: dict[str, str]) -> None:
    # Coverage alone would pass on a cache carrying entries for chunks that no
    # longer exist — the residue of a fixture that shrank. Equality of key sets
    # catches that direction too, and it is free to check.
    corpus_keys = {
        chunk_key(title(regulation), chunk)
        for regulation in EXPECTED_CHUNKS
        for chunk in corpus_chunks(regulation)
    }
    assert len(corpus_keys) == sum(EXPECTED_CHUNKS.values()) == 284
    assert set(committed_cache) == corpus_keys


def test_a_missing_entry_fails_closed_naming_the_chunks_and_the_command(
    committed_cache: dict[str, str],
) -> None:
    chunks = corpus_chunks("gdpr")
    dropped = chunks[7]
    incomplete = {
        k: v for k, v in committed_cache.items() if k != chunk_key(title("gdpr"), dropped)
    }
    assert len(incomplete) == len(committed_cache) - 1

    with pytest.raises(PrefixCacheError) as excinfo:
        cached_prefixes("gdpr", title("gdpr"), chunks, incomplete)

    message = str(excinfo.value)
    assert dropped.ref in message
    assert f"chunk {dropped.idx}" in message
    assert "make prefix-cache" in message
    assert f"1 of {len(chunks)} gdpr chunks" in message
    # Fail closed means fail closed: the deterministic prefix is not a fallback,
    # because a corpus that is 99.6% contextual and 0.4% not is a third corpus
    # nobody measured. Nothing that could serve as one is offered.
    assert deterministic_prefix(title("gdpr"), dropped) not in message

    # Negative control: the *only* difference between this and the call above is
    # the entry that was removed. On the complete cache the same chunks resolve,
    # so the raise is caused by the miss and not by the call shape.
    prefixes = cached_prefixes("gdpr", title("gdpr"), chunks, committed_cache)
    assert len(prefixes) == len(chunks)


def test_a_missing_cache_file_fails_closed(tmp_path) -> None:
    with pytest.raises(PrefixCacheError) as excinfo:
        load_cache(tmp_path / "context_prefixes.json")
    assert "make prefix-cache" in str(excinfo.value)

    # Negative control: the same call against the committed file returns a
    # populated cache, so the raise is about absence, not about the path type.
    assert len(load_cache(CACHE_PATH)) == 284


def test_an_entry_whose_content_changed_counts_as_a_miss_not_a_hit(
    committed_cache: dict[str, str],
) -> None:
    """The reason the key is a content hash and not `(ref, idx)`.

    Under a positional key, editing a chunk's text would silently pair it with
    the sentence a model wrote about the *old* text — a stale prefix that
    embeds cleanly and is invisible in every count.
    """
    original = corpus_chunks("ai_act")[3]
    edited = UnitChunk(
        ref=original.ref,
        heading=original.heading,
        idx=original.idx,
        content=original.content + "\n\nAmended by a later regulation.",
        token_count=original.token_count,
    )
    assert (edited.ref, edited.idx) == (original.ref, original.idx)

    with pytest.raises(PrefixCacheError) as excinfo:
        cached_prefixes("ai_act", title("ai_act"), [edited], committed_cache)
    assert edited.ref in str(excinfo.value)

    # Negative control: the unedited chunk — same ref, same idx, same everything
    # but the two sentences appended above — still hits. So the miss is caused by
    # the content change, not by passing a hand-built chunk or a one-item list.
    assert cached_prefixes("ai_act", title("ai_act"), [original], committed_cache)


def test_the_key_also_covers_content_past_the_prompt_truncation(tmp_path) -> None:
    # The prompt only carries the first 4,000 characters, so hashing the rendered
    # prompt alone would make two long chunks that diverge later collide — and a
    # collision is a hit on the wrong sentence, the exact failure above.
    base = UnitChunk(
        ref="Art. 1", heading="Subject matter", idx=0, content="x" * 5000, token_count=1
    )
    longer = UnitChunk(
        ref="Art. 1", heading="Subject matter", idx=0, content="x" * 5000 + "y", token_count=1
    )
    assert chunk_key("R", base) != chunk_key("R", longer)


def test_a_cache_written_for_a_chunk_round_trips(tmp_path) -> None:
    # Guards the writer against the reader: `refresh_prefixes` builds entries and
    # `load_cache` indexes them, and nothing else checks they agree on the key.
    chunk = corpus_chunks("gdpr")[0]
    path = tmp_path / "context_prefixes.json"
    write_cache(
        [build_entry("gdpr", title("gdpr"), chunk, "A sentence about scope.")],
        model="openai:gpt-4o-mini",
        generated="2026-08-04",
        path=path,
    )
    assert cached_prefixes("gdpr", title("gdpr"), [chunk], load_cache(path)) == [
        "A sentence about scope."
    ]

    # Negative control: a blank prefix is not a usable entry, and committing one
    # would produce a cache "hit" that contributes nothing to the embedding.
    path.write_text(
        json.dumps({"entries": [{"key": chunk_key(title("gdpr"), chunk), "prefix": "  "}]})
    )
    with pytest.raises(PrefixCacheError):
        load_cache(path)


async def test_the_fixture_path_reads_the_cache_and_never_calls_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guarantee in one assertion: `--source fixture` does not sample a model.

    Generating on a miss would be the tidy-looking fix and is the defect itself
    — it is exactly what made two ingests of one fixture disagree.
    """

    async def forbidden(*args: object, **kwargs: object) -> list[str]:
        raise AssertionError("the fixture ingest path called the model")

    monkeypatch.setattr("app.ingestion.pipeline.llm_prefixes", forbidden)
    chunks = corpus_chunks("gdpr")[:5]

    prefixes = await resolve_prefixes("gdpr", title("gdpr"), chunks, True, "fixture", "unused")

    assert len(prefixes) == len(chunks)
    for chunk, prefix in zip(chunks, prefixes, strict=True):
        assert prefix.startswith(deterministic_prefix(title("gdpr"), chunk))
        assert len(prefix) > len(deterministic_prefix(title("gdpr"), chunk))

    # Negative control, two ways. The stub is reachable — the network path hits
    # it, so the fixture path passing is a real difference in behaviour and not
    # a monkeypatch that missed its target...
    with pytest.raises(AssertionError, match="called the model"):
        await resolve_prefixes("gdpr", title("gdpr"), chunks, True, "network", "unused")

    # ...and the fixture path is not passing by quietly skipping the contextual
    # half: with --no-contextual it returns bare deterministic prefixes, which is
    # visibly not what it returned above.
    bare = await resolve_prefixes("gdpr", title("gdpr"), chunks, False, "fixture", "unused")
    assert bare == [deterministic_prefix(title("gdpr"), chunk) for chunk in chunks]
    assert bare != prefixes


async def test_the_fixture_path_fails_closed_when_the_cache_does_not_cover_the_corpus(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    # End to end through the ingest entry point, not just the helper: an empty
    # cache file must stop the ingest rather than let it fall through to either
    # fallback. `write_cache`/`load_cache` reject an empty entries list, so the
    # thinnest cache that parses is one entry for a chunk of the other regulation.
    path = tmp_path / "context_prefixes.json"
    other = corpus_chunks("ai_act")[0]
    write_cache(
        [build_entry("ai_act", title("ai_act"), other, "Scope of the Act.")],
        model="openai:gpt-4o-mini",
        generated="2026-08-04",
        path=path,
    )
    monkeypatch.setattr("app.ingestion.prefix_cache.CACHE_PATH", path)

    async def forbidden(*args: object, **kwargs: object) -> list[str]:
        raise AssertionError("the fixture ingest path called the model")

    monkeypatch.setattr("app.ingestion.pipeline.llm_prefixes", forbidden)

    with pytest.raises(PrefixCacheError, match="make prefix-cache"):
        await resolve_prefixes("gdpr", title("gdpr"), corpus_chunks("gdpr"), True, "fixture", "x")

    # Negative control: restore the committed cache and the identical call
    # succeeds, so the failure is the cache's coverage and not the monkeypatch.
    monkeypatch.setattr("app.ingestion.prefix_cache.CACHE_PATH", CACHE_PATH)
    assert await resolve_prefixes(
        "gdpr", title("gdpr"), corpus_chunks("gdpr"), True, "fixture", "x"
    )
