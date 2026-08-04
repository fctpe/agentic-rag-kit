"""A committed corpus does not fix a moving query vector, so the queries are pinned too.

The corpus was the larger of the two non-determinisms, not the only one.
Measured over three independent embedding passes across the 38 in-scope golden
questions: 2, 3 and 4 questions of 38 differed between passes; largest cosine
distance between two embeddings of the same question 6.21e-07. In the quantity
that decides ranking — the perturbation to d(query, chunk) — that is mean
1.31e-06 against the corpus side's 2.92e-05, i.e. 22x smaller, and swapping the
query pass while holding the corpus fixed moved no metric and reordered no
question's top 6.

Small, real, and not what `evals/thresholds.yaml` claims. Its retrieval floors
sit close to baseline on the grounds that a retrieval number is a function of
the corpus and the question set; one of those two inputs being a fresh API draw
would keep that sentence false. `evals/query_embeddings.json` pins it, the eval
reads it by default, and `--query-embeddings live` runs the production path as
the control that pinning did not change what is measured.

Offline and free, with a negative control on each property.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from app.config import get_settings
from app.embedding_cache import EmbeddingCacheError, embedding_key

_EVALS = Path(__file__).resolve().parents[2] / "evals"
sys.path.insert(0, str(_EVALS))

from query_embeddings import (  # noqa: E402
    CACHE_PATH,
    cached_query_vectors,
    load_golden_questions,
    load_query_cache,
)

MODEL = get_settings().embedding_model
DIMENSIONS = get_settings().embedding_dimensions


@pytest.fixture(scope="module")
def committed_cache():
    return load_query_cache()


def test_every_golden_question_has_a_committed_vector(committed_cache) -> None:
    questions = load_golden_questions()
    assert len(questions) >= 38

    vectors = cached_query_vectors(
        [q["id"] for q in questions],
        [q["question"] for q in questions],
        committed_cache,
        MODEL,
        DIMENSIONS,
    )
    assert len(vectors) == len(questions)
    for vector in vectors:
        assert len(vector) == DIMENSIONS
        assert any(vector)


def test_the_cache_holds_the_question_set_and_nothing_else(committed_cache) -> None:
    # Both directions. A cache with residue is a cache that was regenerated
    # against a different golden set, and coverage alone would not notice.
    question_keys = {
        embedding_key(MODEL, DIMENSIONS, q["question"]) for q in load_golden_questions()
    }
    assert set(committed_cache.vectors) == question_keys

    # Negative control: one extra key breaks the equality, so it is not vacuous.
    assert set(committed_cache.vectors) | {"0" * 64} != question_keys


def test_out_of_scope_probes_are_cached_too(committed_cache) -> None:
    # `--questions` takes arbitrary ids and run_evals.py sends the out-of-scope
    # probes through the same models. A cache covering only "the ones we usually
    # run in retrieval mode" fails closed the first time someone runs a
    # different subset — which reads as a broken cache, not as a design.
    out_of_scope = [q for q in load_golden_questions() if q["query_type"] == "out_of_scope"]
    assert out_of_scope
    assert cached_query_vectors(
        [q["id"] for q in out_of_scope],
        [q["question"] for q in out_of_scope],
        committed_cache,
        MODEL,
        DIMENSIONS,
    )


def test_an_edited_question_fails_closed_naming_it_and_the_command(committed_cache) -> None:
    question = load_golden_questions()[0]
    with pytest.raises(EmbeddingCacheError) as excinfo:
        cached_query_vectors(
            [question["id"]], [question["question"] + "?"], committed_cache, MODEL, DIMENSIONS
        )
    message = str(excinfo.value)
    assert question["id"] in message
    assert "make embedding-cache" in message

    # Negative control: the unedited question — same id, one character less —
    # resolves, so the raise is the edit and not the one-item call.
    assert cached_query_vectors(
        [question["id"]], [question["question"]], committed_cache, MODEL, DIMENSIONS
    )


def test_a_changed_model_misses_every_question(committed_cache) -> None:
    questions = load_golden_questions()[:3]
    with pytest.raises(EmbeddingCacheError, match="different embedding spaces"):
        cached_query_vectors(
            [q["id"] for q in questions],
            [q["question"] for q in questions],
            committed_cache,
            "text-embedding-3-large",
            DIMENSIONS,
        )

    # Negative control: the configured model hits.
    assert cached_query_vectors(
        [q["id"] for q in questions],
        [q["question"] for q in questions],
        committed_cache,
        MODEL,
        DIMENSIONS,
    )


def test_a_missing_cache_file_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(EmbeddingCacheError, match="make embedding-cache"):
        load_query_cache(tmp_path / "query_embeddings.json")

    # Negative control: the committed file loads.
    assert len(load_query_cache(CACHE_PATH)) == len(load_golden_questions())
