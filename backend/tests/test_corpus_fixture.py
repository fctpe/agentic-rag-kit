"""The committed corpus must stay the corpus the eval numbers were measured on.

README quotes "283 chunks, 212 articles" and every eval result under
evals/results/ was produced against exactly that. If someone regenerates
data/fixtures/ from a newer consolidated text, these counts move and the
committed numbers silently stop describing the corpus — so pin them here.
"""

import pytest

from app.ingestion.chunker import chunk_article
from app.ingestion.eurlex import FixtureError, load_fixture

# Per-regulation split of the 212 articles / 283 chunks the results were run on.
EXPECTED = {"ai_act": (113, 164), "gdpr": (99, 119)}


@pytest.mark.parametrize(("regulation", "counts"), EXPECTED.items())
def test_fixture_matches_the_evaluated_corpus(regulation: str, counts: tuple[int, int]) -> None:
    expected_articles, expected_chunks = counts
    articles = load_fixture(regulation)
    chunks = [chunk for article in articles for chunk in chunk_article(article)]

    assert len(articles) == expected_articles
    assert len(chunks) == expected_chunks


def test_totals_match_the_readme() -> None:
    assert sum(articles for articles, _ in EXPECTED.values()) == 212
    assert sum(chunks for _, chunks in EXPECTED.values()) == 283


def test_articles_carry_text_and_a_citable_ref() -> None:
    for article in load_fixture("gdpr"):
        assert article.paragraphs, f"Art. {article.number} has no body"
        assert article.ref.startswith("Art. ")


def test_missing_fixture_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    # A missing corpus must name the path, not ingest an empty document.
    monkeypatch.setattr("app.ingestion.eurlex.FIXTURE_DIR", tmp_path)
    with pytest.raises(FixtureError, match="No corpus fixture"):
        load_fixture("ai_act")
