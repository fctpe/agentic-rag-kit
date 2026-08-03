"""The committed corpus must stay the corpus the eval numbers were measured on.

Regenerating data/fixtures/ from a newer consolidated text — or from a changed
parser — moves these counts, and the committed numbers then silently stop
describing the corpus. Pinning them here makes that a test failure instead.

The counts below are what the unit parser produced from live EUR-Lex on
2026-08-04: the same articles the container-scoped article parser produced,
plus the annexes it dropped. Two earlier corpora are now stale, and every eval
figure in README.md, evals/thresholds.yaml and evals/results/ predates both —
113/164 + 99/119 = 212/283 counted annex text, section headings and the OJ
trailer swept into the preceding article, and 113/145 + 99/117 = 212/262
dropped the annexes entirely.
"""

import json

import pytest

from app.ingestion.chunker import chunk_unit, count_tokens
from app.ingestion.eurlex import Annex, Article, FixtureError, load_fixture
from app.retrieval.citations import citation_url

# Per-regulation (articles, annexes, chunks) in data/fixtures/. The GDPR has no
# annexes at all — zero is the right answer there, not a broken fixture.
EXPECTED = {"ai_act": (113, 13, 163), "gdpr": (99, 0, 117)}


@pytest.mark.parametrize(("regulation", "counts"), EXPECTED.items())
def test_fixture_matches_the_evaluated_corpus(
    regulation: str, counts: tuple[int, int, int]
) -> None:
    expected_articles, expected_annexes, expected_chunks = counts
    units = load_fixture(regulation)
    chunks = [chunk for unit in units for chunk in chunk_unit(unit)]

    assert sum(1 for unit in units if isinstance(unit, Article)) == expected_articles
    assert sum(1 for unit in units if isinstance(unit, Annex)) == expected_annexes
    assert len(chunks) == expected_chunks


def test_corpus_totals() -> None:
    assert sum(articles for articles, _, _ in EXPECTED.values()) == 212
    assert sum(annexes for _, annexes, _ in EXPECTED.values()) == 13
    assert sum(chunks for _, _, chunks in EXPECTED.values()) == 280


def test_units_carry_text_and_a_citable_ref() -> None:
    for regulation in EXPECTED:
        for unit in load_fixture(regulation):
            assert unit.paragraphs, f"{regulation} {unit.ref} has no body"
            assert unit.ref.startswith(("Art. ", "Annex "))


def test_annex_iii_is_present_and_carries_the_high_risk_list() -> None:
    # Art. 6(2) classifies a system as high-risk by pointing at Annex III; the
    # list exists nowhere else, so without the annex the corpus cannot answer
    # "is my system high-risk?" at all.
    annexes = {unit.number: unit for unit in load_fixture("ai_act") if isinstance(unit, Annex)}
    assert sorted(annexes) == sorted(
        ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII"]
    )

    annex_iii = annexes["III"]
    assert annex_iii.ref == "Annex III"
    assert "High-risk AI systems referred to in Article" in annex_iii.heading
    for use_case in ("Biometrics", "Critical infrastructure", "Education", "Employment"):
        assert use_case in annex_iii.text


def test_no_annex_chunk_claims_to_be_an_article() -> None:
    # The failure being repaired: annex text served to users under an article's
    # ref. An annex chunk carries the annex's ref, and its deep link anchors at
    # `#anx_<n>` — `#art_III` resolves to nothing and drops the reader at the
    # top of a 1.2 MB document.
    for regulation in EXPECTED:
        for unit in load_fixture(regulation):
            if not isinstance(unit, Annex):
                continue
            for chunk in chunk_unit(unit):
                assert chunk.ref == unit.ref
                assert not chunk.ref.startswith("Art.")
                assert citation_url(regulation, chunk.ref).endswith(f"#anx_{unit.number}")


def test_every_unit_deep_links_to_its_own_eur_lex_anchor() -> None:
    for regulation in EXPECTED:
        for unit in load_fixture(regulation):
            prefix = "anx" if isinstance(unit, Annex) else "art"
            assert citation_url(regulation, unit.ref).endswith(f"#{prefix}_{unit.number}")


def test_annexes_are_chunked_like_articles_not_left_as_blobs() -> None:
    # An annex ingested whole would be one enormous chunk that outranks every
    # article citing it: the pre-fix Art. 113 blob with a correct label on it.
    annexes = [unit for unit in load_fixture("ai_act") if isinstance(unit, Annex)]
    for unit in annexes:
        chunks = chunk_unit(unit)
        assert chunks
        assert [chunk.idx for chunk in chunks] == list(range(len(chunks)))
        # chunk_unit packs whole paragraphs, so a chunk can run past the
        # 700-token budget by part of the paragraph that tipped it — GDPR
        # article chunks already reach 701. The bound that holds for articles
        # and annexes alike is the budget plus one paragraph.
        ceiling = 700 + max(count_tokens(paragraph) for paragraph in unit.paragraphs)
        for chunk in chunks:
            assert chunk.token_count <= ceiling

    # Annex III is 66 paragraphs. One chunk would mean it was never packed.
    annex_iii = next(unit for unit in annexes if unit.number == "III")
    assert len(chunk_unit(annex_iii)) > 1


def test_no_article_absorbs_the_annexes_or_the_trailer() -> None:
    # Art. 113 was a 523-paragraph blob of annex text served to users as
    # "Entry into force". Nothing outside the enacting terms belongs to it.
    articles = {unit.number: unit for unit in load_fixture("ai_act") if isinstance(unit, Article)}
    entry_into_force = articles["113"]
    assert entry_into_force.heading == "Entry into force and application"
    assert len(entry_into_force.paragraphs) == 9
    for stray in ("ANNEX", "ISSN", "ELI:", "Done at Brussels"):
        assert stray not in entry_into_force.text


def test_no_article_chunk_is_just_a_section_heading() -> None:
    # "SECTION 2 / Requirements for high-risk AI systems" was offered as a
    # source under Art. 7. A heading sitting between two articles belongs to
    # neither and is not citable. Annexes are excluded on purpose: their own
    # subdivision labels ("Section A — Information to be submitted by
    # providers…") are annex text and do belong to the unit holding them.
    # EUR-Lex separates the numeral from the word with a non-breaking space.
    for regulation in EXPECTED:
        for unit in load_fixture(regulation):
            if not isinstance(unit, Article):
                continue
            for chunk in chunk_unit(unit):
                opening = chunk.content.replace("\xa0", " ").lstrip().upper()
                assert not opening.startswith(("SECTION ", "CHAPTER ", "ANNEX ")), (
                    f"{regulation} {unit.ref} chunk {chunk.idx} opens on a heading"
                )


def test_missing_fixture_fails_closed(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    # A missing corpus must name the path, not ingest an empty document.
    monkeypatch.setattr("app.ingestion.eurlex.FIXTURE_DIR", tmp_path)
    with pytest.raises(FixtureError, match="No corpus fixture"):
        load_fixture("ai_act")


def test_fixture_without_an_annexes_key_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    # A fixture written before annexes existed would otherwise load as a
    # perfectly valid AI Act corpus that merely happens to have no Annex III.
    (tmp_path / "ai_act.json").write_text(
        json.dumps(
            {
                "regulation": "ai_act",
                "articles": [{"number": "6", "heading": "Classification", "paragraphs": ["x"]}],
            }
        )
    )
    monkeypatch.setattr("app.ingestion.eurlex.FIXTURE_DIR", tmp_path)
    with pytest.raises(FixtureError, match="no 'annexes' key"):
        load_fixture("ai_act")
