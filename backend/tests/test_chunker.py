from app.ingestion.chunker import chunk_unit, count_tokens
from app.ingestion.eurlex import Annex, Article


def make_article(paragraphs: list[str]) -> Article:
    return Article(number="5", heading="Prohibited AI practices", paragraphs=paragraphs)


def make_annex(paragraphs: list[str]) -> Annex:
    return Annex(
        number="III",
        heading="High-risk AI systems referred to in Article 6(2)",
        paragraphs=paragraphs,
    )


def test_short_article_is_one_chunk():
    chunks = chunk_unit(make_article(["A single short paragraph."]))
    assert len(chunks) == 1
    assert chunks[0].ref == "Art. 5"
    assert chunks[0].idx == 0


def test_chunks_never_exceed_token_budget():
    paragraph = "The provider shall ensure compliance with all requirements. " * 20
    chunks = chunk_unit(make_article([paragraph] * 10), max_tokens=300)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= 300


def test_paragraphs_are_packed_whole_when_they_fit():
    paragraphs = [f"Paragraph number {i} with modest length." for i in range(6)]
    chunks = chunk_unit(make_article(paragraphs), max_tokens=700)
    assert len(chunks) == 1
    for paragraph in paragraphs:
        assert paragraph in chunks[0].content


def test_oversized_paragraph_splits_on_sentences():
    sentence = "This obligation applies to every provider of a high-risk system. "
    huge = sentence * 60
    chunks = chunk_unit(make_article([huge]), max_tokens=200)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.token_count <= 200 + count_tokens(sentence)


def test_chunk_indexes_are_sequential():
    paragraph = "Text that repeats to force multiple chunks. " * 30
    chunks = chunk_unit(make_article([paragraph] * 5), max_tokens=250)
    assert [chunk.idx for chunk in chunks] == list(range(len(chunks)))


def test_an_annex_chunks_exactly_like_an_article():
    # Same packing, same indexing. The only difference that reaches a chunk is
    # the ref it carries — and an annex chunk must never wear "Art.".
    paragraph = "Biometric identification systems used by law enforcement. " * 30
    annex_chunks = chunk_unit(make_annex([paragraph] * 5), max_tokens=250)
    article_chunks = chunk_unit(make_article([paragraph] * 5), max_tokens=250)

    assert len(annex_chunks) == len(article_chunks) > 1
    assert [chunk.idx for chunk in annex_chunks] == list(range(len(annex_chunks)))
    assert {chunk.ref for chunk in annex_chunks} == {"Annex III"}
    assert annex_chunks[0].heading == "High-risk AI systems referred to in Article 6(2)"
