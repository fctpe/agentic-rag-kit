"""Deep links: the ref a chunk carries decides which EUR-Lex anchor it gets.

`chunks.article_ref` holds two shapes now — "Art. 6" and "Annex III" — and
EUR-Lex gives them different subdivision ids (`art_6`, `anx_III`). One rule
for both would send every annex citation to `#art_III`, which matches nothing
in the document: the browser silently drops the reader at the top of a 1.2 MB
page, and the "View on EUR-Lex" link stops being evidence for the claim it
was attached to.
"""

import pytest

from app.ingestion.eurlex import REGULATIONS
from app.retrieval.citations import citation_url

AI_ACT = REGULATIONS["ai_act"]["url"]


@pytest.mark.parametrize(
    ("ref", "anchor"),
    [
        ("Art. 6", "#art_6"),
        ("Art. 5", "#art_5"),
        ("Art. 6a", "#art_6a"),
        ("Annex III", "#anx_III"),
        ("Annex I", "#anx_I"),
        ("Annex XIII", "#anx_XIII"),
    ],
)
def test_each_ref_shape_anchors_at_its_own_subdivision(ref: str, anchor: str) -> None:
    assert citation_url("ai_act", ref) == f"{AI_ACT}{anchor}"


def test_an_annex_is_never_anchored_as_an_article() -> None:
    url = citation_url("ai_act", "Annex III")
    assert "#art_" not in url


def test_an_unknown_regulation_yields_no_link() -> None:
    assert citation_url("nis2", "Art. 6") == ""


def test_an_unrecognisable_ref_links_to_the_document_not_a_wrong_anchor() -> None:
    # Better to land on the regulation than on an anchor that does not exist.
    assert citation_url("ai_act", "Recital 27") == AI_ACT
