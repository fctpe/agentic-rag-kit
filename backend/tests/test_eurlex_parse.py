"""Unit extraction, and the boundaries that keep non-unit text out.

The HTML here mirrors what EUR-Lex actually serves: each article sits in its
own `div id="art_<n>"`, chapter and section headings sit in sibling title
divs, each annex is a `div id="anx_<n>"` after the enacting terms, and the
closing formula lands in `fnp_1`. Section headings and the trailer carry no
container of their own, which is why the parser reads containers and not a
flat run of paragraphs.

Annexes are ingested as their own units — `Annex III` is where the AI Act's
high-risk list actually lives — but they are never articles, and neither kind
may borrow the other's ref or deep-link anchor.
"""

import pytest

from app.ingestion.eurlex import Annex, Article, ParseError, parse_units, unit_ref

MODERN_HTML = """
<html><body>
<div id="art_1" class="eli-subdivision">
  <div id="art_1.tit_1">
    <p class="oj-ti-art">Article 1</p>
    <p class="oj-sti-art">Subject matter</p>
  </div>
  <p class="oj-normal">This Regulation lays down harmonised rules.</p>
  <p class="oj-normal">It applies to providers and deployers.</p>
</div>
<div id="art_2" class="eli-subdivision">
  <div id="art_2.tit_1">
    <p class="oj-ti-art">Article 2</p>
    <p class="oj-sti-art">Scope</p>
  </div>
  <p class="oj-normal">This Regulation applies to the placing on the market of AI systems.</p>
</div>
</body></html>
"""

LEGACY_HTML = """
<html><body>
<div id="art_6" class="eli-subdivision">
  <p class="ti-art">Article 6</p>
  <p class="sti-art">Lawfulness of processing</p>
  <p class="normal">Processing shall be lawful only if consent has been given.</p>
</div>
</body></html>
"""

# Article 7 is the last article of Chapter III Section 1, so the Section 2
# heading, the annexes and the OJ trailer all follow its body with nothing in
# between. Before the container fix they were appended to Art. 7 and Art. 8.
TRAILING_MATTER_HTML = """
<html><body>
<div id="enc_1" class="eli-subdivision">
  <div id="cpt_III">
    <div id="cpt_III.sct_1">
      <div id="art_7" class="eli-subdivision">
        <div id="art_7.tit_1">
          <p class="oj-ti-art">Article 7</p>
          <p class="oj-sti-art">Amendments to Annex III</p>
        </div>
        <p class="oj-normal">The Commission is empowered to adopt delegated acts.</p>
      </div>
    </div>
    <div id="cpt_III.sct_2.tit_1">
      <p class="oj-ti-section-1">SECTION 2</p>
      <p class="oj-ti-section-2">Requirements for high-risk AI systems</p>
    </div>
    <div id="cpt_III.sct_2">
      <div id="art_8" class="eli-subdivision">
        <div id="art_8.tit_1">
          <p class="oj-ti-art">Article 8</p>
          <p class="oj-sti-art">Compliance with the requirements</p>
        </div>
        <p class="oj-normal">High-risk AI systems shall comply with the requirements.</p>
      </div>
    </div>
  </div>
</div>
<div id="fnp_1" class="eli-subdivision">
  <div class="oj-final">
    <p class="oj-normal">This Regulation shall be binding in its entirety.</p>
    <p class="oj-normal">Done at Brussels, 13 June 2024.</p>
  </div>
  <p class="oj-note">( 1 ) OJ C 517, 22.12.2021, p. 56 .</p>
</div>
<div id="anx_II" class="eli-container">
  <p class="oj-doc-ti">ANNEX II</p>
  <p class="oj-doc-ti">List of criminal offences</p>
  <p class="oj-normal">terrorism,</p>
</div>
<div id="anx_III" class="eli-container">
  <p class="oj-doc-ti">ANNEX III</p>
  <p class="oj-doc-ti">High-risk AI systems referred to in Article 6(2)</p>
  <p class="oj-normal">Biometrics, in so far as their use is permitted.</p>
</div>
<p class="oj-normal">ELI: http://data.europa.eu/eli/reg/2024/1689/oj</p>
</body></html>
"""


def test_parses_modern_markup():
    units = parse_units(MODERN_HTML)
    assert [unit.ref for unit in units] == ["Art. 1", "Art. 2"]
    assert units[0].heading == "Subject matter"
    assert len(units[0].paragraphs) == 2
    assert units[1].paragraphs == [
        "This Regulation applies to the placing on the market of AI systems."
    ]


def test_parses_legacy_markup():
    units = parse_units(LEGACY_HTML)
    assert [unit.ref for unit in units] == ["Art. 6"]
    assert units[0].heading == "Lawfulness of processing"


def test_articles_without_body_are_dropped():
    html = (
        '<div id="art_9"><p class="oj-ti-art">Article 9</p></div>'
        '<div id="art_10"><p class="oj-ti-art">Article 10</p>'
        '<p class="oj-normal">Body of ten.</p></div>'
    )
    units = parse_units(html)
    assert [unit.ref for unit in units] == ["Art. 10"]


def test_heading_strips_markup_artifacts():
    html = (
        '<div id="art_3"><p class="oj-ti-art">Article 3</p>'
        '<p class="oj-sti-art">Definitions`</p><p class="oj-normal">Body.</p></div>'
    )
    units = parse_units(html)
    assert units[0].heading == "Definitions"


def test_section_heading_never_lands_in_a_unit():
    units = parse_units(TRAILING_MATTER_HTML)
    body = "\n".join(paragraph for unit in units for paragraph in unit.paragraphs)
    assert "SECTION 2" not in body
    assert "Requirements for high-risk AI systems" not in body


def test_the_oj_trailer_is_still_not_ingested():
    units = parse_units(TRAILING_MATTER_HTML)
    body = "\n".join(paragraph for unit in units for paragraph in unit.paragraphs)
    for stray in ("Done at Brussels", "OJ C 517", "ELI:", "binding in its entirety"):
        assert stray not in body


def test_annexes_are_their_own_units():
    # The point of the exercise: Art. 6(2) makes a system high-risk by pointing
    # at Annex III, and the list itself exists nowhere but the annex.
    annexes = [unit for unit in parse_units(TRAILING_MATTER_HTML) if isinstance(unit, Annex)]
    assert [annex.ref for annex in annexes] == ["Annex II", "Annex III"]

    annex_iii = annexes[-1]
    assert annex_iii.heading == "High-risk AI systems referred to in Article 6(2)"
    assert annex_iii.paragraphs == ["Biometrics, in so far as their use is permitted."]


def test_an_annex_never_claims_to_be_an_article():
    units = parse_units(TRAILING_MATTER_HTML)
    articles = [unit for unit in units if isinstance(unit, Article)]
    annexes = [unit for unit in units if isinstance(unit, Annex)]

    # Refs do not overlap in either direction, and no annex text reaches an
    # article: "Art. 113" used to be 523 paragraphs of annex served under an
    # article's name, which is the failure this asserts against.
    assert [article.ref for article in articles] == ["Art. 7", "Art. 8"]
    assert all(not annex.ref.startswith("Art.") for annex in annexes)
    article_body = "\n".join(p for article in articles for p in article.paragraphs)
    for stray in ("ANNEX II", "ANNEX III", "Biometrics", "terrorism"):
        assert stray not in article_body


def test_the_annex_marker_line_is_not_body_text():
    # "ANNEX III" names the unit; repeating it as content would put a bare
    # heading into a retrievable chunk.
    annexes = [unit for unit in parse_units(TRAILING_MATTER_HTML) if isinstance(unit, Annex)]
    for annex in annexes:
        assert not any(p.upper().startswith("ANNEX ") for p in annex.paragraphs)


def test_units_are_articles_then_annexes():
    assert [type(unit) for unit in parse_units(TRAILING_MATTER_HTML)] == [
        Article,
        Article,
        Annex,
        Annex,
    ]


def test_a_regulation_with_no_annexes_yields_articles_only():
    # The GDPR has none. Zero annexes is an ordinary result, not an error.
    units = parse_units(MODERN_HTML)
    assert units
    assert not [unit for unit in units if isinstance(unit, Annex)]


def test_article_title_outside_a_container_fails_closed():
    # The pre-fix markup shape. Parsing it would yield a corpus silently missing
    # every article whose container disappeared, so refuse rather than guess.
    html = '<p class="oj-ti-art">Article 5</p><p class="oj-normal">Prohibited practices.</p>'
    with pytest.raises(ParseError, match="outside an article container"):
        parse_units(html)


def test_annex_marker_outside_a_container_fails_closed():
    # Without this check the same markup change reads as "this regulation has
    # no annexes" — indistinguishable from the GDPR, where that is true.
    html = (
        '<div id="art_6"><p class="oj-ti-art">Article 6</p>'
        '<p class="oj-normal">Classification rules.</p></div>'
        '<p class="oj-doc-ti">ANNEX III</p>'
        '<p class="oj-normal">Biometrics.</p>'
    )
    with pytest.raises(ParseError, match="outside an annex container"):
        parse_units(html)


def test_a_prose_reference_to_an_annex_does_not_open_one():
    # EUR-Lex gives list markers their own <p>, so "Annex III" can appear as a
    # whole paragraph inside an article. That is a cross-reference, not a
    # marker, and treating it as one would fail an entirely healthy document.
    html = (
        '<div id="art_6"><p class="oj-ti-art">Article 6</p>'
        '<p class="oj-normal">Annex III</p>'
        '<p class="oj-normal">AI systems referred to above shall be high-risk.</p></div>'
    )
    units = parse_units(html)
    assert [unit.ref for unit in units] == ["Art. 6"]


@pytest.mark.parametrize(
    ("supplied", "expected"),
    [
        ("5", "Art. 5"),
        (" 22 ", "Art. 22"),
        ("Annex III", "Annex III"),
        ("annex iii", "Annex III"),
        ("Annex\tIV", "Annex IV"),
    ],
)
def test_unit_ref_normalises_what_the_agent_supplies(supplied: str, expected: str):
    # read_article takes this straight from the model; "annex iii" has to find
    # the chunks stored under "Annex III" rather than report the annex missing.
    assert unit_ref(supplied) == expected


class TestCaptureRatioFailsClosed:
    """The check that would have caught Annex I on the day it broke.

    Every other guard in the module counts *units*: orphan markers outside a
    container, units with no paragraphs at all. None of them compared the text a
    container holds against the text taken out of it, which is why Annex I could
    fall from 5,872 characters to 238 without moving a unit count, a chunk count,
    or a single test. It kept a heading and a non-empty body — it just lost every
    directive it was there to list.
    """

    # The real shape of the loss: EUR-Lex puts the list marker in a <p> and the
    # text it labels in a bare <span>, and a parser reading only <p> takes the
    # numbering and leaves the content.
    _SPAN_CONTENT = (
        '<div id="anx_I"><p class="oj-doc-ti">ANNEX I</p>'
        '<p class="oj-doc-ti">List of Union harmonisation legislation</p>'
        "<table><tr><td></td><td><p>1.</p></td>"
        "<td><span>Directive 2006/42/EC of the European Parliament and of the Council "
        "of 17 May 2006 on machinery, and amending Directive 95/16/EC;</span></td></tr>"
        "<tr><td></td><td><p>2.</p></td>"
        "<td><span>Directive 2009/48/EC of the European Parliament and of the Council "
        "of 18 June 2009 on the safety of toys;</span></td></tr></table></div>"
    )

    def test_span_wrapped_content_is_captured(self):
        units = parse_units(self._SPAN_CONTENT)
        assert len(units) == 1
        assert "Directive 2006/42/EC" in units[0].text
        assert "Directive 2009/48/EC" in units[0].text

    def test_the_old_p_only_rule_would_have_been_rejected(self, monkeypatch):
        """The negative control, and the one that matters.

        Restoring the `<p>`-only extraction must now raise rather than return a
        unit of bare numbering. Without this, `test_span_wrapped_content_is_
        captured` above proves only that the new extractor works — not that the
        guard would stop the old one, which is the property claimed in prose.
        """
        from app.ingestion import eurlex

        def p_only(container, exempt):
            return [
                node.get_text(" ", strip=True)
                for node in container.find_all("p")
                if id(node) not in exempt and node.get_text(strip=True)
            ]

        monkeypatch.setattr(eurlex, "_blocks", p_only)
        with pytest.raises(ParseError, match="word characters"):
            parse_units(self._SPAN_CONTENT)

    def test_the_error_names_the_unit_and_the_shortfall(self, monkeypatch):
        # A guard that fires without saying which unit lost what sends you
        # reading a 1.2 MB document by hand.
        from app.ingestion import eurlex

        monkeypatch.setattr(eurlex, "_blocks", lambda container, exempt: ["1.", "2."])
        with pytest.raises(ParseError) as caught:
            parse_units(self._SPAN_CONTENT)
        message = str(caught.value)
        assert "Annex I" in message
        assert str(eurlex.MIN_CAPTURE_RATIO) in message

    def test_a_genuinely_short_unit_is_not_a_shortfall(self):
        """Negative control for the guard itself. It measures the ratio of what
        was captured to what the container held — not length — so a two-line
        article passes. A guard that reds healthy documents gets deleted."""
        html = (
            '<div id="art_1"><p class="oj-ti-art">Article 1</p>'
            '<p class="oj-sti-art">Subject matter</p>'
            '<p class="oj-normal">This Regulation lays down harmonised rules.</p></div>'
        )
        units = parse_units(html)
        assert [unit.ref for unit in units] == ["Art. 1"]
