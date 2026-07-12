from app.ingestion.eurlex import parse_articles

MODERN_HTML = """
<html><body>
<p class="oj-ti-art">Article 1</p>
<p class="oj-sti-art">Subject matter</p>
<p class="oj-normal">This Regulation lays down harmonised rules.</p>
<p class="oj-normal">It applies to providers and deployers.</p>
<p class="oj-ti-art">Article 2</p>
<p class="oj-sti-art">Scope</p>
<p class="oj-normal">This Regulation applies to the placing on the market of AI systems.</p>
</body></html>
"""

LEGACY_HTML = """
<html><body>
<p class="ti-art">Article 6</p>
<p class="sti-art">Lawfulness of processing</p>
<p class="normal">Processing shall be lawful only if consent has been given.</p>
</body></html>
"""


def test_parses_modern_oj_markup():
    articles = parse_articles(MODERN_HTML)
    assert [article.ref for article in articles] == ["Art. 1", "Art. 2"]
    assert articles[0].heading == "Subject matter"
    assert len(articles[0].paragraphs) == 2
    assert "harmonised rules" in articles[0].text


def test_parses_legacy_markup():
    articles = parse_articles(LEGACY_HTML)
    assert len(articles) == 1
    assert articles[0].ref == "Art. 6"
    assert articles[0].heading == "Lawfulness of processing"


def test_articles_without_body_are_dropped():
    html = (
        '<p class="oj-ti-art">Article 9</p><p class="oj-ti-art">Article 10</p>'
        '<p class="oj-normal">Body of ten.</p>'
    )
    articles = parse_articles(html)
    assert [article.ref for article in articles] == ["Art. 10"]


def test_heading_strips_markup_artifacts():
    html = (
        '<p class="oj-ti-art">Article 3</p><p class="oj-sti-art">Definitions`</p>'
        '<p class="oj-normal">Body.</p>'
    )
    articles = parse_articles(html)
    assert articles[0].heading == "Definitions"
