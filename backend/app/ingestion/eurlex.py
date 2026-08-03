"""Fetch and parse consolidated regulations from EUR-Lex into citable units.

EUR-Lex marks articles with `ti-art` / `sti-art` paragraph classes (newer
documents prefix them with `oj-`), which makes structure-aware parsing far
more reliable than generic text splitting for statutory text.

Two kinds of unit are ingested, each addressable and cited under its own ref:
articles (`Art. 6`) and annexes (`Annex III`). Annexes are not appendix
material to be dropped — Art. 6(2) makes an AI system high-risk by pointing at
the list in Annex III and nowhere else, so a corpus of articles alone cannot
answer "is my system high-risk?", the most common question a user brings.
Recitals, section headings and the OJ trailer are still not ingested; see
`ARTICLE_CONTAINER_ID` / `ANNEX_CONTAINER_ID` for how that boundary is drawn.

The same units are also committed under `data/fixtures/` so ingestion runs
without EUR-Lex; `load_fixture` reads those and `parse_units` is bypassed.
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import httpx
from bs4 import BeautifulSoup, Tag

REGULATIONS = {
    "ai_act": {
        "title": "Regulation (EU) 2024/1689 (Artificial Intelligence Act)",
        "celex": "32024R1689",
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32024R1689",
    },
    "gdpr": {
        "title": "Regulation (EU) 2016/679 (General Data Protection Regulation)",
        "celex": "32016R0679",
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679",
    },
}

ARTICLE_TITLE_CLASSES = ("oj-ti-art", "ti-art")
ARTICLE_SUBTITLE_CLASSES = ("oj-sti-art", "sti-art")

# Annexes carry no article marker of any kind: both the "ANNEX III" line and
# the title under it are plain document titles, the same class the
# regulation's own masthead uses ("REGULATION (EU) 2024/1689 …"). The class
# therefore cannot say "an annex starts here" — only the container id can.
ANNEX_TITLE_CLASSES = ("oj-doc-ti", "doc-ti")

# Paragraph classes are not enough on their own: annexes, section headings and
# the OJ trailer carry no article marker, so a flat scan that keeps appending
# until the next `ti-art` sweeps all of them into whichever article came last.
# EUR-Lex does mark the boundary, one level up — every subdivision gets its own
# div id: `art_113` for the article, `art_113.tit_1` for its title block,
# `cpt_XIII` for the chapter, `cpt_III.sct_2.tit_1` for a section heading,
# `anx_III` for an annex, `fnp_1` for the closing formula and footnotes. The
# `[^.]` keeps the nested `art_113.tit_1` out; it holds the title, which is
# read from its own container anyway. The same ids back the `#art_113` /
# `#anx_III` deep links in retrieval/citations.py, so the scheme is already
# load-bearing here.
ARTICLE_CONTAINER_ID = re.compile(r"art_[^.]+")
ANNEX_CONTAINER_ID = re.compile(r"anx_[^.]+")


@dataclass
class Unit:
    """One citable subdivision of a regulation: an article or an annex.

    Subclasses differ in exactly two facts — the ref a citation prints and the
    EUR-Lex id prefix that ref anchors at — and both are declared here so the
    parser, the chunker and the citation builder cannot drift apart on them.
    """

    #: Printed ref prefix: "Art. 6", "Annex III".
    label: ClassVar[str]
    #: EUR-Lex subdivision id prefix: `#art_6`, `#anx_III`.
    anchor_prefix: ClassVar[str]

    number: str
    heading: str
    paragraphs: list[str] = field(default_factory=list)

    @property
    def ref(self) -> str:
        return f"{self.label} {self.number}"

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)


@dataclass
class Article(Unit):
    label: ClassVar[str] = "Art."
    anchor_prefix: ClassVar[str] = "art"


@dataclass
class Annex(Unit):
    label: ClassVar[str] = "Annex"
    anchor_prefix: ClassVar[str] = "anx"


#: Ref label -> EUR-Lex id prefix. `retrieval/citations.py` builds deep links
#: from the persisted ref string alone (a `chunks.article_ref` value, with no
#: Unit object in reach), so it needs this mapping; deriving it from the unit
#: classes keeps "Annex III" from ever anchoring at `#art_III`.
ANCHOR_PREFIXES: dict[str, str] = {cls.label: cls.anchor_prefix for cls in (Article, Annex)}

_ANNEX_INPUT = re.compile(r"annex\s*(\w+)", re.IGNORECASE)


def unit_ref(number: str) -> str:
    """Normalise a caller-supplied unit number to the ref shape chunks store.

    "5" -> "Art. 5"; "Annex III" and "annex iii" -> "Annex III". Article
    numbers are arabic and annex numbers roman, so the word is what
    disambiguates them — a bare "III" is not assumed to mean an annex.
    """
    match = _ANNEX_INPUT.fullmatch(number.strip())
    if match:
        return f"{Annex.label} {match.group(1).upper()}"
    return f"{Article.label} {number.strip()}"


# A served article page is hundreds of KB. Anything this small is an
# interstitial or an empty body, never the regulation.
_MIN_DOCUMENT_BYTES = 20_000


class FetchError(RuntimeError):
    """The document could not be retrieved. Distinct from a parse failure."""


def fetch_html(url: str, timeout: float = 60.0) -> str:
    headers = {"User-Agent": "agentic-rag-kit/0.1 (research; contact via GitHub)"}
    response = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
    response.raise_for_status()

    # EUR-Lex answers 202 with an empty body when it declines to serve the
    # document — bot protection or a queued render, not an error status, so
    # raise_for_status() lets it through. Without this check the empty string
    # reaches the parser, which finds no articles and reports that the markup
    # changed. That sends you looking for a parser bug that does not exist.
    if len(response.text) < _MIN_DOCUMENT_BYTES:
        raise FetchError(
            f"EUR-Lex returned HTTP {response.status_code} with "
            f"{len(response.text)} bytes for {url}. The document was not served "
            "— this is a fetch problem (rate limiting or bot protection), not a "
            "parsing problem. Retry later; the parser is untouched."
        )
    return response.text


# Repo root: backend/app/ingestion/eurlex.py -> parents[3]. Resolved from the
# module rather than the working directory so `make ingest-smoke` (which cds
# into backend/) and the eval workflow find the same files.
FIXTURE_DIR = Path(__file__).resolve().parents[3] / "data" / "fixtures"


class FixtureError(RuntimeError):
    """The committed corpus is missing or malformed. Distinct from a fetch failure."""


def load_fixture(regulation: str) -> list[Unit]:
    """Read a regulation's units from the committed corpus instead of EUR-Lex.

    The fixture holds what `parse_units` returned on the date recorded in it,
    so the two sources feed the pipeline the same `Unit` objects: articles
    first, then annexes.
    """
    path = FIXTURE_DIR / f"{regulation}.json"
    if not path.is_file():
        raise FixtureError(
            f"No corpus fixture at {path}. Expected one committed per regulation; "
            "run without --source fixture to ingest from EUR-Lex instead."
        )

    document = json.loads(path.read_text())
    if document.get("regulation") != regulation:
        raise FixtureError(
            f"{path} declares regulation {document.get('regulation')!r}, expected {regulation!r}."
        )
    # A regulation may genuinely have no annexes (the GDPR has none), so an
    # empty list is valid and a missing key is not: it means the file predates
    # annex ingestion, and reading it would quietly serve an AI Act corpus with
    # no Annex III in it.
    if "annexes" not in document:
        raise FixtureError(
            f"{path} has no 'annexes' key — it was written before annexes were ingested as "
            "their own units. Regenerate it from EUR-Lex rather than ingesting a corpus that "
            "is silently missing them."
        )

    articles: list[Unit] = [
        Article(
            number=str(entry["number"]),
            heading=entry["heading"],
            paragraphs=list(entry["paragraphs"]),
        )
        for entry in document["articles"]
    ]
    if not articles:
        raise FixtureError(f"{path} contains no articles.")

    annexes: list[Unit] = [
        Annex(
            number=str(entry["number"]),
            heading=entry["heading"],
            paragraphs=list(entry["paragraphs"]),
        )
        for entry in document["annexes"]
    ]
    return [*articles, *annexes]


def _has_class(node: Tag, classes: tuple[str, ...]) -> bool:
    # bs4 returns a list for multi-valued attributes but a bare str when the
    # parser saw a single value; normalise so `in` never falls back to
    # substring matching against a string.
    raw: str | list[str] = node.get("class") or []
    node_classes: list[str] = raw if isinstance(raw, list) else [raw]
    return any(cls in node_classes for cls in classes)


def _article_number(title_text: str) -> str | None:
    # Titles read "Article 5" or "Article 5a".
    parts = title_text.split()
    if len(parts) >= 2 and parts[0].lower() == "article":
        return parts[1].strip()
    return None


def _annex_number(title_text: str) -> str | None:
    # Annex markers read "ANNEX III" and nothing else — EUR-Lex separates the
    # two with a non-breaking space, which str.split() treats as whitespace.
    # Requiring the line to be *only* the marker is what keeps a prose mention
    # ("systems listed in Annex III") from opening a new annex.
    parts = title_text.split()
    if len(parts) == 2 and parts[0].lower() == "annex":
        return parts[1].strip()
    return None


class ParseError(RuntimeError):
    """The markup no longer carries the structure the parser reads."""


def _is_article_container(node: Tag) -> bool:
    if node.name != "div":
        return False
    return ARTICLE_CONTAINER_ID.fullmatch(str(node.get("id") or "")) is not None


def _is_annex_container(node: Tag) -> bool:
    if node.name != "div":
        return False
    return ANNEX_CONTAINER_ID.fullmatch(str(node.get("id") or "")) is not None


def _parse_article_container(container: Tag) -> Article | None:
    """Read one `art_<n>` subdivision. Everything in it belongs to that article."""
    paragraphs = [node for node in container.find_all("p") if isinstance(node, Tag)]
    title = next((node for node in paragraphs if _has_class(node, ARTICLE_TITLE_CLASSES)), None)
    if title is None:
        return None
    number = _article_number(title.get_text(" ", strip=True))
    if number is None:
        return None

    article = Article(number=number, heading="")
    for node in paragraphs:
        if node is title:
            continue
        text = node.get_text(" ", strip=True)
        if not text:
            continue
        if _has_class(node, ARTICLE_SUBTITLE_CLASSES):
            if not article.heading:
                # EUR-Lex markup occasionally leaks stray backticks/asterisks.
                article.heading = text.strip("`* ")
            continue
        article.paragraphs.append(text)
    return article


def _parse_annex_container(container: Tag) -> Annex | None:
    """Read one `anx_<n>` container. Everything in it belongs to that annex.

    Mirrors `_parse_article_container`: the marker line names the unit, the
    title line under it becomes the heading, and the rest is body. The one
    difference is that both marker and heading wear the same class, so the
    marker is identified by its text rather than by its class.
    """
    paragraphs = [node for node in container.find_all("p") if isinstance(node, Tag)]
    marker: Tag | None = None
    number: str | None = None
    for node in paragraphs:
        if not _has_class(node, ANNEX_TITLE_CLASSES):
            continue
        number = _annex_number(node.get_text(" ", strip=True))
        if number is not None:
            marker = node
            break
    if marker is None or number is None:
        return None

    annex = Annex(number=number, heading="")
    for node in paragraphs:
        if node is marker:
            continue
        text = node.get_text(" ", strip=True)
        if not text:
            continue
        if _has_class(node, ANNEX_TITLE_CLASSES):
            if not annex.heading:
                annex.heading = text.strip("`* ")
            continue
        annex.paragraphs.append(text)
    return annex


def _containerless(
    paragraphs: list[Tag],
    is_marker: Callable[[Tag], bool],
    is_container: Callable[[Tag], bool],
) -> list[Tag]:
    """Marker paragraphs that no enclosing container claims."""
    return [
        node
        for node in paragraphs
        if is_marker(node) and not any(is_container(parent) for parent in node.parents)
    ]


def parse_units(html: str) -> list[Unit]:
    """Every citable subdivision of the document: articles first, then annexes."""
    soup = BeautifulSoup(html, "lxml")
    paragraphs = [node for node in soup.find_all("p") if isinstance(node, Tag)]

    # A unit marker outside its container means the subdivision scheme this
    # parser depends on is gone. Silently returning the units that still have
    # one would ingest a corpus quietly missing the rest, so refuse instead.
    # Both checks are needed: the article one alone would leave a markup change
    # around the annexes to fall through as "this regulation has none", which
    # is indistinguishable from the GDPR, where it is true.
    orphan_articles = _containerless(
        paragraphs,
        lambda node: _has_class(node, ARTICLE_TITLE_CLASSES),
        _is_article_container,
    )
    if orphan_articles:
        raise ParseError(
            f"{len(orphan_articles)} article title(s) sit outside an article container, the "
            f"first being {orphan_articles[0].get_text(' ', strip=True)!r}. EUR-Lex marks each "
            "article with a div id of the form 'art_<n>'; that structure is what separates "
            "article text from annexes and the OJ trailer. Refusing to parse a partial corpus."
        )

    # Class *and* text: EUR-Lex puts list markers in their own <p>, so a cell
    # reading exactly "Annex III" inside an article would otherwise look like a
    # loose annex marker and fail an entirely healthy document.
    orphan_annexes = _containerless(
        paragraphs,
        lambda node: (
            _has_class(node, ANNEX_TITLE_CLASSES)
            and _annex_number(node.get_text(" ", strip=True)) is not None
        ),
        _is_annex_container,
    )
    if orphan_annexes:
        raise ParseError(
            f"{len(orphan_annexes)} annex marker(s) sit outside an annex container, the first "
            f"being {orphan_annexes[0].get_text(' ', strip=True)!r}. EUR-Lex marks each annex "
            "with a div id of the form 'anx_<n>'; without it there is nothing separating one "
            "annex from the next. Refusing to parse a partial corpus."
        )

    articles: list[Unit] = []
    annexes: list[Unit] = []
    for node in soup.find_all("div"):
        if not isinstance(node, Tag):
            continue
        unit: Unit | None
        if _is_article_container(node):
            unit = _parse_article_container(node)
            target = articles
        elif _is_annex_container(node):
            unit = _parse_annex_container(node)
            target = annexes
        else:
            continue
        if unit is not None and unit.paragraphs:
            target.append(unit)
    return [*articles, *annexes]
