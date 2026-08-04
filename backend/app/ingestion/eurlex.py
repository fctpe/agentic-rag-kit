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
from bs4 import BeautifulSoup, CData, NavigableString, Tag

# Two URLs per regulation, and the split is deliberate.
#
# `url` is where a *reader* goes: the EUR-Lex page, which is what a citation in
# an answer should link to and what anyone checking the claim expects to see.
#
# `fetch_url` is where the *ingester* goes: Cellar, the Publications Office's
# machine-readable interface to the same repository EUR-Lex itself reads from.
# The EUR-Lex web endpoint answers 202 with an empty body under bot protection —
# not rarely, but for half an hour at a stretch — which makes a fresh ingest a
# coin flip. Cellar served both documents without complaint through exactly that
# window, and content negotiation (`Accept: application/xhtml+xml`) returns the
# same XHTML: parsing both sources produces byte-identical units, verified over
# all 225 of them. So this is the same content from the same publisher through
# the interface intended for programs, not a substitute source.
#
# Formex XML (`Accept: application/xml;notice=branch`) is also available and is
# structurally richer — explicit ARTICLE and ANNEX elements would make the
# `<p>`-versus-`<span>` class of loss impossible by construction rather than
# caught by a coverage check. It is the better long-term parser target and a
# full rewrite; ADR 0003 records why it is not this change.
REGULATIONS = {
    "ai_act": {
        "title": "Regulation (EU) 2024/1689 (Artificial Intelligence Act)",
        "celex": "32024R1689",
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32024R1689",
        "fetch_url": "http://publications.europa.eu/resource/celex/32024R1689",
    },
    "gdpr": {
        "title": "Regulation (EU) 2016/679 (General Data Protection Regulation)",
        "celex": "32016R0679",
        "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32016R0679",
        "fetch_url": "http://publications.europa.eu/resource/celex/32016R0679",
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
    headers = {
        "User-Agent": "agentic-rag-kit/0.1 (research; contact via GitHub)",
        # Content negotiation, for the Cellar endpoint. EUR-Lex's own web URL
        # ignores it and serves HTML either way, so one request shape covers
        # both and the caller does not have to know which it is talking to.
        "Accept": "application/xhtml+xml, text/html;q=0.9",
        "Accept-Language": "eng",
    }
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


#: Elements that end a run of inline text; everything else is inline.
#:
#: EUR-Lex's unit of body text is the table cell, not the paragraph. It wraps
#: cell content in `<p class="oj-normal">` for ordinary two-column list rows,
#: but a three-column extra-indent row puts a bare `<span>` in the content cell
#: while the *marker* cell keeps its `<p>`:
#:
#:     <tr><td></td><td><p>1.</p></td><td><span>Directive 2006/42/EC …</span></td></tr>
#:
#: Reading `container.find_all("p")` is a whitelist of exactly one tag, so it
#: emitted "1." "2." "3." and dropped every Directive they label — Annex I of
#: the AI Act came out as 238 characters of bare list markers. The asymmetry is
#: what made it silent: the unit still had a heading and a non-empty
#: `paragraphs`, so every structural check downstream stayed green.
#:
#: The rule below is a blacklist of *structure* instead: a block ends a
#: paragraph, and a paragraph's text is whatever is not a nested block. That
#: survives EUR-Lex swapping `<span>` for `<em>`, `<font>` or a bare text node,
#: because none of them is named anywhere. Only a genuinely new *block* element
#: would need this set extended, and that failure mode is loud (two paragraphs
#: run together) rather than silent (text disappears).
_BLOCK_ELEMENTS = frozenset(
    {
        "p", "div", "table", "thead", "tbody", "tfoot", "tr", "td", "th",
        "ul", "ol", "li", "dl", "dt", "dd", "blockquote",
        "h1", "h2", "h3", "h4", "h5", "h6",
    }
)  # fmt: skip

#: The node types bs4's own `get_text()` treats as text (`MAIN_CONTENT_STRING_TYPES`).
#: Matching it exactly — by type, not `isinstance` — is what keeps comments and
#: processing instructions out of the body, and keeps the extraction below
#: byte-identical to `get_text(" ", strip=True)` wherever nothing was being lost.
_TEXT_NODE_TYPES = (NavigableString, CData)


def _blocks(container: Tag, exempt: frozenset[int]) -> list[str]:
    """Body text of `container`, one string per leaf block, in document order.

    `exempt` holds `id()`s of nodes whose subtrees are not body — the lines that
    name the unit. Identity rather than text equality: two blocks can carry the
    same characters, and matching on text would silently exempt the wrong one.

    Each block is joined with bs4's `get_text(" ", strip=True)` semantics, so
    `\\xa0` is preserved inside a run (the corpus has `1.\\xa0\\xa0\\xa0Where …`)
    and only whitespace-only runs disappear.
    """
    blocks: list[str] = []
    run: list[str] = []

    def flush() -> None:
        text = " ".join(part for part in (raw.strip() for raw in run) if part)
        run.clear()
        if text:
            blocks.append(text)

    def walk(node: Tag) -> None:
        for child in node.children:
            if type(child) in _TEXT_NODE_TYPES:
                run.append(str(child))
            elif isinstance(child, Tag) and id(child) not in exempt:
                if child.name in _BLOCK_ELEMENTS:
                    flush()
                    walk(child)
                    flush()
                else:
                    walk(child)

    walk(container)
    flush()
    return blocks


_NON_WORD = re.compile(r"\W+")


def _word_chars(text: str) -> int:
    """Length of `text` counting only word characters.

    Whitespace is exactly what the two sides of the coverage check are allowed
    to disagree about — one has been through `get_text(" ", strip=True)` and the
    other has not — so it is the one thing the measure must ignore.
    """
    return len(_NON_WORD.sub("", text))


#: Floor for the fraction of a container's body characters a unit must carry.
#:
#: Measured over both regulations (225 units: 212 articles + 13 annexes), the
#: extraction above scores exactly 1.000 on every one of them — it partitions
#: the container, so each character is either in an exempt title/heading block
#: or in exactly one body block. The `<p>`-only rule it replaces scored 0.034 on
#: Annex I, 0.452 on Annex VII and 0.755 on Annex XI. Any floor in (0.755, 1.0]
#: separates the two, so the number is chosen for headroom rather than for
#: sensitivity: 0.95 tolerates a future extractor dropping something genuinely
#: incidental while still failing every loss ever measured here by a wide
#: margin. Raising it to 1.0 would be a stricter check that no real corpus has
#: ever needed and that any whitespace-adjacent bug would turn into a false
#: alarm; lowering it below 0.755 would re-admit the failure this exists for.
MIN_CAPTURE_RATIO = 0.95


def _check_capture(unit: Unit, container: Tag, title: Tag) -> None:
    """Fail closed if the unit carries materially less text than its container.

    Every other fail-closed check in this module is structural and counts
    *units*: `_containerless` looks for orphan markers, `parse_units` keeps a
    unit if it has any paragraphs at all. None of them compares the text a
    container holds against the text extracted from it, which is why Annex I
    could shrink from 5,789 characters to 196 without moving a single unit or
    chunk count. This is that comparison.

    Only the title line — "Article 6", "ANNEX III" — is excused: it names the
    unit and is deliberately not stored. Everything else the container holds is
    expected back, the heading included, so a dropped heading counts as loss
    and so does a second subtitle line that `_unit_body` held out of the body.
    """
    available = _word_chars(container.get_text(" ")) - _word_chars(title.get_text(" "))
    if available <= 0:
        # Nothing but the title line — an empty article. `parse_units` drops it.
        return
    captured = _word_chars(unit.heading) + _word_chars(unit.text)
    if captured / available < MIN_CAPTURE_RATIO:
        raise ParseError(
            f"{unit.ref} captured {captured} of {available} word characters "
            f"({captured / available:.3f}) present in its EUR-Lex container, below the "
            f"{MIN_CAPTURE_RATIO} floor. The markup carries body text in a shape this "
            "parser no longer reads — see _BLOCK_ELEMENTS. Refusing to ingest a unit "
            "that silently drops most of what it cites."
        )


def _unit_body(
    container: Tag, title: Tag, subtitle_classes: tuple[str, ...]
) -> tuple[str, list[str]]:
    """Split a container into its heading and its body paragraphs.

    The title line and every line wearing a subtitle class name the unit rather
    than belong to it; the first of the latter with text is the heading. Those
    are the only blocks held out of the body — everything else in the container
    is body, whatever tag happens to wrap it.
    """
    heading = ""
    exempt = [id(title)]
    for node in container.find_all("p"):
        if not isinstance(node, Tag) or node is title or not _has_class(node, subtitle_classes):
            continue
        exempt.append(id(node))
        text = node.get_text(" ", strip=True)
        if text and not heading:
            # EUR-Lex markup occasionally leaks stray backticks/asterisks.
            heading = text.strip("`* ")
    return heading, _blocks(container, frozenset(exempt))


def _parse_article_container(container: Tag) -> Article | None:
    """Read one `art_<n>` subdivision. Everything in it belongs to that article."""
    paragraphs = [node for node in container.find_all("p") if isinstance(node, Tag)]
    title = next((node for node in paragraphs if _has_class(node, ARTICLE_TITLE_CLASSES)), None)
    if title is None:
        return None
    number = _article_number(title.get_text(" ", strip=True))
    if number is None:
        return None

    heading, body = _unit_body(container, title, ARTICLE_SUBTITLE_CLASSES)
    article = Article(number=number, heading=heading, paragraphs=body)
    _check_capture(article, container, title)
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

    heading, body = _unit_body(container, marker, ANNEX_TITLE_CLASSES)
    annex = Annex(number=number, heading=heading, paragraphs=body)
    _check_capture(annex, container, marker)
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
