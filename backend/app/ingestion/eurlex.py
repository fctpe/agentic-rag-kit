"""Fetch and parse consolidated regulations from EUR-Lex into articles.

EUR-Lex marks articles with `ti-art` / `sti-art` paragraph classes (newer
documents prefix them with `oj-`), which makes structure-aware parsing far
more reliable than generic text splitting for statutory text.
"""

from dataclasses import dataclass, field

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


@dataclass
class Article:
    number: str
    heading: str
    paragraphs: list[str] = field(default_factory=list)

    @property
    def ref(self) -> str:
        return f"Art. {self.number}"

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)


def fetch_html(url: str, timeout: float = 60.0) -> str:
    headers = {"User-Agent": "agentic-rag-kit/0.1 (research; contact via GitHub)"}
    response = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
    response.raise_for_status()
    return response.text


def _has_class(node: Tag, classes: tuple[str, ...]) -> bool:
    node_classes = node.get("class") or []
    return any(cls in node_classes for cls in classes)


def _article_number(title_text: str) -> str | None:
    # Titles read "Article 5" or "Article 5a".
    parts = title_text.split()
    if len(parts) >= 2 and parts[0].lower() == "article":
        return parts[1].strip()
    return None


def parse_articles(html: str) -> list[Article]:
    soup = BeautifulSoup(html, "lxml")
    articles: list[Article] = []
    current: Article | None = None

    for node in soup.find_all("p"):
        if not isinstance(node, Tag):
            continue
        text = node.get_text(" ", strip=True)
        if not text:
            continue

        if _has_class(node, ARTICLE_TITLE_CLASSES):
            number = _article_number(text)
            if number:
                current = Article(number=number, heading="")
                articles.append(current)
            continue

        if current is None:
            continue

        if _has_class(node, ARTICLE_SUBTITLE_CLASSES):
            if not current.heading:
                # EUR-Lex markup occasionally leaks stray backticks/asterisks.
                current.heading = text.strip("`* ")
            continue

        current.paragraphs.append(text)

    return [article for article in articles if article.paragraphs]
