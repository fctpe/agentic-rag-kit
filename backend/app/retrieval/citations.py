import re

from app.ingestion.eurlex import REGULATIONS

_ARTICLE_NUMBER = re.compile(r"Art\.\s*(\w+)")


def citation_url(regulation: str, article_ref: str) -> str:
    """Deep link into the EUR-Lex HTML text, anchored at the article."""
    meta = REGULATIONS.get(regulation)
    if not meta:
        return ""
    match = _ARTICLE_NUMBER.search(article_ref)
    anchor = f"#art_{match.group(1)}" if match else ""
    return f"{meta['url']}{anchor}"
