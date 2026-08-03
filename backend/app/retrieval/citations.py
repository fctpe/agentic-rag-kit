import re

from app.ingestion.eurlex import ANCHOR_PREFIXES, REGULATIONS

# `chunks.article_ref` holds one of the two ref shapes the parser produces —
# "Art. 6" or "Annex III" — and they anchor at different EUR-Lex subdivision
# ids (`art_6` vs `anx_III`). A single `#art_<n>` rule would deep link
# "Annex III" to `#art_III`, which resolves to nothing: the reader lands at the
# top of a 1.2 MB document instead of the high-risk list they were shown. The
# labels come from the unit classes so a new unit kind cannot be added without
# its anchor.
_REF = re.compile(rf"({'|'.join(re.escape(label) for label in ANCHOR_PREFIXES)})\s*(\w+)")


def citation_url(regulation: str, article_ref: str) -> str:
    """Deep link into the EUR-Lex HTML text, anchored at the cited unit."""
    meta = REGULATIONS.get(regulation)
    if not meta:
        return ""
    match = _REF.search(article_ref)
    anchor = f"#{ANCHOR_PREFIXES[match.group(1)]}_{match.group(2)}" if match else ""
    return f"{meta['url']}{anchor}"
