"""Inline source markers, resolved by code rather than requested from the model.

A `[n]` marker is not decoration: it is the only thing that ties a sentence to
the source-panel entry it came from, and the frontend links exactly one shape —
`/\\[(\\d{1,3})\\]/`. Anything else the model writes ("(AI Act, Art. 50(1))",
`[4, Art. 4(5)]`, `[2(a)]`, `[7, 8]`) renders as literal text and the source it
names is never linked.

Asking the model for the shape was tried and measured: a single sentence in the
system prompt warning against one wrong bracket moved unmarked answers from 2 to
9 and 10 out of 37 (reverted in bb4e582). Prompt text is a request. This module
is the guarantee.

Three rules, and they are deliberately narrow:

  `[n, Art. 4(5)]`  ->  `(Art. 4(5)) [n]`   the reference is kept, moved out
  `[n(a)]`          ->  `[n]`               the sub-point is not a source id
  `[n, m]`          ->  `[n][m]`            a list of ids is a list of markers

Everything else fails closed, and "closed" here means *not linked*:

  * an index that is not in the retrieved source list is REMOVED, never
    renumbered and never guessed at — a marker resolving to a source that does
    not support the sentence looks checked and is not, which is strictly worse
    than no marker;
  * a bracket whose spelled-out reference contradicts the source it numbers
    (`[5, Art. 4(7)]` where source 5 is GDPR Art. 28) is left exactly as the
    model wrote it, so it stays visible as text rather than becoming a
    confidently wrong link;
  * a bracket shaped like nothing above is left alone.

Every one of those three cases is reported in `issues`, so a bracket that does
not link is never silently swallowed either.

An answer that cites nothing — a refusal, an out-of-scope decline — contains no
brackets, so it passes through untouched and unmarked. Nothing here invents a
citation.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# The linkable marker, and the ONLY shape the frontend turns into a button.
# Kept byte-identical to CITE_PATTERN in frontend/lib/citationMarkers.ts —
# frontend/tests/citation-markers.test.mjs pins that side of the contract.
MARKER_PATTERN = re.compile(r"\[(\d{1,3})\]")

# Any bracket that opens with 1-3 digits: the superset of MARKER_PATTERN that
# includes every merged shape the model actually writes.
_BRACKET = re.compile(r"\[(\d{1,3})([^\[\]]*)\]")

# Code is not prose. The frontend plugin skips `code`/`pre` subtrees, so this
# skips fenced blocks and inline spans for the same reason: rewriting `[1, 2]`
# inside a snippet would corrupt it, and it would never have been linked anyway.
# The `|\Z` alternatives keep an unterminated fence from swallowing the rewrite.
_CODE = re.compile(r"```[\s\S]*?(?:```|\Z)|~~~[\s\S]*?(?:~~~|\Z)|`[^`\n]*`")

_INDEX_LIST = re.compile(r"(?:,\s*\d{1,3})+")
_SUBPOINT = re.compile(r"(?:\(\w{1,4}\))+")
_ARTICLE_UNIT = re.compile(r"\bArt(?:icle)?s?\.?\s*(\d{1,3})")
_ANNEX_UNIT = re.compile(r"\bAnnex\s+([IVXLCDM]+)\b", re.IGNORECASE)

_REGULATION_LABELS = {"ai_act": "AI Act", "gdpr": "GDPR"}


@dataclass(frozen=True)
class ResolvedAnswer:
    """The text that may be shown, and every bracket that will not link."""

    text: str
    issues: list[str]


def _unit_key(text: str) -> tuple[str, str] | None:
    """The regulation unit a string names, comparable across spellings.

    Annexes are matched first: "Annex III" carries no digits, and the article
    pattern must not fall through to something else on the same line.
    """
    annex = _ANNEX_UNIT.search(text)
    if annex:
        return ("annex", annex.group(1).upper())
    article = _ARTICLE_UNIT.search(text)
    if article:
        return ("art", str(int(article.group(1))))
    return None


def _describe(index: int, citation: Mapping[str, Any]) -> str:
    regulation = str(citation.get("regulation", ""))
    label = _REGULATION_LABELS.get(regulation, regulation)
    return f"source {index} is {label} {citation.get('article', '?')}".strip()


def _dropped_issue(literal: str, unknown: Sequence[int], count: int) -> str:
    names = ", ".join(str(index) for index in unknown)
    return (
        f'Removed "{literal}": no source {names} was retrieved '
        f"({count} source{'' if count == 1 else 's'} in this thread), so the marker "
        "could not point anywhere."
    )


def _mismatch_issue(literal: str, index: int, citation: Mapping[str, Any]) -> str:
    return (
        f'Left "{literal}" unlinked: it spells out a different provision than '
        f"{_describe(index, citation)}. Linking it would have pointed the reader at a "
        "source that does not say what the sentence claims."
    )


def _unrecognised_issue(literal: str) -> str:
    return f'Left "{literal}" unlinked: it is not a source marker this system can resolve.'


def _resolve_bracket(
    literal: str,
    index: int,
    tail: str,
    by_index: Mapping[int, Mapping[str, Any]],
) -> tuple[str, str | None]:
    """One bracket -> (replacement text, issue or None).

    An empty replacement means "remove it": the only branch that deletes, and it
    deletes exactly the markers that resolve to nothing.
    """
    tail = tail.strip()

    if tail == "":
        if index in by_index:
            return literal, None
        return "", _dropped_issue(literal, [index], len(by_index))

    if _INDEX_LIST.fullmatch(tail):
        indices = [index, *(int(part) for part in re.findall(r"\d{1,3}", tail))]
        unknown = [candidate for candidate in indices if candidate not in by_index]
        if unknown:
            return "", _dropped_issue(literal, unknown, len(by_index))
        return "".join(f"[{candidate}]" for candidate in indices), None

    if _SUBPOINT.fullmatch(tail):
        # `[2(a)]` names source 2 and a point inside it. The point is dropped,
        # not moved into the prose: "(a)" on its own resolves to nothing a
        # reader could open, and the citation scheme's finest unit is the
        # article anyway. The source id is what survives, because it is the part
        # that means something.
        if index not in by_index:
            return "", _dropped_issue(literal, [index], len(by_index))
        return f"[{index}]", None

    if tail.startswith(","):
        reference = tail[1:].strip()
        if index not in by_index:
            return "", _dropped_issue(literal, [index], len(by_index))
        claimed = _unit_key(reference)
        actual = _unit_key(str(by_index[index].get("article", "")))
        if claimed is not None and actual is not None and claimed != actual:
            return literal, _mismatch_issue(literal, index, by_index[index])
        if not reference:
            return f"[{index}]", None
        return f"({reference}) [{index}]", None

    return literal, _unrecognised_issue(literal)


def _rewrite_prose(
    segment: str, by_index: Mapping[int, Mapping[str, Any]]
) -> tuple[str, list[str]]:
    parts: list[str] = []
    issues: list[str] = []
    cursor = 0
    for match in _BRACKET.finditer(segment):
        parts.append(segment[cursor : match.start()])
        cursor = match.end()
        replacement, issue = _resolve_bracket(
            match.group(0), int(match.group(1)), match.group(2), by_index
        )
        if issue is not None:
            issues.append(issue)
        if replacement:
            parts.append(replacement)
        elif parts and parts[-1].endswith(" "):
            # Removing " [12]" before a full stop would otherwise leave " .".
            following = segment[cursor] if cursor < len(segment) else ""
            if following == "" or following in ".,;:!?)":
                parts[-1] = parts[-1][:-1]
    parts.append(segment[cursor:])
    return "".join(parts), issues


def resolve_markers(text: str, citations: Sequence[Mapping[str, Any]]) -> ResolvedAnswer:
    """Rewrite an answer so every `[n]` left in it resolves to a retrieved source.

    Postcondition, enforced below rather than assumed: `unlinkable_brackets` of
    the returned text contains no plain `[n]` marker. It is checked because the
    guarantee this function exists to make is exactly that one, and a rewrite
    rule that quietly stops applying would otherwise look like success.
    """
    by_index: dict[int, Mapping[str, Any]] = {}
    for citation in citations:
        index = citation.get("index")
        if isinstance(index, int):
            by_index[index] = citation

    parts: list[str] = []
    issues: list[str] = []
    cursor = 0
    for code in _CODE.finditer(text):
        rewritten, found = _rewrite_prose(text[cursor : code.start()], by_index)
        parts.append(rewritten)
        issues.extend(found)
        parts.append(code.group(0))
        cursor = code.end()
    rewritten, found = _rewrite_prose(text[cursor:], by_index)
    parts.append(rewritten)
    issues.extend(found)
    result = "".join(parts)

    stranded = [
        literal
        for literal in unlinkable_brackets(result, by_index)
        if MARKER_PATTERN.fullmatch(literal)
    ]
    if stranded:
        # Unreachable by construction, and checked anyway: a marker that links to
        # nothing is the one failure this module must not ship. Strip and say so.
        result, _ = _rewrite_prose(result, by_index)
        issues.append(
            "Removed "
            + ", ".join(f'"{literal}"' for literal in sorted(set(stranded)))
            + ": the marker survived resolution without a matching source."
        )
    return ResolvedAnswer(text=result, issues=issues)


def unlinkable_brackets(
    text: str, by_index: Mapping[int, Mapping[str, Any]] | None = None
) -> list[str]:
    """Every citation-shaped bracket in `text` that the frontend will NOT link.

    Per bracket, not per answer. The eval's `INLINE_CITATION.search(answer)` is
    satisfied by one good marker anywhere, so an answer could ship ten
    unlinkable `[2(a)]…[2(j)]` brackets and still count as marked. The thing
    that breaks is brackets, so brackets are what this counts.

    With `by_index`, a plain `[n]` whose index was never retrieved counts too:
    it renders as a button that scrolls to nothing.
    """
    found: list[str] = []
    spans: list[tuple[int, int]] = [(code.start(), code.end()) for code in _CODE.finditer(text)]
    for match in _BRACKET.finditer(text):
        if any(start <= match.start() < end for start, end in spans):
            continue
        if match.group(2).strip() == "":
            if by_index is not None and int(match.group(1)) not in by_index:
                found.append(match.group(0))
            continue
        found.append(match.group(0))
    return found
