"""Structural PII redaction applied to user input before it reaches the
model, the database, or traces (OWASP LLM02).

Regex-based by design: it catches the structured identifiers that appear in
compliance-question phrasing (emails, phones, IBANs, cards) with zero heavy
dependencies. Swap in Microsoft Presidio behind the same function signature
for NER-grade coverage — see docs/security.md for the trade-off.
"""

import re
from dataclasses import dataclass

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b")),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,8}\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("PHONE", re.compile(r"(?<![\w/])\+?\d{2,4}[ \-/]?\d{3,}[ \-/]?\d{2,}(?:[ \-/]?\d{2,})?\b")),
]


@dataclass
class RedactionResult:
    text: str
    found: list[str]


def redact_pii(text: str) -> RedactionResult:
    found: list[str] = []
    redacted = text
    for label, pattern in _PATTERNS:
        if pattern.search(redacted):
            found.append(label)
            redacted = pattern.sub(f"[{label} REDACTED]", redacted)
    return RedactionResult(text=redacted, found=found)
