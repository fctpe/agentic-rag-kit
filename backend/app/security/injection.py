"""Prompt-injection heuristics for user input (OWASP LLM01, first layer).

Heuristics gate obvious attacks cheaply and log them to the audit trail;
the deeper defenses are structural — retrieved chunks are delimited as
untrusted <source> blocks, the system prompt instructs the model to treat
them as data, and report-type answers pass a human approval gate. The
red-team suite in evals/redteam asserts all three layers.
"""

import re
from dataclasses import dataclass

_SIGNALS: list[tuple[str, re.Pattern[str]]] = [
    (
        "override_instructions",
        re.compile(r"ignore (all|any|previous|prior|above) (instructions?|prompts?|rules?)", re.I),
    ),
    (
        "role_hijack",
        re.compile(r"\byou are (now|no longer)\b|\bpretend to be\b|\bact as (?!a lawyer)", re.I),
    ),
    (
        "system_prompt_probe",
        re.compile(
            r"(reveal|show|print|repeat|output).{0,40}(system prompt|instructions|rules)", re.I
        ),
    ),
    ("delimiter_smuggling", re.compile(r"</?source[^>]*>|<\|im_(start|end)\|>|```system", re.I)),
    (
        "exfiltration",
        re.compile(r"(send|post|forward|upload).{0,40}(http|www\.|api|webhook)", re.I),
    ),
]


@dataclass
class InjectionAssessment:
    flagged: bool
    signals: list[str]


def assess_injection(text: str) -> InjectionAssessment:
    signals = [name for name, pattern in _SIGNALS if pattern.search(text)]
    return InjectionAssessment(flagged=bool(signals), signals=signals)
