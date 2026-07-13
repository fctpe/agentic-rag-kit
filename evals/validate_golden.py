"""Structural validation of the eval datasets — runs in CI without any
provider key. Guards the golden set and red-team cases against the kind of
drift (missing fields, malformed article refs, empty expectations) that
silently makes a scored eval run meaningless.
"""

import re
import sys
from pathlib import Path

import yaml

EVALS_DIR = Path(__file__).resolve().parent
ARTICLE_LABEL = re.compile(r"^(AI Act|GDPR)\s+Art\.?\s*\S+$")
QUERY_TYPES = {
    "definition",
    "obligation",
    "prohibition",
    "scope",
    "cross_regulation",
    "out_of_scope",
}


def validate_golden() -> list[str]:
    errors: list[str] = []
    data = yaml.safe_load((EVALS_DIR / "golden_questions.yaml").read_text())
    questions = data["questions"]
    ids = [q.get("id") for q in questions]
    if len(ids) != len(set(ids)):
        errors.append("golden: duplicate question ids")
    for q in questions:
        qid = q.get("id", "?")
        for field in ("id", "question", "expected_articles", "reference_answer", "query_type"):
            if field not in q:
                errors.append(f"golden[{qid}]: missing field {field}")
        if q.get("query_type") not in QUERY_TYPES:
            errors.append(f"golden[{qid}]: bad query_type {q.get('query_type')!r}")
        expected = q.get("expected_articles", [])
        if q.get("query_type") == "out_of_scope":
            if expected:
                errors.append(f"golden[{qid}]: out_of_scope must have empty expected_articles")
        else:
            if not expected:
                errors.append(f"golden[{qid}]: in-scope question has no expected_articles")
            for label in expected:
                if not ARTICLE_LABEL.match(str(label).strip()):
                    errors.append(f"golden[{qid}]: malformed article label {label!r}")
    return errors


def validate_redteam() -> list[str]:
    errors: list[str] = []
    data = yaml.safe_load((EVALS_DIR / "redteam" / "cases.yaml").read_text())
    cases = data["cases"] if isinstance(data, dict) else data
    valid_asserts = {"refused", "deflected", "not_echoed", "answered"}
    for case in cases:
        cid = case.get("id", "?")
        for field in ("id", "category", "assert", "message"):
            if field not in case:
                errors.append(f"redteam[{cid}]: missing field {field}")
        if case.get("assert") not in valid_asserts:
            errors.append(f"redteam[{cid}]: bad assert {case.get('assert')!r}")
    return errors


def main() -> int:
    errors = validate_golden() + validate_redteam()
    if errors:
        print("\n".join(errors), file=sys.stderr)
        print(f"\n{len(errors)} validation error(s)", file=sys.stderr)
        return 1
    print("golden set and red-team cases valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
