"""Threshold gating shared by the three scored eval suites.

The suites already returned the right exit codes; what was missing was anything
that said which numbers count as a pass. This module is that, and nothing more.

Design rules, all learned from the ways a naive gate goes wrong:

* **Fail closed.** A metric the judge could not score comes back ``None``. That
  is a failure, not a skip — skipping is how a broken suite reports green.
* **Gate the sample size, not just the score.** A run that scores 12 of 38
  questions can post a better mean while covering a third of the ground. Every
  check asserts the expected population first.
* **Refuse to judge a subset.** ``--smoke`` runs a handful of questions; scoring
  that against a full-run baseline is meaningless in both directions.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

THRESHOLDS_PATH = Path(__file__).resolve().parent / "thresholds.yaml"


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


class GateResult:
    def __init__(self, suite: str) -> None:
        self.suite = suite
        self.checks: list[Check] = []

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append(Check(name, passed, detail))

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def report(self, *, stream: Any = None) -> None:
        out = stream or sys.stdout
        print(f"\n## Gate: {self.suite}\n", file=out)
        for c in self.checks:
            print(f"  {'PASS' if c.passed else 'FAIL'}  {c.name}: {c.detail}", file=out)
        verdict = "passed" if self.passed else "FAILED"
        ok = sum(c.passed for c in self.checks)
        print(f"\ngate {verdict} ({ok}/{len(self.checks)})", file=out)


def load_thresholds() -> dict[str, Any]:
    return yaml.safe_load(THRESHOLDS_PATH.read_text())


def _check_metric(
    result: GateResult, name: str, value: float | None, spec: dict[str, float]
) -> None:
    if value is None:
        # The judge errored on this metric. Reporting it as "not compared" would
        # let an outage look like a clean run.
        result.add(name, False, "not scored (judge returned no value) — treated as a failure")
        return
    if "floor" in spec:
        floor = spec["floor"]
        result.add(
            name,
            value >= floor,
            f"{value:.4f} vs floor {floor} (baseline {spec.get('baseline')})",
        )
    if "ceiling" in spec:
        ceiling = spec["ceiling"]
        result.add(
            f"{name} (ceiling)",
            value <= ceiling,
            f"{value:.4f} vs ceiling {ceiling} (baseline {spec.get('baseline')})",
        )


def gate_ragas(payload: dict[str, Any], *, partial: bool) -> GateResult:
    result = GateResult("ragas")
    if partial:
        result.add("population", False, "subset run cannot be gated against a full-run baseline")
        return result

    spec = load_thresholds()["ragas"]
    n_scored = payload.get("n_scored", 0)
    result.add(
        "population",
        n_scored == spec["expect_n_scored"],
        f"{n_scored} questions scored, expected {spec['expect_n_scored']}",
    )
    failures = payload.get("n_chat_failures", 0)
    result.add(
        "chat failures",
        failures <= spec["max_chat_failures"],
        f"{failures}, max {spec['max_chat_failures']}",
    )
    # A run recorded before this check existed carries no field. Reading that as
    # "none missing" would pass every pre-fix run — the same silent green the
    # None-metric rule above refuses.
    unmarked = payload.get("answers_without_inline_citation")
    if unmarked is None:
        result.add(
            "inline citations",
            False,
            "not measured — this run predates the check; re-run run_evals.py",
        )
    else:
        allowed = spec["max_answers_without_inline_citation"]
        detail = f"{len(unmarked)} answer(s) carry no [n] marker, max {allowed}"
        result.add(
            "inline citations",
            len(unmarked) <= allowed,
            detail + (f" — {', '.join(unmarked)}" if unmarked else ""),
        )
    summary = payload.get("summary", {})
    for metric, mspec in spec["metrics"].items():
        _check_metric(result, metric, summary.get(metric), mspec)
    return result


def gate_retrieval(summary: dict[str, Any]) -> GateResult:
    mode = summary.get("mode", "unknown")
    result = GateResult(f"retrieval:{mode}")
    if summary.get("smoke"):
        result.add("population", False, "smoke run cannot be gated against a full-run baseline")
        return result

    spec = load_thresholds()["retrieval"]
    n = summary.get("n_questions", 0)
    result.add(
        "population",
        n == spec["expect_n_questions"],
        f"{n} questions, expected {spec['expect_n_questions']}",
    )
    mode_spec = spec["modes"].get(mode)
    if mode_spec is None:
        result.add("mode", False, f"no thresholds defined for retrieval mode {mode!r}")
        return result
    for metric, mspec in mode_spec["metrics"].items():
        _check_metric(result, metric, summary.get(metric), mspec)
    return result


def gate_redteam(summary: dict[str, Any], *, partial: bool) -> GateResult:
    result = GateResult("redteam")
    if partial:
        result.add("population", False, "smoke run cannot be gated against a full-run baseline")
        return result

    spec = load_thresholds()["redteam"]
    total = summary.get("total", 0)
    passed = summary.get("passed", 0)
    result.add(
        "population",
        total == spec["expect_total"],
        f"{total} cases, expected {spec['expect_total']}",
    )
    result.add(
        "refusals hold", passed >= spec["min_passed"], f"{passed}/{total}, min {spec['min_passed']}"
    )
    return result
