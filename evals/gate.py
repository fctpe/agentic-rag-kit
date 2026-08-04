"""Threshold gating shared by the three scored eval suites.

The suites already returned the right exit codes; what was missing was anything
that said which numbers count as a pass. This module is that, and nothing more.

Design rules, all learned from the ways a naive gate goes wrong:

* **Fail closed.** A metric the judge could not score comes back ``None``. That
  is a failure, not a skip — skipping is how a broken suite reports green.
* **Gate the sample size, not just the score.** A run that scores 12 of 38
  questions can post a better mean while covering a third of the ground. Every
  check asserts the expected population first.
* **Gate it per metric, not just per run.** The run-level count is how many
  questions got an *answer*. A judge call that dies takes its question out of
  one metric's mean and leaves it in the other three, so that count never
  moves — which is exactly how a 23-of-38 faithfulness mean shipped under a
  run reporting 38. Every published mean has to name its own denominator.
* **Refuse to judge a subset.** ``--smoke`` runs a handful of questions; scoring
  that against a full-run baseline is meaningless in both directions.
* **A baseline nothing compares against is decoration.** Every check printed the
  baseline next to the floor and then compared against the floor alone. That is
  how the committed hybrid MRR moved 0.8912 -> 0.875 in commit 88131af — a side
  effect of a run nobody promoted, still above the 0.87 floor, so nothing said a
  word while the README went on quoting 0.891. A metric spec may now carry
  ``max_drift``, and the measured value has to sit within that of the declared
  baseline.

  Only the retrieval metrics carry it, and they carry ``0.0``. That split is not
  timidity, it is what the two suites can actually promise. Retrieval is
  deterministic — pinned chunk text, pinned chunk vectors, pinned query vectors,
  every ``ORDER BY`` totally ordered — so a re-run cannot move the number on its
  own, and a number that moved is a change somebody made. RAGAS is judged by an
  LLM: two runs of one corpus scored faithfulness 0.9176 and 0.9316, so a
  two-sided band there is either wide enough to pass anything or narrow enough
  to red the build on a draw. Its baseline is held instead by the fact that
  ``promote.py`` is the only thing that can move the committed artifact.

  There is no deadlock in ``max_drift: 0.0``, which is the objection to a hard
  equality check. Improving retrieval means editing ``thresholds.yaml`` to
  declare the new number and then promoting the run that measured it. That is
  one extra deliberate line in the diff, and it is exactly the line whose
  absence let the committed number and the published number disagree.
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
    if "max_drift" in spec:
        baseline = spec.get("baseline")
        if baseline is None:
            # A tolerance around nothing. Refusing is the only reading that does
            # not silently drop the check the key was added to switch on.
            result.add(f"{name} (drift)", False, "max_drift is set but no baseline is declared")
            return
        drift = abs(value - baseline)
        max_drift = spec["max_drift"]
        result.add(
            f"{name} (drift)",
            drift <= max_drift,
            f"{value:.4f} vs declared baseline {baseline} "
            f"(drift {drift:.4f}, max {max_drift}) — a deterministic suite that "
            "moved changed something; declare the new baseline, then promote"
            if drift > max_drift
            else f"{value:.4f} matches declared baseline {baseline}",
        )


def gate_ragas(payload: dict[str, Any], *, partial: bool) -> GateResult:
    result = GateResult("ragas")
    if partial:
        result.add("population", False, "subset run cannot be gated against a full-run baseline")
        return result

    spec = load_thresholds()["ragas"]
    expected = spec["expect_n_scored"]
    n_scored = payload.get("n_scored", 0)
    result.add(
        "population",
        n_scored == expected,
        f"{n_scored} questions scored, expected {expected}",
    )
    # The denominator of each individual mean. `n_scored` above counts questions
    # that produced an answer, not questions that produced a score, so it stays
    # at 38 while a metric quietly averages over whatever its judge managed to
    # return. Absent means the run predates the check — same fail-closed reading
    # as the inline-citation field below, and for the same reason: defaulting to
    # "nothing missing" passes every artifact that has the defect.
    contributing = payload.get("n_contributing")
    if contributing is None:
        result.add(
            "per-metric population",
            False,
            "means do not state their denominators — this run predates the check; "
            "re-run run_evals.py",
        )
    else:
        for metric in spec["metrics"]:
            n = contributing.get(metric)
            result.add(
                f"{metric} population",
                n == expected,
                f"{n} of {expected} questions contributed to the mean",
            )
    # Belt to that braces: the count above is derived from the scores that
    # survived, this is the raw incident count. A metric that is scored but not
    # listed in `metrics` below has no denominator check at all, and would
    # otherwise fail silently exactly as faithfulness did.
    judge_failures = payload.get("n_judge_failures")
    if judge_failures is None:
        result.add(
            "judge failures",
            False,
            "not measured — this run predates the check; re-run run_evals.py",
        )
    else:
        result.add(
            "judge failures",
            judge_failures <= spec["max_judge_failures"],
            f"{judge_failures}, max {spec['max_judge_failures']}",
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
    # The denominator of the check above, and the ids it excluded.
    #
    # An answer the grounding verifier refused makes no claims, so it has
    # nothing to cite and is not counted. That exclusion is correct and it is
    # also a hole: with every answer refused, `unmarked` is empty and the
    # citation check passes over nothing at all. Same shape as the per-metric
    # population above — a shrinking denominator hiding behind a clean count.
    #
    # Refusals are not failures. `max_answers_refused_as_ungrounded` is sized
    # from what the suite actually does: 0 or 1 across five scored runs, always
    # the same question (G11), where the verifier disagrees with itself run to
    # run. A rising count is not a broken gate, it is answers getting worse.
    citable = payload.get("n_citable")
    refused = payload.get("answers_refused_as_ungrounded")
    if citable is None or refused is None:
        result.add(
            "citation population",
            False,
            "the citation check does not state its denominator — this run "
            "predates it; re-run run_evals.py",
        )
    else:
        max_refused = spec["max_answers_refused_as_ungrounded"]
        result.add(
            "citation population",
            citable + len(refused) == expected and len(refused) <= max_refused,
            f"{citable} answers citable, {len(refused)} refused as ungrounded "
            f"(max {max_refused})" + (f" — {', '.join(refused)}" if refused else ""),
        )
    summary = payload.get("summary", {})
    for metric, mspec in spec["metrics"].items():
        _check_metric(result, metric, summary.get(metric), mspec)
    return result


def gate_retrieval(summary: dict[str, Any]) -> GateResult:
    mode = summary.get("mode", "unknown")
    result = GateResult(f"retrieval:{mode}")
    # `--smoke` and `--questions` are the two ways to score fewer than the full
    # set. Both are refused outright rather than scored short, so neither can be
    # promoted over a full run. `subset` absent means the run predates the field;
    # that is safe to read as "full run" only because the population check below
    # catches a short run on its own — there are 38 scorable questions and any
    # strict subset of them lands under 38.
    if summary.get("smoke") or summary.get("subset"):
        result.add("population", False, "subset run cannot be gated against a full-run baseline")
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
