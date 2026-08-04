"""The gate must pass the committed baselines and fail a real regression.

These run offline against the JSON already committed under evals/results/, so
CI exercises the gate logic on every push without a provider key. A gate nobody
tests is a gate that silently stops gating.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

_EVALS = Path(__file__).resolve().parents[2] / "evals"
sys.path.insert(0, str(_EVALS))

from gate import gate_ragas, gate_redteam, gate_retrieval  # noqa: E402

RESULTS = _EVALS / "results"

# The canonical artifacts the README's tables are read from. Every one of them is
# now written only by evals/promote.py, so every one of them carries provenance —
# `retrieval_vector.json` is gone, a 12 July file that no table was built from
# and that the README's own reproduction command nonetheless pointed at.
RETRIEVAL_ARTIFACTS = [
    "retrieval_hybrid.json",
    "retrieval_vector_only.json",
    "retrieval_text_only.json",
]
PROMOTED_ARTIFACTS = ["ragas.json", *RETRIEVAL_ARTIFACTS]


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def _thresholds() -> dict:
    from gate import load_thresholds  # noqa: PLC0415

    return load_thresholds()


def _summary_at_declared_baseline(mode: str) -> dict:
    """A synthetic run that measured exactly what thresholds.yaml declares.

    Built from the threshold file rather than from a committed artifact on
    purpose: it is the payload every drift test below mutates away from, so it
    has to be clean by construction, not clean by luck.
    """
    spec = _thresholds()["retrieval"]["modes"][mode]["metrics"]
    summary = {
        "mode": mode,
        "k": 6,
        "regulation_filter": "on",
        "smoke": False,
        "subset": False,
        "n_questions": _thresholds()["retrieval"]["expect_n_questions"],
    }
    summary.update({metric: mspec["baseline"] for metric, mspec in spec.items()})
    return summary


def _failed(result) -> list[str]:
    return [c.name for c in result.checks if not c.passed]


def _complete_ragas_payload() -> dict:
    """The committed run, with the fields a genuinely complete run would carry.

    `ragas.json` itself is a 23-of-38 faithfulness mean and now fails the gate
    on exactly that (see `TestPerMetricPopulation`). Tests that mutate ONE check
    and assert that check is what failed need a payload that is otherwise clean
    — on the raw artifact every one of them would pass on the artifact's own
    defect and prove nothing about the check it names.
    """
    payload = copy.deepcopy(_load("ragas.json"))
    payload["answers_without_inline_citation"] = []
    payload["n_contributing"] = dict.fromkeys(payload["summary"], payload["n_scored"])
    payload["n_judge_failures"] = 0
    payload["n_citable"] = payload["n_scored"]
    payload["answers_refused_as_ungrounded"] = []
    return payload


class TestCommittedBaselinesPass:
    def test_ragas(self):
        assert gate_ragas(_load("ragas.json"), partial=False).passed

    @pytest.mark.parametrize("name", RETRIEVAL_ARTIFACTS)
    def test_retrieval(self, name):
        """Floors, ceilings, population — and now the declared baseline.

        A `(drift)` failure here does not mean retrieval got worse. It means the
        number in `evals/results/` and the number `thresholds.yaml` declares are
        not the same number, which is the state commit 88131af left the hybrid
        MRR in and which nothing could see. Clearing it is two deliberate steps:
        write the measured value into `thresholds.yaml`, then promote the run
        that measured it.
        """
        assert _failed(gate_retrieval(_load(name)["summary"])) == []

    def test_redteam(self):
        assert gate_redteam(_load("redteam_final_14of14.json")["summary"], partial=False).passed

    def test_the_committed_regression_artifact_is_caught(self):
        """13/14 is kept in the repo precisely as the run that must not pass."""
        payload = _load("redteam_run1_13of14.json")
        assert not gate_redteam(payload["summary"], partial=False).passed


class TestRegressionsFail:
    def test_faithfulness_collapse(self):
        payload = _complete_ragas_payload()
        payload["summary"]["faithfulness"] = 0.60
        assert _failed(gate_ragas(payload, partial=False)) == ["faithfulness"]

    def test_unscored_metric_is_a_failure_not_a_skip(self):
        payload = _complete_ragas_payload()
        payload["summary"]["faithfulness"] = None
        assert _failed(gate_ragas(payload, partial=False)) == ["faithfulness"]

    def test_a_run_level_shrink_fails_even_with_better_scores(self):
        """Fewer questions asked, higher mean, looks like an improvement.

        This guards the run-level count only — the number of questions that
        reached the judge at all. It is NOT the shrink that actually shipped:
        that one never touched `n_scored`, which is why it needed its own
        class below.
        """
        payload = _complete_ragas_payload()
        payload["n_scored"] = 12
        for metric in payload["summary"]:
            payload["summary"][metric] = 1.0
        assert _failed(gate_ragas(payload, partial=False)) == ["population"]

    def test_chat_failures_fail(self):
        payload = _complete_ragas_payload()
        payload["n_chat_failures"] = 1
        assert _failed(gate_ragas(payload, partial=False)) == ["chat failures"]

    def test_retrieval_mrr_regression(self):
        summary = copy.deepcopy(_load("retrieval_hybrid.json")["summary"])
        # The measured cost of switching the FTS arm to OR semantics.
        summary["mrr"] = 0.772
        assert not gate_retrieval(summary).passed

    def test_text_only_arm_scoring_well_trips_the_ceiling(self):
        """Not a win — it means the query semantics changed underneath the
        committed hybrid-vs-text ablation."""
        summary = copy.deepcopy(_load("retrieval_text_only.json")["summary"])
        summary["hit_rate_at_k"] = 0.85
        assert not gate_retrieval(summary).passed

    def test_one_lost_refusal_fails(self):
        assert not gate_redteam({"passed": 13, "failed": 1, "total": 14}, partial=False).passed


class TestTheDeclaredBaselineIsActuallyChecked:
    """The baseline was printed next to the floor and compared against nothing.

    Commit 88131af moved the committed hybrid MRR from 0.8912 to 0.875 — a side
    effect of an eval run that overwrote the committed file in place, never
    promoted, never noticed, while README.md went on quoting 0.891. It cleared
    the 0.87 floor comfortably, so the gate said PASS and printed the baseline it
    had just walked past.

    Retrieval can carry `max_drift: 0.0` because it is deterministic: chunk text,
    chunk vectors and query vectors are all committed and every ORDER BY is a
    total order, so the number cannot move on its own. RAGAS cannot — two runs of
    one corpus scored faithfulness 0.9176 and 0.9316 — and deliberately has no
    drift check at all.
    """

    def test_a_run_that_reproduces_the_declared_baseline_passes(self):
        """Negative control. A drift check that reds a faithful re-run is a check
        that gets deleted, and then the baseline is decoration again."""
        result = gate_retrieval(_summary_at_declared_baseline("hybrid"))
        assert _failed(result) == []
        # And it really did run: a check that silently vanished would also pass.
        assert any(c.name.endswith("(drift)") for c in result.checks)

    def test_the_move_that_shipped_is_caught_and_the_floor_could_not_catch_it(self):
        """0.8912 -> 0.875, exactly as committed. Naming the check matters: the
        floor is 0.87, so `not .passed` alone would not prove which rule fired,
        and the whole point is that the floor did not."""
        summary = _summary_at_declared_baseline("hybrid")
        summary["mrr"] = 0.875
        result = gate_retrieval(summary)
        assert _failed(result) == ["mrr (drift)"]
        assert (
            summary["mrr"]
            >= _thresholds()["retrieval"]["modes"]["hybrid"]["metrics"]["mrr"]["floor"]
        ), "if the floor now catches this, the test is no longer about the drift check"

    def test_an_unexplained_improvement_fails_too(self):
        """Two-sided, because a deterministic number that got better on its own
        is the same event as one that got worse: something changed and nobody
        said so. The answer is to declare it and promote, not to absorb it."""
        summary = _summary_at_declared_baseline("vector_only")
        summary["mrr"] = summary["mrr"] + 0.05
        assert _failed(gate_retrieval(summary)) == ["mrr (drift)"]

    def test_the_negative_result_is_pinned_the_same_way(self):
        """text_only is gated by a ceiling, and a ceiling has the same hole: the
        arm could halve its score without tripping 0.20."""
        summary = _summary_at_declared_baseline("text_only")
        summary["hit_rate_at_k"] = 0.0
        assert _failed(gate_retrieval(summary)) == ["hit_rate_at_k (drift)"]

    def test_ragas_deliberately_carries_no_drift_check(self):
        """Not an oversight. A band wide enough to survive 0.9176 vs 0.9316 on
        one corpus gates nothing; a narrow one reds the build on a judge draw."""
        result = gate_ragas(_complete_ragas_payload(), partial=False)
        assert [c.name for c in result.checks if c.name.endswith("(drift)")] == []

    def test_a_drift_tolerance_with_nothing_to_compare_against_fails_closed(self):
        """`max_drift` with no `baseline` is a check switched on over an empty
        slot. Skipping it would be indistinguishable from it passing."""
        from gate import GateResult, _check_metric  # noqa: PLC0415

        result = GateResult("t")
        _check_metric(result, "mrr", 0.9, {"floor": 0.5, "max_drift": 0.0})
        assert _failed(result) == ["mrr (drift)"]


class TestInlineCitationMarkers:
    """A [n] marker is what links a sentence to the source it came from; the
    frontend can do nothing with a prose reference like "(AI Act, Art. 50(1))".
    None of the four RAGAS metrics can tell the two apart — an answer that cites
    entirely in prose scores exactly as well — so without this check the format
    was only ever held by the prompt asking nicely.
    """

    def test_a_run_with_every_answer_marked_passes(self):
        assert gate_ragas(_complete_ragas_payload(), partial=False).passed

    def test_a_single_unmarked_answer_fails(self):
        payload = _complete_ragas_payload()
        payload["answers_without_inline_citation"] = ["A02"]
        assert _failed(gate_ragas(payload, partial=False)) == ["inline citations"]

    def test_an_unmeasured_run_is_a_failure_not_a_skip(self):
        """The absent field is the state every pre-fix artifact is in. Treating
        it as "no answers missing" would pass all of them."""
        payload = _complete_ragas_payload()
        payload.pop("answers_without_inline_citation", None)
        assert _failed(gate_ragas(payload, partial=False)) == ["inline citations"]


class TestPerMetricPopulation:
    """The shrink that actually shipped, and the reason the run-level count
    could not see it.

    A judge call that raised was turned into `None`, and `None` was filtered out
    of that metric's mean. The question left one denominator and stayed in the
    other three, so `n_scored` — which counts questions that got an *answer* —
    never moved. The committed `ragas.json` reports 38 and publishes a
    faithfulness averaged over 23.

    Worse than the arithmetic: the loss was a length filter, not noise. The
    judge ran out of output tokens on long answers, so what dropped out were the
    multi-hop cross-regulation questions — the hardest ones. The surviving
    population is biased easy and the published mean is biased high, which is
    the direction that flatters. Across the three committed runs, fewer scored
    consistently meant a higher mean.
    """

    def test_a_metric_averaged_over_fewer_questions_fails(self):
        payload = _complete_ragas_payload()
        payload["n_contributing"]["faithfulness"] = 23
        assert _failed(gate_ragas(payload, partial=False)) == ["faithfulness population"]

    def test_it_fails_even_when_every_mean_is_above_its_floor(self):
        """The whole point. A shrunken population does not lower the mean — it
        raises it, because the questions that fall out are the hard ones. A gate
        that only reads the means would call this an improvement."""
        payload = _complete_ragas_payload()
        payload["n_contributing"]["faithfulness"] = 23
        for metric in payload["summary"]:
            payload["summary"][metric] = 0.99
        result = gate_ragas(payload, partial=False)
        assert not result.passed
        assert _failed(result) == ["faithfulness population"]

    def test_a_complete_run_passes(self):
        """Negative control. A population check that reds a healthy run is a
        check that gets switched off, and then nothing is checked."""
        assert gate_ragas(_complete_ragas_payload(), partial=False).passed

    def test_an_unmeasured_run_is_a_failure_not_a_skip(self):
        """Absent means the run predates the fix — which is every artifact that
        carries the defect. Reading it as "nothing missing" passes all of them."""
        payload = _complete_ragas_payload()
        payload.pop("n_contributing")
        assert "per-metric population" in _failed(gate_ragas(payload, partial=False))

    def test_a_recorded_judge_failure_fails_the_run(self):
        """Independent of the counts above: `n_contributing` is derived from the
        scores that survived, this is the raw incident count. A metric that is
        scored but absent from `thresholds.yaml` has no denominator check at all,
        and would go silent exactly as faithfulness did."""
        payload = _complete_ragas_payload()
        payload["n_judge_failures"] = 1
        assert _failed(gate_ragas(payload, partial=False)) == ["judge failures"]

    def test_the_real_contaminated_run_is_caught_by_these_checks_specifically(self):
        """These checks are not hypothetical.

        `ragas_run_23of38.json` is the run that shipped: it reports
        `n_scored: 38` and publishes a faithfulness averaged over 23, because
        the judge ran out of output tokens on the fifteen longest answers. It is
        kept in the repo for the same reason `redteam_run1_13of14.json` is —
        as the artifact these checks are proven against, not as a result.

        Naming the two checks matters. That file predates the inline-citation
        field as well, so a bare `not .passed` would stay green with the
        population checks deleted, and would be measuring the wrong defect.
        """
        payload = _load("ragas_run_23of38.json")
        assert payload["n_scored"] == 38
        scored = sum(1 for s in payload["samples"] if s["scores"].get("faithfulness") is not None)
        assert scored == 23, "the fixture stopped being the contaminated run"

        failed = _failed(gate_ragas(payload, partial=False))
        assert "per-metric population" in failed
        assert "judge failures" in failed


class TestCitationPopulation:
    """A citation check that passes over nothing.

    An answer the grounding verifier refused makes no claims and cites nothing,
    so it is excluded from the marker check — counting it would make the
    fail-closed path look like a formatting defect, and the cheapest way to
    green that would be to weaken the verifier.

    The exclusion needs its own denominator for the same reason the metric means
    do: without it, a run where every answer was refused reports zero missing
    markers and passes, having checked nothing.
    """

    def test_a_complete_run_passes(self):
        payload = _complete_ragas_payload()
        assert gate_ragas(payload, partial=False).passed

    def test_a_refusal_does_not_count_as_a_missing_marker(self):
        """The false positive this closes. G11 is a real answer the verifier
        refused in three of five scored runs; it was reported as an uncited
        answer and failed the gate."""
        payload = _complete_ragas_payload()
        payload["n_citable"] = 37
        payload["answers_refused_as_ungrounded"] = ["G11"]
        assert gate_ragas(payload, partial=False).passed

    def test_a_shrunken_citable_set_fails(self):
        """Zero missing markers out of two answers is not a pass."""
        payload = _complete_ragas_payload()
        payload["n_citable"] = 2
        payload["answers_refused_as_ungrounded"] = []
        assert _failed(gate_ragas(payload, partial=False)) == ["citation population"]

    def test_too_many_refusals_fail_even_though_refusing_is_correct(self):
        payload = _complete_ragas_payload()
        payload["n_citable"] = 33
        payload["answers_refused_as_ungrounded"] = ["G11", "G08", "A01", "A02", "X01"]
        assert _failed(gate_ragas(payload, partial=False)) == ["citation population"]

    def test_an_unmeasured_run_is_a_failure_not_a_skip(self):
        payload = _complete_ragas_payload()
        payload.pop("n_citable")
        assert "citation population" in _failed(gate_ragas(payload, partial=False))


class TestSubsetRunsCannotClaimAGate:
    def test_ragas_subset(self):
        assert not gate_ragas(_load("ragas.json"), partial=True).passed

    def test_retrieval_smoke(self):
        summary = _summary_at_declared_baseline("hybrid")
        summary["smoke"] = True
        assert not gate_retrieval(summary).passed

    def test_retrieval_question_subset(self):
        """`--questions A01 G03` is the other way to score fewer than the set.
        The population count catches it on its own; this refuses it by name so a
        subset can never reach promote.py's gate at all."""
        summary = _summary_at_declared_baseline("hybrid")
        summary["subset"] = True
        assert not gate_retrieval(summary).passed

    def test_a_full_run_is_not_mistaken_for_a_subset(self):
        """Negative control for both refusals above."""
        assert gate_retrieval(_summary_at_declared_baseline("hybrid")).passed

    def test_unknown_retrieval_mode_fails_closed(self):
        summary = _summary_at_declared_baseline("hybrid")
        summary["mode"] = "some_new_backend"
        assert not gate_retrieval(summary).passed


class TestProvenanceStampsStillResolve:
    """`ragas.json` shipped `promoted_git_sha: 79f3c3c`, and a later rebase
    orphaned that commit — the field whose entire job is provenance pointed at
    something `git merge-base --is-ancestor` rejected. Nothing noticed, because
    nothing looked.

    This looks. It fails on an orphaned stamp and skips (loudly) when the
    checkout genuinely cannot answer, which is what a shallow CI clone is.

    It covers the retrieval artifacts too, and did not used to. They had no
    stamps to check because nothing stamped them: `run_retrieval_eval.py` wrote
    `retrieval_<mode>.json` in place on every run, so the file the README quotes
    could not say which run produced it or from what commit. Both halves of that
    are fixed — the runner writes a timestamped file, `promote.py` writes the
    canonical one — and this is the half that stays honest afterwards.
    """

    @pytest.mark.parametrize("name", PROMOTED_ARTIFACTS)
    def test_stamp_is_reachable_from_head(self, name: str) -> None:
        from promote import stamp_status  # noqa: PLC0415

        payload = _load(name)
        sha = payload.get("promoted_git_sha")
        assert sha, (
            f"{name} carries no promoted_git_sha — it was written in place by a "
            f"run rather than promoted. Re-run the suite and "
            f"`promote.py {name.removesuffix('.json')}`."
        )

        verdict, detail = stamp_status(sha)
        if verdict == "unverifiable":
            pytest.skip(f"cannot verify {name} provenance here: {detail}")
        assert verdict == "ok", f"{name} provenance is stale — {detail}"

    def test_a_fabricated_sha_is_reported_as_orphaned(self) -> None:
        """Negative control: without it, `verdict == 'ok'` above could be
        satisfied by a checker that says ok to everything."""
        from promote import stamp_status  # noqa: PLC0415

        verdict, _ = stamp_status("0" * 40)
        assert verdict in {"orphaned", "unverifiable"}
        if verdict == "unverifiable":
            pytest.skip("no git worktree available to prove the negative case")

    @pytest.mark.parametrize("name", PROMOTED_ARTIFACTS)
    def test_short_shas_are_no_longer_written(self, name: str) -> None:
        """A 7-char stamp is the shape the orphaned one had; full shas also stay
        unambiguous as the repo grows."""
        sha = _load(name).get("promoted_git_sha")
        assert sha and len(sha) == 40, f"{name}: expected a full 40-char sha, got {sha!r}"
