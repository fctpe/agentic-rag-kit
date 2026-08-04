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


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


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
    return payload


class TestCommittedBaselinesPass:
    def test_ragas_is_no_longer_among_them(self):
        """The committed ragas run does not pass its own gate, and should not.

        Its faithfulness figure is a mean over 23 of the 38 questions — the
        judge could not complete the other 15 — while the file reports
        `n_scored: 38`. It stays in the repo as the artifact the new checks are
        proven against (`TestPerMetricPopulation` below) until a full run
        replaces it. The retrieval and redteam baselines are unaffected: no LLM
        judge sits between those runs and their numbers.
        """
        assert not gate_ragas(_load("ragas.json"), partial=False).passed

    @pytest.mark.parametrize(
        "name", ["retrieval_hybrid.json", "retrieval_vector.json", "retrieval_text_only.json"]
    )
    def test_retrieval(self, name):
        assert gate_retrieval(_load(name)["summary"]).passed

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

    def test_the_committed_artifact_is_caught_by_these_checks_specifically(self):
        """These checks are not hypothetical — run them against the real
        23-of-38 file and it fails.

        Naming the two checks matters. `ragas.json` predates the inline-citation
        field as well, so a bare `not .passed` here would stay green with the
        population checks deleted, and would be measuring the wrong defect.
        """
        failed = _failed(gate_ragas(_load("ragas.json"), partial=False))
        assert "per-metric population" in failed
        assert "judge failures" in failed


class TestSubsetRunsCannotClaimAGate:
    def test_ragas_subset(self):
        assert not gate_ragas(_load("ragas.json"), partial=True).passed

    def test_retrieval_smoke(self):
        summary = copy.deepcopy(_load("retrieval_hybrid.json")["summary"])
        summary["smoke"] = True
        assert not gate_retrieval(summary).passed

    def test_unknown_retrieval_mode_fails_closed(self):
        summary = copy.deepcopy(_load("retrieval_hybrid.json")["summary"])
        summary["mode"] = "some_new_backend"
        assert not gate_retrieval(summary).passed


class TestProvenanceStampsStillResolve:
    """`ragas.json` shipped `promoted_git_sha: 79f3c3c`, and a later rebase
    orphaned that commit — the field whose entire job is provenance pointed at
    something `git merge-base --is-ancestor` rejected. Nothing noticed, because
    nothing looked.

    This looks. It fails on an orphaned stamp and skips (loudly) when the
    checkout genuinely cannot answer, which is what a shallow CI clone is.
    """

    @pytest.mark.parametrize("name", ["ragas.json"])
    def test_stamp_is_reachable_from_head(self, name: str) -> None:
        from promote import stamp_status  # noqa: PLC0415

        payload = _load(name)
        sha = payload.get("promoted_git_sha")
        assert sha, f"{name} carries no promoted_git_sha"

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

    def test_short_shas_are_no_longer_written(self) -> None:
        """A 7-char stamp is the shape the orphaned one had; full shas also stay
        unambiguous as the repo grows."""
        sha = _load("ragas.json").get("promoted_git_sha")
        assert sha and len(sha) == 40, f"expected a full 40-char sha, got {sha!r}"
