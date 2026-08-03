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


class TestCommittedBaselinesPass:
    def test_ragas_scores(self):
        """Every scored threshold — deliberately not the whole gate.

        A run recorded before the inline-citation check existed carries no
        `answers_without_inline_citation` field, and the gate fails closed on
        that rather than assuming a clean one. The committed artifact is such a
        run (18 of its 38 answers carry no [n] marker at all, which is how the
        regression went unseen), so asserting `.passed` here would mean either
        defaulting the absent field or dropping the check. The citation check is
        exercised on its own below; this test holds the numbers.
        """
        result = gate_ragas(_load("ragas.json"), partial=False)
        scored = [c for c in result.checks if c.name != "inline citations"]
        assert all(c.passed for c in scored)

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
        payload = copy.deepcopy(_load("ragas.json"))
        payload["summary"]["faithfulness"] = 0.60
        assert not gate_ragas(payload, partial=False).passed

    def test_unscored_metric_is_a_failure_not_a_skip(self):
        payload = copy.deepcopy(_load("ragas.json"))
        payload["summary"]["faithfulness"] = None
        assert not gate_ragas(payload, partial=False).passed

    def test_shrinking_sample_set_fails_even_with_better_scores(self):
        """The failure mode a naive threshold check misses: score fewer
        questions, post a higher mean, look like an improvement."""
        payload = copy.deepcopy(_load("ragas.json"))
        payload["n_scored"] = 12
        for metric in payload["summary"]:
            payload["summary"][metric] = 1.0
        assert not gate_ragas(payload, partial=False).passed

    def test_chat_failures_fail(self):
        payload = copy.deepcopy(_load("ragas.json"))
        payload["n_chat_failures"] = 1
        assert not gate_ragas(payload, partial=False).passed

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
        payload = copy.deepcopy(_load("ragas.json"))
        payload["answers_without_inline_citation"] = []
        assert gate_ragas(payload, partial=False).passed

    def test_a_single_unmarked_answer_fails(self):
        payload = copy.deepcopy(_load("ragas.json"))
        payload["answers_without_inline_citation"] = ["A02"]
        assert not gate_ragas(payload, partial=False).passed

    def test_an_unmeasured_run_is_a_failure_not_a_skip(self):
        """The absent field is the state every pre-fix artifact is in. Treating
        it as "no answers missing" would pass all of them."""
        payload = copy.deepcopy(_load("ragas.json"))
        payload.pop("answers_without_inline_citation", None)
        assert not gate_ragas(payload, partial=False).passed


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
