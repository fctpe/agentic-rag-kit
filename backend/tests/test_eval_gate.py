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
    def test_ragas(self):
        assert gate_ragas(_load("ragas.json"), partial=False).passed

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
