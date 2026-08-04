"""Promotion discipline: the rules that decide which run becomes the number.

`evals/promote.py` is the only thing allowed to write the files README.md quotes.
These run offline against a throwaway git repository, so CI exercises the real
promotion path — argv, gate, git, stamp — on every push with no provider key.

The tests are written against the way it actually broke. `run_retrieval_eval.py`
used to write `evals/results/retrieval_<mode>.json` in place on every run, and
commit 88131af moved the committed hybrid MRR from 0.8912 to 0.875 that way:
nobody promoted it, nobody could tell, and the README kept quoting 0.891.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_EVALS = Path(__file__).resolve().parents[2] / "evals"
sys.path.insert(0, str(_EVALS))

import promote  # noqa: E402
from gate import load_thresholds  # noqa: E402

RESULTS = _EVALS / "results"


def _full_run_summary(mode: str = "hybrid") -> dict:
    """A run that measured exactly what thresholds.yaml declares, so it passes."""
    spec = load_thresholds()["retrieval"]
    summary = {
        "mode": mode,
        "k": 6,
        "regulation_filter": "on",
        "query_embeddings": "committed" if mode != "text_only" else "none",
        "smoke": False,
        "subset": False,
        "n_questions": spec["expect_n_questions"],
    }
    summary.update({m: s["baseline"] for m, s in spec["modes"][mode]["metrics"].items()})
    return summary


def _payload(mode: str = "hybrid", **overrides) -> dict:
    summary = _full_run_summary(mode)
    summary.update(overrides)
    return {"summary": summary, "questions": []}


class TestTheRunnerNeverWritesTheCanonicalFile:
    """The defect itself. `run_retrieval_eval.py` wrote
    `evals/results/retrieval_<mode>.json` in place, unconditionally, on every
    run — so the number README.md quotes could be changed by anyone who typed
    `make eval-retrieval`, with nothing in the file recording that it had been."""

    @pytest.mark.parametrize("mode", ["hybrid", "vector_only", "text_only"])
    @pytest.mark.parametrize("use_filter", [True, False])
    def test_a_run_lands_on_a_timestamped_path(self, mode, use_filter):
        from run_retrieval_eval import output_path  # noqa: PLC0415

        path = output_path(mode, use_filter, "20260804T100000Z")
        assert path.name != f"retrieval_{mode}.json"
        assert path.name.endswith("_20260804T100000Z.json")
        # And it is a path promote.py will actually consider — a timestamped
        # name in a shape the promoter cannot match is not promotion discipline,
        # it is a file nobody can publish.
        expected_candidate = use_filter
        assert bool(promote.run_pattern(f"retrieval_{mode}").match(path.name)) is expected_candidate

    def test_the_scratch_redirect_still_works(self):
        """Negative control: CI's two-ingest determinism diff runs the eval twice
        with `--json`, and a path rule that ignored the flag would break it."""
        from run_retrieval_eval import output_path  # noqa: PLC0415

        scratch = Path("eval-1.json")
        assert output_path("hybrid", True, "20260804T100000Z", scratch) == scratch

    def test_the_determinism_job_does_not_enforce_the_baseline(self):
        """CI ingests the fixture twice and diffs two retrieval runs against each
        other. That step asks whether two runs agree, not whether they agree with
        `thresholds.yaml` — so it passes `--no-gate`, and without it a PR that
        deliberately moves retrieval would fail there instead of at promote.py.

        The flag has to exist and the workflow has to use it; either half alone
        is the failure, so both are asserted.
        """
        from run_retrieval_eval import build_parser  # noqa: PLC0415

        parser = build_parser()
        assert parser.parse_args(["--no-gate"]).no_gate is True
        assert parser.parse_args([]).no_gate is False  # and it is opt-in
        workflow = (_EVALS.parent / ".github/workflows/ci.yml").read_text()
        determinism_runs = [
            line for line in workflow.splitlines() if "--json eval-" in line and "hybrid" in line
        ]
        assert determinism_runs, "the two-ingest determinism step is gone"
        assert all("--no-gate" in line for line in determinism_runs)


class TestOnlyTimestampedRunsAreCandidates:
    """`<suite>_*.json` matched more than the runs, and lexical order made the
    extras win: digits sort before letters, so `redteam_run1_13of14.json` — the
    committed 13-of-14 regression artifact — came out newest and every real
    redteam run was refused as "older" than it."""

    def test_a_real_run_matches(self):
        """Negative control first: a pattern that matches nothing would satisfy
        every exclusion below."""
        assert promote.run_pattern("redteam").match("redteam_20260804T075854Z.json")
        assert promote.run_pattern("retrieval_hybrid").match(
            "retrieval_hybrid_20260804T075854Z.json"
        )

    @pytest.mark.parametrize(
        ("suite", "name"),
        [
            ("redteam", "redteam_run1_13of14.json"),
            ("redteam", "redteam_final_14of14.json"),
            ("redteam", "redteam.json"),
            # A different ablation wearing a prefix of the same name. Promoting
            # an unfiltered run into retrieval_hybrid.json would replace the
            # README's hybrid row with a number measured over the full corpus.
            ("retrieval_hybrid", "retrieval_hybrid_nofilter_20260804T075854Z.json"),
            ("retrieval_hybrid", "retrieval_hybrid.json"),
            ("retrieval_vector_only", "retrieval_vector_only_smoke.json"),
        ],
    )
    def test_hand_named_files_are_not_runs(self, suite, name):
        assert not promote.run_pattern(suite).match(name)

    def test_the_newest_redteam_candidate_in_the_repo_is_a_real_run(self):
        """Against the actual results directory, not a fixture: this is the
        state the glob left the repo in."""
        candidates = promote._candidates("redteam")
        assert candidates, "no timestamped redteam runs committed"
        assert promote.run_pattern("redteam").match(candidates[-1].name)

    def test_every_retrieval_mode_is_registered(self):
        for mode in ("hybrid", "vector_only", "text_only"):
            assert f"retrieval_{mode}" in promote.SUITES


class TestRetrievalRunsMustMatchTheArtifactTheyReplace:
    """A filename is not a measurement. `--k 20` writes
    `retrieval_hybrid_<ts>.json` and reports hit@20, which is a different
    quantity under the same label as the README's hit@6 column."""

    def test_a_matching_run_is_accepted(self):
        """Negative control: a shape check that rejects everything is a shape
        check nobody can promote past."""
        target = RESULTS / "retrieval_hybrid.json"
        assert (
            promote.retrieval_shape_mismatch("retrieval_hybrid", _full_run_summary(), target)
            is None
        )

    def test_a_different_k_is_refused(self):
        target = RESULTS / "retrieval_hybrid.json"
        summary = _full_run_summary() | {"k": 20}
        assert "k=20" in promote.retrieval_shape_mismatch("retrieval_hybrid", summary, target)

    def test_an_unfiltered_run_is_refused(self):
        target = RESULTS / "retrieval_hybrid.json"
        summary = _full_run_summary() | {"regulation_filter": "off"}
        assert promote.retrieval_shape_mismatch("retrieval_hybrid", summary, target) is not None

    def test_a_run_of_the_wrong_mode_is_refused(self):
        target = RESULTS / "retrieval_hybrid.json"
        summary = _full_run_summary("vector_only")
        assert "vector_only" in promote.retrieval_shape_mismatch(
            "retrieval_hybrid", summary, target
        )


class TestBlockingDirt:
    """The dirty-worktree refusal exists so a stamp cannot name a commit that is
    not what produced the numbers. Uncommitted code invalidates that claim.
    Other files under evals/results/ cannot — they were outputs, not inputs —
    and refusing on them makes promoting the three retrieval modes in sequence
    impossible, because the first promotion dirties results/ and the second is
    then refused for the damage the first one did."""

    def test_uncommitted_code_blocks(self):
        assert promote.blocking_dirt(" M backend/app/retrieval/hybrid.py") == [
            "backend/app/retrieval/hybrid.py"
        ]

    def test_an_edited_threshold_file_blocks(self):
        assert promote.blocking_dirt(" M evals/thresholds.yaml") == ["evals/thresholds.yaml"]

    def test_other_result_files_do_not_block(self):
        status = (
            "?? evals/results/ragas_20260804T081758Z.json\n M evals/results/retrieval_hybrid.json"
        )
        assert promote.blocking_dirt(status) == []

    def test_a_rename_is_read_at_its_destination(self):
        assert promote.blocking_dirt("R  evals/old.py -> backend/app/new.py") == [
            "backend/app/new.py"
        ]


class TestPromotionEndToEnd:
    """The whole chain in a throwaway repository: newest-only, gate, dirty
    worktree, full sha, stamp."""

    @pytest.fixture
    def repo(self, tmp_path, monkeypatch):
        root = tmp_path / "repo"
        results = root / "evals" / "results"
        results.mkdir(parents=True)
        run = lambda *a: subprocess.run(  # noqa: E731
            ["git", *a], cwd=root, check=True, capture_output=True, text=True
        )
        run("init", "-q", "-b", "main")
        run("config", "user.email", "t@example.invalid")
        run("config", "user.name", "t")
        (root / "seed.txt").write_text("seed\n")
        # Tracked, so `git status --porcelain` reports the run files inside it
        # individually. Left untracked, git collapses the whole tree to a single
        # `evals/` line and the dirt filter would be tested against a path shape
        # it never sees in the real repository.
        (results / ".gitkeep").write_text("")
        run("add", "seed.txt", "evals/results/.gitkeep")
        run("commit", "-qm", "seed")
        monkeypatch.setattr(promote, "EVALS_DIR", root / "evals")
        monkeypatch.setattr(promote, "RESULTS_DIR", results)
        return root, results, run

    def _write(self, results: Path, name: str, payload: dict) -> Path:
        path = results / name
        path.write_text(json.dumps(payload))
        return path

    def _promote(self, monkeypatch, *argv) -> int:
        monkeypatch.setattr(sys, "argv", ["promote.py", *argv])
        return promote.main()

    def test_the_newest_passing_run_is_stamped_into_the_canonical_file(
        self, repo, monkeypatch, capsys
    ):
        root, results, _ = repo
        self._write(results, "retrieval_hybrid_20260804T090000Z.json", _payload())
        self._write(results, "retrieval_hybrid_20260804T100000Z.json", _payload())

        assert self._promote(monkeypatch, "retrieval_hybrid") == 0

        promoted = json.loads((results / "retrieval_hybrid.json").read_text())
        assert promoted["promoted_from"] == "retrieval_hybrid_20260804T100000Z.json"
        assert len(promoted["promoted_git_sha"]) == 40
        assert promoted["promoted_at"]
        assert promoted["summary"]["mrr"] == _full_run_summary()["mrr"]

        # And the result satisfies every rule test_eval_gate.py applies to a
        # committed artifact. That is the loop closing: the three retrieval files
        # fail those checks today precisely because nothing ever promoted them,
        # and this is what promotion produces instead.
        from gate import gate_retrieval  # noqa: PLC0415

        assert gate_retrieval(promoted["summary"]).passed
        verdict, detail = promote.stamp_status(promoted["promoted_git_sha"])
        assert verdict == "ok", detail

    def test_an_older_run_cannot_be_promoted_over_a_newer_one(self, repo, monkeypatch):
        _, results, _ = repo
        older = self._write(results, "retrieval_hybrid_20260804T090000Z.json", _payload())
        self._write(results, "retrieval_hybrid_20260804T100000Z.json", _payload(mrr=0.87))

        assert self._promote(monkeypatch, "retrieval_hybrid", "--run", str(older)) == 1
        assert not (results / "retrieval_hybrid.json").exists()

    def test_a_run_that_drifts_from_the_declared_baseline_is_refused(self, repo, monkeypatch):
        """The 88131af value exactly. It clears the 0.87 floor, which is why the
        floors alone let it through and why it needed the drift check."""
        _, results, _ = repo
        self._write(results, "retrieval_hybrid_20260804T100000Z.json", _payload(mrr=0.875))

        assert self._promote(monkeypatch, "retrieval_hybrid") == 1
        assert not (results / "retrieval_hybrid.json").exists()

    def test_a_smoke_run_is_refused(self, repo, monkeypatch):
        _, results, _ = repo
        self._write(results, "retrieval_hybrid_20260804T100000Z.json", _payload(smoke=True))
        assert self._promote(monkeypatch, "retrieval_hybrid") == 1

    def test_a_question_subset_is_refused(self, repo, monkeypatch):
        """`n_questions` is left at the full count on purpose. The population
        check would catch a short run anyway, so a payload that is also short
        would pass this test with the subset rule deleted."""
        _, results, _ = repo
        self._write(results, "retrieval_hybrid_20260804T100000Z.json", _payload(subset=True))
        assert self._promote(monkeypatch, "retrieval_hybrid") == 1

    def test_a_clean_worktree_with_only_run_output_promotes(self, repo, monkeypatch):
        """Negative control for the refusal below: the untracked run file itself
        must not be read as dirt, or nothing is ever promotable."""
        _, results, _ = repo
        self._write(results, "retrieval_hybrid_20260804T100000Z.json", _payload())
        assert self._promote(monkeypatch, "retrieval_hybrid") == 0

    def test_a_dirty_worktree_is_refused(self, repo, monkeypatch):
        root, results, _ = repo
        self._write(results, "retrieval_hybrid_20260804T100000Z.json", _payload())
        (root / "seed.txt").write_text("edited after the run\n")

        assert self._promote(monkeypatch, "retrieval_hybrid") == 1
        assert not (results / "retrieval_hybrid.json").exists()

    def test_three_modes_promote_in_sequence(self, repo, monkeypatch):
        """The regression the dirt filter exists for: after the first promotion
        the worktree is dirty under evals/results/, and a blanket refusal would
        make the second and third mode unpromotable."""
        _, results, _ = repo
        for mode in ("hybrid", "vector_only", "text_only"):
            self._write(results, f"retrieval_{mode}_20260804T100000Z.json", _payload(mode=mode))
        for mode in ("hybrid", "vector_only", "text_only"):
            assert self._promote(monkeypatch, f"retrieval_{mode}") == 0
            assert (results / f"retrieval_{mode}.json").exists()

    def test_an_unfiltered_ablation_cannot_take_the_hybrid_file(self, repo, monkeypatch):
        _, results, _ = repo
        self._write(
            results,
            "retrieval_hybrid_nofilter_20260804T100000Z.json",
            _payload(regulation_filter="off"),
        )
        # Not even a candidate: the suite has no runs at all.
        assert self._promote(monkeypatch, "retrieval_hybrid") == 1
        assert not (results / "retrieval_hybrid.json").exists()


class TestPorcelainLeadingColumnsSurvive:
    """`git status --porcelain` puts the state in the first two columns, so the
    leading space of an unstaged line is data, not padding.

    `_git` stripped it — and only from the FIRST line, because `.strip()` acts on
    the whole captured string. `blocking_dirt` then cut one character too many
    off that one path: "evals/results/x.json" arrived as "vals/results/x.json",
    missed the results/ exemption, and refused the promotion. Promoting the three
    retrieval modes in sequence failed at the second, since the first promotion
    is what dirties results/.
    """

    def test_git_hands_back_the_status_columns_intact(self):
        """The defect lives in `_git`, not in `blocking_dirt` — so this drives
        `_git` against a real dirty worktree rather than a hand-built string.

        A test that feeds `blocking_dirt` a literal is decoration here: it stays
        green with the bug restored, because the truncation happens before
        `blocking_dirt` ever sees the line.
        """
        import subprocess

        from promote import _git

        raw = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        if not raw.strip():
            pytest.skip("clean worktree — nothing to preserve the columns of")

        got = _git("status", "--porcelain")
        assert got is not None
        # Every line keeps its two status columns. `.strip()` on the captured
        # string ate the leading space of the first line only.
        for line, original in zip(got.splitlines(), raw.splitlines(), strict=True):
            assert line == original

    def test_an_unstaged_results_file_does_not_block(self):
        from promote import blocking_dirt

        status = " M evals/results/retrieval_hybrid.json\n M evals/results/ragas.json"
        assert blocking_dirt(status) == []

    def test_uncommitted_code_still_blocks(self):
        """Negative control. An exemption that swallows everything is not an
        exemption, and the stamp's whole claim is that a named commit produced
        the numbers."""
        from promote import blocking_dirt

        status = " M evals/results/retrieval_hybrid.json\n M backend/app/retrieval/hybrid.py"
        assert blocking_dirt(status) == ["backend/app/retrieval/hybrid.py"]

    def test_the_first_line_is_not_treated_differently_from_the_rest(self):
        """The defect only ever touched line one, so a fixture whose first line
        is exempt-by-accident would pass over it."""
        from promote import blocking_dirt

        first = " M evals/thresholds.yaml\n M evals/results/ragas.json"
        rest = " M evals/results/ragas.json\n M evals/thresholds.yaml"
        assert blocking_dirt(first) == blocking_dirt(rest) == ["evals/thresholds.yaml"]
