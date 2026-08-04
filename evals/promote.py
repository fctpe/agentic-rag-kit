#!/usr/bin/env python3
"""Promote a timestamped eval run to the committed result the README cites.

The suites write ``ragas_<timestamp>.json``, ``redteam_<timestamp>.json`` and
``retrieval_<mode>_<timestamp>.json``; the files committed under
``evals/results/`` are ``ragas.json``, ``redteam.json`` and
``retrieval_<mode>.json``. That gap used to be closed by hand with ``mv``, which
meant the committed numbers had no provable link to any particular run — and
nothing stopped a older, friendlier run being promoted over a newer, worse one.

This script closes it with three rules:

1. **Only the newest run.** Promoting an older run over a newer one is refused.
   If the latest run is worse, that is the number.
2. **Only a run that passes the gate.** A breaching run cannot become the
   committed baseline; fix the regression or move the threshold deliberately.
3. **Stamped.** The promoted file records the source filename, the commit that
   produced it, and when it was promoted.

    cd backend && uv run --group evals python ../evals/promote.py ragas
    cd backend && uv run --group evals python ../evals/promote.py retrieval_hybrid

Retrieval arrived here late. It had none of the above: ``run_retrieval_eval.py``
wrote ``retrieval_<mode>.json`` in place on every run, so commit 88131af moved
the committed hybrid MRR from 0.8912 to 0.875 as a side effect of a run nobody
had decided to publish, while the README kept quoting 0.891 — unfalsifiably, the
file carrying nothing that said where it came from. It is promoted through this
script rather than a second one of its own, because a second promotion path is a
second thing to forget.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVALS_DIR / "results"

sys.path.insert(0, str(EVALS_DIR))
from gate import GateResult, gate_ragas, gate_redteam, gate_retrieval  # noqa: E402

RETRIEVAL_MODES = ("hybrid", "vector_only", "text_only")
SUITES = ("ragas", "redteam", *(f"retrieval_{mode}" for mode in RETRIEVAL_MODES))

# The timestamp the three runners stamp into a filename: UTC, zero-padded.
_RUN_STAMP = r"\d{8}T\d{6}Z"

# The shape of the run that the committed artifact represents. A file called
# `retrieval_hybrid_<ts>.json` is not automatically a full-corpus k=6 filtered
# hybrid run — `--k 20` writes exactly that name and measures hit@20, which is a
# different quantity wearing the same label. Rather than hardcode the intended
# values in a second place, the run has to agree with the artifact it replaces
# on all of them.
_RETRIEVAL_SHAPE_KEYS = ("mode", "k", "regulation_filter")


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=EVALS_DIR,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None
    # `None` means the command failed; `""` means it succeeded and said nothing.
    # `merge-base --is-ancestor` and `cat-file -e` communicate purely through
    # their exit code, so collapsing the two would invert both checks.
    # Strip newlines only. `git status --porcelain` encodes the staged/unstaged
    # state in the first two columns, so the leading space of an unstaged line is
    # data: `.strip()` ate it on the FIRST line alone, and `line[3:]` then cut a
    # character off that one path. "evals/results/x.json" arrived as
    # "vals/results/x.json", missed the results/ exemption, and refused the
    # promotion — so promoting the three retrieval modes in sequence failed at
    # the second one. Callers that want a bare value strip their own whitespace.
    return out.stdout.strip("\n") if out.returncode == 0 else None


def _git_sha() -> str | None:
    """Full sha, not short.

    `ragas.json` shipped `promoted_git_sha: 79f3c3c` and that commit stopped
    being an ancestor of main — a later rebase orphaned it, so the one field
    whose whole job is provenance pointed at nothing reachable. Two changes
    follow from that: record the full sha (short ones also grow ambiguous as a
    repo ages), and check reachability rather than assume it — see
    `stamp_status` and backend/tests/test_eval_gate.py.
    """
    return _git("rev-parse", "HEAD")


def blocking_dirt(status: str) -> list[str]:
    """Porcelain lines that make a provenance stamp a lie, given `git status`.

    Not every dirty path does. The stamp claims "commit X produced these
    numbers", so uncommitted application or eval code invalidates it and must
    block. Other files under ``evals/results/`` cannot: they are the outputs of
    runs, they were not inputs to this one, and refusing on them makes promoting
    the three retrieval modes in sequence impossible — the first promotion
    dirties ``results/`` and the second is then refused for the damage the first
    one did.
    """
    dirty = []
    for line in status.splitlines():
        path = line[3:].strip().strip('"')
        # Renames report "old -> new"; the destination is what exists now.
        path = path.split(" -> ")[-1]
        if path.startswith("evals/results/"):
            continue
        dirty.append(path)
    return dirty


def _worktree_is_dirty() -> list[str]:
    status = _git("status", "--porcelain")
    return blocking_dirt(status) if status else []


def stamp_status(sha: str | None) -> tuple[str, str]:
    """(verdict, detail) for a stamped sha. Verdict is one of:

    ``ok``            — the commit is reachable from HEAD
    ``orphaned``      — the commit exists but HEAD no longer descends from it
    ``unverifiable``  — no git, or a shallow clone that cannot see the commit
    """
    if not sha:
        return "unverifiable", "no sha recorded"
    if _git("rev-parse", "--git-dir") is None:
        return "unverifiable", "not a git worktree"
    if _git("cat-file", "-e", f"{sha}^{{commit}}") is None:
        # A shallow clone (actions/checkout defaults to depth 1) legitimately
        # cannot see it. Absent is not the same as orphaned; saying so would
        # turn a CI checkout setting into a false alarm.
        if _git("rev-parse", "--is-shallow-repository") == "true":
            return "unverifiable", "shallow clone: commit not fetched"
        return "orphaned", f"{sha[:7]} is not in this repository"
    if _git("merge-base", "--is-ancestor", sha, "HEAD") is None:
        return "orphaned", f"{sha[:7]} exists but HEAD does not descend from it"
    return "ok", f"{sha[:7]} is an ancestor of HEAD"


def run_pattern(suite: str) -> re.Pattern[str]:
    """Exactly ``<suite>_<timestamp>.json`` — not a ``<suite>_*.json`` glob.

    The glob matched more than the runs. Under ``redteam_*.json`` it also caught
    the two hand-named artifacts the gate tests are built on, and since digits
    sort before letters ``redteam_run1_13of14.json`` came out newest — so the
    only promotable redteam run was the committed 13-of-14 regression, refused
    at the gate, with every real run judged "older than" it. Under
    ``retrieval_hybrid_*.json`` it would additionally catch
    ``retrieval_hybrid_nofilter_<ts>.json``, a different ablation that must never
    land in the file the README's hybrid row is read from.
    """
    return re.compile(rf"^{re.escape(suite)}_{_RUN_STAMP}\.json$")


def _candidates(suite: str) -> list[Path]:
    # Timestamps are UTC and zero-padded, so lexical order is chronological.
    pattern = run_pattern(suite)
    return sorted(p for p in RESULTS_DIR.iterdir() if pattern.match(p.name))


def retrieval_shape_mismatch(suite: str, summary: dict, target: Path) -> str | None:
    """Why this run cannot stand in for `target`, or None if it can."""
    expected_mode = suite.removeprefix("retrieval_")
    if summary.get("mode") != expected_mode:
        return f"run mode is {summary.get('mode')!r}, but {target.name} holds {expected_mode!r}"
    if not target.exists():
        # Nothing to be consistent with yet. There is also no README row reading
        # this file, so there is no published claim to contradict.
        return None
    committed = json.loads(target.read_text()).get("summary", {})
    for key in _RETRIEVAL_SHAPE_KEYS:
        if key in committed and summary.get(key) != committed[key]:
            return (
                f"run has {key}={summary.get(key)!r}, but {target.name} was measured "
                f"with {key}={committed[key]!r} — those are not the same quantity"
            )
    return None


def gate_for(suite: str, payload: dict) -> GateResult:
    partial = bool(payload.get("partial") or payload.get("smoke"))
    if suite == "ragas":
        return gate_ragas(payload, partial=partial)
    if suite == "redteam":
        return gate_redteam(payload["summary"], partial=partial)
    return gate_retrieval(payload["summary"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("suite", choices=sorted(SUITES))
    parser.add_argument("--run", help="promote this file (must still be the newest)")
    args = parser.parse_args()

    runs = _candidates(args.suite)
    if not runs:
        print(f"no {args.suite} runs found under {RESULTS_DIR}", file=sys.stderr)
        return 1
    newest = runs[-1]

    chosen = Path(args.run).resolve() if args.run else newest
    if chosen != newest:
        print(
            f"refusing to promote {chosen.name}: {newest.name} is newer.\n"
            "Promoting an older run over a newer one is how a committed number "
            "stops meaning anything.",
            file=sys.stderr,
        )
        return 1

    payload = json.loads(chosen.read_text())
    target = RESULTS_DIR / f"{args.suite}.json"

    if args.suite.startswith("retrieval_"):
        mismatch = retrieval_shape_mismatch(args.suite, payload.get("summary", {}), target)
        if mismatch:
            print(f"refusing to promote {chosen.name}: {mismatch}", file=sys.stderr)
            return 1

    gate = gate_for(args.suite, payload)
    gate.report()
    if not gate.passed:
        print(
            f"\nrefusing to promote {chosen.name}: it does not pass the gate.",
            file=sys.stderr,
        )
        return 1

    sha = _git_sha()
    dirt = _worktree_is_dirty() if sha else []
    if dirt:
        print(
            "refusing to promote from a dirty worktree: the stamp would name a "
            "commit that is not what produced these numbers.\n"
            f"Commit or stash first, then promote. Uncommitted: {', '.join(dirt[:5])}"
            + (f" (+{len(dirt) - 5} more)" if len(dirt) > 5 else ""),
            file=sys.stderr,
        )
        return 1

    payload["promoted_from"] = chosen.name
    payload["promoted_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    payload["promoted_git_sha"] = sha
    payload["promoted_git_ref"] = _git("rev-parse", "--abbrev-ref", "HEAD")

    target.write_text(json.dumps(payload, indent=2))
    print(f"\npromoted {chosen.name} -> {target.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
