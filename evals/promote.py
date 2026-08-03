#!/usr/bin/env python3
"""Promote a timestamped eval run to the committed result the README cites.

The suites write ``ragas_<timestamp>.json`` and ``redteam_<timestamp>.json``;
the files committed under ``evals/results/`` are ``ragas.json`` and
``redteam.json``. That gap used to be closed by hand with ``mv``, which meant
the committed numbers had no provable link to any particular run — and nothing
stopped a older, friendlier run being promoted over a newer, worse one.

This script closes it with three rules:

1. **Only the newest run.** Promoting an older run over a newer one is refused.
   If the latest run is worse, that is the number.
2. **Only a run that passes the gate.** A breaching run cannot become the
   committed baseline; fix the regression or move the threshold deliberately.
3. **Stamped.** The promoted file records the source filename, the commit that
   produced it, and when it was promoted.

    cd backend && uv run --group evals python ../evals/promote.py ragas
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVALS_DIR / "results"

sys.path.insert(0, str(EVALS_DIR))
from gate import gate_ragas, gate_redteam  # noqa: E402

SUITES = {"ragas": "ragas_*.json", "redteam": "redteam_*.json"}


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=EVALS_DIR,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None
    return out.stdout.strip() or None


def _candidates(suite: str) -> list[Path]:
    # Timestamps are UTC and zero-padded, so lexical order is chronological.
    return sorted(p for p in RESULTS_DIR.glob(SUITES[suite]) if p.name != f"{suite}.json")


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
    partial = bool(payload.get("partial") or payload.get("smoke"))
    gate = (
        gate_ragas(payload, partial=partial)
        if args.suite == "ragas"
        else gate_redteam(payload["summary"], partial=partial)
    )
    gate.report()
    if not gate.passed:
        print(
            f"\nrefusing to promote {chosen.name}: it does not pass the gate.",
            file=sys.stderr,
        )
        return 1

    payload["promoted_from"] = chosen.name
    payload["promoted_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    payload["promoted_git_sha"] = _git_sha()

    target = RESULTS_DIR / f"{args.suite}.json"
    target.write_text(json.dumps(payload, indent=2))
    print(f"\npromoted {chosen.name} -> {target.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
