"""Prose must agree with the artifacts it describes.

The promotion machinery worked and the prose layer never consumed its output:
`evals/thresholds.yaml` carried every baseline to the digit while README.md
quoted an older draw, gave three mutually inconsistent chunk counts, and told
the reader in a blockquote to discount figures that had in fact been re-measured.
Nothing failed, because nothing was checking the documents.

These tests are that check. They read the numbers out of the artifacts —
`thresholds.yaml`, the promoted runs, the committed fixtures — and fail when a
document states something else. They run offline and free, so a stale README is
a red build rather than a discovery a reviewer makes.

Each assertion carries a negative control: `test_*_detects_drift` mutates the
artifact or the prose and asserts the same check fails. A sync test that cannot
fail is the defect it was written to prevent, one layer up.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
EVALS = ROOT / "evals"
FIXTURES = ROOT / "data" / "fixtures"

# Every document that quotes a measured number to a reader. docs/adr/ is
# deliberately absent: an ADR records what was known when the decision was made,
# and rewriting its measurements would destroy the record. ADR 0003's *current*
# corpus line is checked explicitly below, because it is also the definition the
# rest of the repo copies from.
PUBLISHED = [ROOT / "README.md", EVALS / "README.md"]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _thresholds() -> dict:
    return yaml.safe_load(_read(EVALS / "thresholds.yaml"))


def _result(name: str) -> dict:
    return json.loads(_read(EVALS / "results" / name))


# --------------------------------------------------------------------------
# corpus counts
# --------------------------------------------------------------------------


def _corpus_counts() -> dict[str, int]:
    """Chunk counts derived from the committed fixtures, not from any document."""
    entries = json.loads(_read(FIXTURES / "chunk_embeddings.json"))["entries"]
    counts: dict[str, int] = {"total": len(entries)}
    for entry in entries:
        counts[entry["regulation"]] = counts.get(entry["regulation"], 0) + 1
    for name, key in (("ai_act.json", "ai_act"), ("gdpr.json", "gdpr")):
        doc = json.loads(_read(FIXTURES / name))
        counts[f"{key}_articles"] = len(doc["articles"])
        counts[f"{key}_annexes"] = len(doc.get("annexes", []))
    return counts


def test_corpus_fixtures_agree_with_each_other() -> None:
    counts = _corpus_counts()
    prefixes = json.loads(_read(FIXTURES / "context_prefixes.json"))["entries"]
    digest = json.loads(_read(FIXTURES / "corpus_digest.json"))
    assert counts["total"] == len(prefixes) == digest["chunks"]
    assert counts["ai_act"] + counts["gdpr"] == counts["total"]


@pytest.mark.parametrize("path", PUBLISHED, ids=lambda p: p.parent.name + "/" + p.name)
def test_published_chunk_counts_are_current(path: Path) -> None:
    """No document may state a chunk count the fixtures do not produce.

    Written as "every `-> N chunks` in the prose is a real count" rather than
    "the right number appears somewhere", because the defect was a README that
    contained 284 *and* 280 *and* 283 and was correct by the weaker test.
    """
    counts = _corpus_counts()
    legal = {counts["total"], counts["ai_act"], counts["gdpr"]}
    stated = {
        int(m) for m in re.findall(r"(?:→|->)\s*\*{0,2}(\d{2,4})\s*chunks", _read(path))
    }
    assert stated, f"{path} states no chunk count at all"
    assert stated <= legal, f"{path} states chunk counts {sorted(stated - legal)}"


def test_published_chunk_counts_detect_drift(tmp_path: Path) -> None:
    """Negative control for the check above."""
    counts = _corpus_counts()
    stale = tmp_path / "README.md"
    stale.write_text(f"the corpus is 212 articles -> {counts['total'] - 4} chunks\n")
    with pytest.raises(AssertionError):
        test_published_chunk_counts_are_current(stale)


def test_adr_0003_states_the_current_corpus() -> None:
    counts = _corpus_counts()
    adr = _read(ROOT / "docs" / "adr" / "0003-structural-chunking-contextual-prefixes.md")
    articles = counts["ai_act_articles"] + counts["gdpr_articles"]
    annexes = counts["ai_act_annexes"] + counts["gdpr_annexes"]
    expected = (
        f"**Corpus: {articles} articles + {annexes} annexes → {counts['total']} chunks** "
        f"(AI Act {counts['ai_act']}, GDPR {counts['gdpr']})"
    )
    assert expected in adr


def test_committed_digest_compare_accepts_the_committed_pair() -> None:
    from app.ingestion.corpus_digest import compare, load_committed  # noqa: PLC0415

    committed = load_committed()
    measured = (committed["chunks"], committed["text"], committed["vector"])
    assert compare(measured) == []


@pytest.mark.parametrize("field", ["chunks", "text", "vector"])
def test_committed_digest_compare_detects_drift(field: str) -> None:
    """Negative control, one per field.

    The vector case is the one that matters: the previous round's reproducibility
    claim was checked on the text digest alone and passed while `chunks.embedding`
    changed on every ingest. A check that only notices the text is the bug.
    """
    from app.ingestion.corpus_digest import compare, load_committed  # noqa: PLC0415

    committed = load_committed()
    values = {k: committed[k] for k in ("chunks", "text", "vector")}
    values[field] = 1 if field == "chunks" else "0" * 64
    mismatches = compare((values["chunks"], values["text"], values["vector"]))
    assert [m.split(":")[0] for m in mismatches] == [field]


# --------------------------------------------------------------------------
# eval metrics
# --------------------------------------------------------------------------


def _published_metrics() -> list[tuple[str, str]]:
    """(label, value-as-written) pairs the README must quote."""
    thresholds = _thresholds()
    rows: list[tuple[str, str]] = []
    ragas = _result("ragas.json")["summary"]
    for metric, spec in thresholds["ragas"]["metrics"].items():
        assert ragas[metric] == spec["baseline"], (
            f"thresholds.yaml baseline for {metric} does not match ragas.json"
        )
        rows.append((f"ragas.{metric}", f"{spec['baseline']:.4f}".rstrip("0")))
    for mode, spec in thresholds["retrieval"]["modes"].items():
        summary = _result(f"retrieval_{mode}.json")["summary"]
        for metric, bounds in spec["metrics"].items():
            assert summary[metric] == bounds["baseline"], (
                f"thresholds.yaml baseline for {mode}.{metric} does not match its run"
            )
    return rows


def test_thresholds_baselines_match_the_promoted_runs() -> None:
    assert _published_metrics()


def _gate_passing_ragas_runs() -> dict[str, float]:
    """Every timestamped RAGAS run under results/, filtered by the gate."""
    import sys  # noqa: PLC0415

    sys.path.insert(0, str(EVALS))
    from gate import gate_ragas  # noqa: PLC0415

    passing = {}
    for path in sorted((EVALS / "results").glob("ragas_2*.json")):
        run = json.loads(_read(path))
        if gate_ragas(run, partial=False).passed:
            passing[path.name] = run["summary"]["faithfulness"]
    return passing


_NUMBER_WORDS = {
    2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six",
    7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten",
}


def _spell(n: int) -> str:
    """The README writes counts as words, so the check has to read them as words."""
    return _NUMBER_WORDS.get(n, str(n))


def test_the_quoted_spread_is_every_run_the_gate_accepts() -> None:
    """n=2 is a selection the gate makes, not one the author made.

    The README quotes a two-run spread and says how many other runs the gate
    refuses. That exclusion has to be reproducible or the n looks cherry-picked:
    three of the refused runs score all 38 questions and fail only on the
    inline-citation count, which is exactly the kind of exclusion a reader is
    right to distrust.

    The count is asserted, not just the pass set. The prose said five when the
    gate refused eight, and nothing caught it — a check that counts only what
    passes is blind to a miscount of what does not.
    """
    passing = _gate_passing_ragas_runs()
    promoted = _result("ragas.json")
    assert len(passing) == 2, f"gate now accepts {sorted(passing)}"

    refused = len(list((EVALS / "results").glob("ragas_2*.json"))) - len(passing)
    readme = _read(ROOT / "README.md")
    assert f"**{_spell(refused)} other runs" in readme or f"{_spell(refused)} other runs" in readme, (
        f"the gate refuses {refused} runs; the README states a different count"
    )
    assert promoted["promoted_from"] in passing
    for name, faithfulness in passing.items():
        if name != promoted["promoted_from"]:
            assert name in readme, f"README omits the other passing run {name}"
        assert f"{faithfulness:.4f}" in readme


def test_gate_selection_check_detects_a_loosened_gate() -> None:
    """Negative control: if the gate stopped refusing the rejected runs, this fails."""
    total = len(list((EVALS / "results").glob("ragas_2*.json")))
    assert total > len(_gate_passing_ragas_runs()), (
        "every committed RAGAS run passes the gate — the refusals the README "
        "describes are no longer being made"
    )


@pytest.mark.parametrize("path", PUBLISHED, ids=lambda p: p.parent.name + "/" + p.name)
def test_no_retired_metric_survives_in_prose(path: Path) -> None:
    """The specific numbers that must never reappear.

    Not a general "is every decimal current" scan — that cannot be written
    without parsing English. These are the draws that were published and are now
    known wrong, so their reappearance is a copy-paste regression, which is
    exactly how each of them arrived in the first place.
    """
    retired = {
        "0.918": "faithfulness mean over 23 of 38 questions",
        "0.932": "faithfulness mean over 28 of 38 questions",
        "0.964": "single July draw, no committed artifact",
        "0.886": "no committed artifact",
        "0.881": "superseded answer_relevancy draw",
        "0.849": "superseded context_precision draw",
        "0.961": "superseded context_recall draw",
        "0.891": "pre-fix hybrid MRR, untied ORDER BY",
        "0.904": "pre-fix vector-only MRR, untied ORDER BY",
        "0.897": "pre-fix article recall",
        "0.772": "OR-semantics variant, no committed artifact",
    }
    body = _read(path)
    # Anchored so 0.891 does not match the 0.8912 in evals/README.md's account of
    # commit 88131af — that is a historical narrative about this exact class of
    # drift, and a scan that cannot tell the two apart would force it deleted.
    found = sorted(v for v in retired if re.search(rf"{re.escape(v)}(?!\d)", body))
    assert not found, "\n".join(f"{v} — {retired[v]}" for v in found)


def test_retired_metric_scan_detects_drift(tmp_path: Path) -> None:
    """Negative control for the check above."""
    stale = tmp_path / "README.md"
    stale.write_text("faithfulness 0.918-0.932 over two runs\n")
    with pytest.raises(AssertionError):
        test_no_retired_metric_survives_in_prose(stale)


def _assert_readme_quotes(readme: str, ragas: dict, retrieval: dict) -> None:
    for metric, value in ragas.items():
        assert f"{value:.4f}".rstrip("0") in readme, f"README omits {metric} {value}"
    for mode, summary in retrieval.items():
        assert f"{summary['mrr']:.4f}".rstrip("0") in readme, f"README omits {mode} MRR"


def _retrieval_summaries() -> dict[str, dict]:
    modes = _thresholds()["retrieval"]["modes"]
    return {m: _result(f"retrieval_{m}.json")["summary"] for m in modes}


def test_readme_quotes_the_current_ragas_and_retrieval_numbers() -> None:
    _assert_readme_quotes(
        _read(ROOT / "README.md"),
        _result("ragas.json")["summary"],
        _retrieval_summaries(),
    )


def test_readme_quote_check_detects_drift() -> None:
    """Negative control: a moved baseline must fail the README check."""
    moved = dict(_result("ragas.json")["summary"], faithfulness=0.5001)
    with pytest.raises(AssertionError):
        _assert_readme_quotes(_read(ROOT / "README.md"), moved, _retrieval_summaries())


# --------------------------------------------------------------------------
# the hybrid-vs-vector delta the README explains in prose
# --------------------------------------------------------------------------


def _mrr_delta() -> list[dict]:
    """The questions where hybrid and vector-only disagree — the README's jq, in Python."""
    vector = {q["id"]: q["mrr"] for q in _result("retrieval_vector_only.json")["questions"]}
    return [
        {"id": q["id"], "hybrid": q["mrr"], "vector": vector[q["id"]]}
        for q in _result("retrieval_hybrid.json")["questions"]
        if q["mrr"] != vector[q["id"]]
    ]


def test_readme_names_every_question_in_the_hybrid_vector_delta() -> None:
    """The prose said three questions; the README's own jq returns one.

    A reviewer who runs the documented command must not get an answer that
    contradicts the paragraph above it.
    """
    ids = [row["id"] for row in _mrr_delta()]
    readme = _read(ROOT / "README.md")
    assert len(ids) == 1, f"delta is now {ids} — the README paragraph needs rewriting"
    assert f"one question, {ids[0]}" in readme
    # And no question outside the delta may be named as part of it.
    for question in ("A01", "A07"):
        if question not in ids:
            assert "gap is three questions" not in readme
            assert re.search(rf"\b{question}\b.{{0,40}}ranks", readme) is None


def test_mrr_delta_matches_the_headline_gap() -> None:
    hybrid = _result("retrieval_hybrid.json")["summary"]["mrr"]
    vector = _result("retrieval_vector_only.json")["summary"]["mrr"]
    n = _thresholds()["retrieval"]["expect_n_questions"]
    from_delta = sum(row["vector"] - row["hybrid"] for row in _mrr_delta()) / n
    assert round(vector - hybrid, 4) == round(from_delta, 4)
    assert f"{round(vector - hybrid, 4)}" in _read(ROOT / "README.md")
