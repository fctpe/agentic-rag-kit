"""A linkable marker is a property of the system, not a request to the model.

The frontend links exactly one shape, `[n]`. Everything the model actually
writes instead — `[4, Art. 4(5)]`, `[2(a)]`, `[7, 8]` — renders as literal text
and the source it names is never linked, and no RAGAS metric can see the
difference. Asking for the shape in the system prompt was tried and measured: it
made things worse (bb4e582). So the shape is produced by code, and these tests
are what hold it to that.

Every guarantee below has a negative control: an assertion that the test can
fail, or that the fixture really contains the defect being asserted against.
The last class replays every answer in the committed eval artifacts — real
model output against its real source list — offline and free.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from app.agent.markers import (
    MARKER_PATTERN,
    resolve_markers,
    unlinkable_brackets,
)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "evals" / "results"


def _citation(index: int, regulation: str, article: str) -> dict[str, Any]:
    return {
        "index": index,
        "regulation": regulation,
        "document": "Regulation (EU) 2016/679",
        "article": article,
        "heading": "Definitions",
        "url": "https://eur-lex.europa.eu/...",
        "snippet": "…",
        "score": 0.1,
    }


SOURCES = [
    _citation(1, "gdpr", "Art. 25"),
    _citation(2, "gdpr", "Art. 4"),
    _citation(3, "ai_act", "Art. 9"),
    _citation(4, "gdpr", "Art. 4"),
    _citation(5, "gdpr", "Art. 28"),
]


def _links(text: str) -> list[int]:
    """The indices this text will actually turn into buttons in the browser."""
    return [int(match.group(1)) for match in MARKER_PATTERN.finditer(text)]


class TestTheDefectIsRealBeforeAnythingClaimsToFixIt:
    """Negative control for the whole file: the shapes under test must not
    already be linkable, or every assertion below passes for the wrong reason."""

    @pytest.mark.parametrize(
        "shape",
        ["[4, Art. 4(5)]", "[2(a)]", "[7, 8]", "(GDPR, Art. 4(5))"],
    )
    def test_the_shape_links_nothing_as_written(self, shape: str) -> None:
        assert _links(f"A pseudonymised record is personal data {shape}.") == []

    def test_the_shape_the_frontend_does_link(self) -> None:
        assert _links("A pseudonymised record is personal data [4].") == [4]


class TestMergedBracketsBecomeMarkers:
    def test_a_reference_welded_into_the_bracket_is_moved_out_of_it(self) -> None:
        resolved = resolve_markers(
            "…may not be attributed to an identifiable natural person [4, Art. 4(5)].", SOURCES
        )
        assert resolved.text.endswith("natural person (Art. 4(5)) [4].")
        assert _links(resolved.text) == [4]
        # The spelled-out reference is kept, not discarded: it is what a lawyer
        # reads. Only its position changes.
        assert "Art. 4(5)" in resolved.text
        assert resolved.issues == []

    def test_a_sub_point_suffix_is_dropped_and_the_source_id_survives(self) -> None:
        resolved = resolve_markers("Explicit consent is one basis [2(a)].", SOURCES)
        assert resolved.text == "Explicit consent is one basis [2]."
        assert _links(resolved.text) == [2]

    def test_a_list_of_ids_becomes_a_list_of_markers(self) -> None:
        resolved = resolve_markers("Both instruments apply here [3, 5].", SOURCES)
        assert _links(resolved.text) == [3, 5]
        assert resolved.issues == []

    def test_an_already_correct_answer_is_returned_byte_for_byte(self) -> None:
        # Negative control for all three rewrites: house style is
        # "AI Act, Art. 9(2) [3]", and the normaliser must not touch it. A pass
        # here proves the rules fire on the defect rather than on any bracket.
        original = "Risk management runs across the lifecycle (AI Act, Art. 9(2)) [3]."
        resolved = resolve_markers(original, SOURCES)
        assert resolved.text == original
        assert resolved.issues == []


class TestItNeverInventsAnIndex:
    """A marker pointing at a source that does not support the sentence looks
    checked and is not — strictly worse than no marker."""

    def test_a_bracket_whose_reference_contradicts_its_source_is_not_linked(self) -> None:
        # Source 5 is GDPR Art. 28; the bracket spells out Art. 4(7).
        resolved = resolve_markers("A processor acts on instructions [5, Art. 4(7)].", SOURCES)
        assert _links(resolved.text) == []
        assert "[5, Art. 4(7)]" in resolved.text  # visible as text, not silently deleted
        assert len(resolved.issues) == 1
        assert "Art. 28" in resolved.issues[0]

    def test_the_same_shape_with_an_agreeing_reference_does_link(self) -> None:
        # Negative control for the test above: identical shape, identical code
        # path, one field changed. If the mismatch branch were removed, the
        # previous test would produce this result — so it is the branch, not the
        # shape, that blocks the link.
        resolved = resolve_markers("A controller determines the purposes [2, Art. 4(7)].", SOURCES)
        assert _links(resolved.text) == [2]

    def test_an_index_that_was_never_retrieved_never_reaches_the_client(self) -> None:
        resolved = resolve_markers("Fines reach EUR 35 000 000 [12].", SOURCES)
        assert _links(resolved.text) == []
        assert "[12]" not in resolved.text
        assert resolved.text == "Fines reach EUR 35 000 000."  # no stranded space
        assert "12" in resolved.issues[0]

    def test_a_merged_bracket_on_an_index_that_was_never_retrieved_is_removed(self) -> None:
        resolved = resolve_markers("Providers must register [9, Art. 49(1)].", SOURCES)
        assert _links(resolved.text) == []
        assert "[9" not in resolved.text
        assert resolved.issues

    def test_no_marker_in_the_output_can_ever_miss_its_source(self) -> None:
        # The postcondition, over every shape at once, including hostile ones.
        answer = (
            "One [1]. Two [2, Art. 4(5)]. Three [3, 4]. Four [4(b)]. Five [5, Art. 4(7)]. "
            "Six [6]. Seven [99, Art. 1]. Eight [0]."
        )
        resolved = resolve_markers(answer, SOURCES)
        valid = {source["index"] for source in SOURCES}
        assert _links(resolved.text)  # the rewrite did not simply erase everything
        assert set(_links(resolved.text)) <= valid


class TestAnswersThatCiteNothing:
    def test_a_refusal_is_untouched_and_reports_no_problem(self) -> None:
        refusal = (
            "I can't process this request: it looks like an attempt to override my "
            "instructions rather than a compliance question."
        )
        resolved = resolve_markers(refusal, [])
        assert resolved.text == refusal
        assert resolved.issues == []
        assert _links(resolved.text) == []

    def test_an_out_of_scope_decline_stays_unmarked(self) -> None:
        decline = "HIPAA is outside this corpus, which contains only the EU AI Act and the GDPR."
        resolved = resolve_markers(decline, SOURCES)
        assert resolved.text == decline
        assert resolved.issues == []

    def test_nothing_is_invented_when_there_are_no_sources_at_all(self) -> None:
        # Negative control for the two above: the same code path WITH a marker
        # and no sources must strip it, proving "unchanged" was a decision about
        # the text and not a code path that never runs.
        resolved = resolve_markers("Article 5 prohibits social scoring [1].", [])
        assert _links(resolved.text) == []
        assert resolved.issues


class TestCodeIsNotProse:
    def test_a_fenced_block_is_left_exactly_as_written(self) -> None:
        answer = 'See the payload:\n\n```json\n{"ids": [1, 2]}\n```\n\nSource [2].'
        resolved = resolve_markers(answer, SOURCES)
        assert '{"ids": [1, 2]}' in resolved.text
        assert _links(resolved.text) == [2]

    def test_an_inline_span_is_left_exactly_as_written(self) -> None:
        resolved = resolve_markers("The literal `[7, 8]` is not a citation [3].", SOURCES)
        assert "`[7, 8]`" in resolved.text
        assert _links(resolved.text) == [3]

    def test_the_same_text_outside_a_span_is_rewritten(self) -> None:
        # Negative control: proves the two tests above pass because of the code
        # skip and not because the rewrite rule fails to match at all.
        resolved = resolve_markers("Both apply [3, 5].", SOURCES)
        assert _links(resolved.text) == [3, 5]


class TestPerBracketCounting:
    """The eval's per-ANSWER boolean under-reports: one good marker anywhere
    satisfies it, so an answer could ship ten unlinkable brackets and pass."""

    def test_an_answer_with_one_good_marker_and_ten_bad_ones_is_not_clean(self) -> None:
        answer = "Good [1]. " + " ".join(f"Point ({c}) [2({c})]." for c in "abcdefghij")
        assert MARKER_PATTERN.search(answer)  # the per-answer check is satisfied…
        assert len(unlinkable_brackets(answer)) == 10  # …and ten brackets link to nothing

    def test_a_resolved_answer_has_none(self) -> None:
        answer = "Good [1]. " + " ".join(f"Point ({c}) [2({c})]." for c in "abcdefghij")
        resolved = resolve_markers(answer, SOURCES)
        by_index = {source["index"]: source for source in SOURCES}
        assert unlinkable_brackets(resolved.text, by_index) == []


def _committed_samples() -> list[tuple[str, str, str, list[dict[str, Any]]]]:
    samples: list[tuple[str, str, str, list[dict[str, Any]]]] = []
    for path in sorted(RESULTS_DIR.glob("ragas_2026*.json")):
        payload = json.loads(path.read_text())
        for sample in payload.get("samples", []):
            answer = sample.get("answer") or ""
            citations = sample.get("citations") or []
            if answer and citations:
                samples.append((path.name, sample["id"], answer, citations))
    return samples


class TestReplayOfEveryCommittedAnswer:
    """Real model output, real source lists, no API calls and no cost.

    Synthetic fixtures prove the rules fire; this proves they fire on what the
    model actually wrote. The artifacts are read only — nothing here writes to
    evals/results/ or depends on a measured number.
    """

    def test_the_corpus_of_answers_is_not_empty_and_contains_the_defect(self) -> None:
        # Vacuity control. Without this the class below passes on zero samples.
        samples = _committed_samples()
        assert len(samples) > 100
        merged = [
            bracket
            for _, _, answer, _ in samples
            for bracket in unlinkable_brackets(answer)
            if not MARKER_PATTERN.fullmatch(bracket)
        ]
        assert len(merged) > 100, "the committed answers no longer contain the defect under test"

    def test_every_marker_left_in_every_answer_resolves_to_a_retrieved_source(self) -> None:
        stranded: list[str] = []
        for name, sample_id, answer, citations in _committed_samples():
            resolved = resolve_markers(answer, citations)
            valid = {c["index"] for c in citations if isinstance(c.get("index"), int)}
            for index in _links(resolved.text):
                if index not in valid:
                    stranded.append(f"{name}:{sample_id}:[{index}]")
        assert stranded == []

    def test_almost_every_merged_bracket_becomes_a_link_and_the_rest_are_reported(self) -> None:
        rewritten = 0
        refused: list[str] = []
        for _, sample_id, answer, citations in _committed_samples():
            before = [b for b in unlinkable_brackets(answer) if not MARKER_PATTERN.fullmatch(b)]
            if not before:
                continue
            resolved = resolve_markers(answer, citations)
            after = [
                b for b in unlinkable_brackets(resolved.text) if not MARKER_PATTERN.fullmatch(b)
            ]
            rewritten += len(before) - len(after)
            refused.extend(f"{sample_id}:{b}" for b in after)
            # Nothing is ever dropped in silence: every bracket that survived
            # unlinked has an issue explaining why.
            assert len(resolved.issues) >= len(after)
        # The census the README and ADR 0007 quote. Printed rather than pinned to
        # a literal, because the denominator is "whatever ragas_*.json is
        # committed" and a promoted run would make a hard-coded number stale —
        # the failure mode this repo has already been bitten by.
        #   uv run --group dev --group evals pytest tests/test_citation_markers.py -s -k merged
        print(
            f"\nmerged brackets in committed answers: {rewritten + len(refused)} — "
            f"{rewritten} became links, {len(refused)} refused: {sorted(set(refused))}"
        )
        assert rewritten > 100
        # The refusals are the brackets whose spelled-out reference contradicts
        # the source they number — every one of them, rather than an
        # unrecognised shape that simply fell through.
        assert all(re.search(r"Art", item) for item in refused)


class TestTheContractMatchesTheFrontend:
    def test_the_pattern_is_the_one_the_browser_uses(self) -> None:
        # frontend/lib/citationMarkers.ts is the client half; the two regexes
        # must stay identical or the backend guarantees a shape the UI ignores.
        # frontend/tests/citation-markers.test.mjs pins the other side.
        source = (
            Path(__file__).resolve().parents[2] / "frontend" / "lib" / "citationMarkers.ts"
        ).read_text()
        assert "/\\[(\\d{1,3})\\]/g" in source
        assert MARKER_PATTERN.pattern == r"\[(\d{1,3})\]"
