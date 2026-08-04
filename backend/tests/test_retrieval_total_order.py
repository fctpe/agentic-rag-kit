"""Retrieval must order totally, or the corpus reproducing buys nothing.

Pinning the text and the vectors makes the *inputs* to retrieval a function of
the fixture. It does not make the *output* one, because RRF scores are sums of
`1/(k + rank)` over small integers and exact ties are arithmetic rather than
accident. Golden question A07 produced two chunks with bit-identical fused
scores — AI Act Art. 4 at vector rank 1 and Art. 66 at text rank 1, both
`0x1.0c9714fbcda3bp-6` — and under `ORDER BY f.score DESC` alone the winner was
whatever the query plan happened to emit first:

    final_k=2 -> Art. 4      final_k=3 -> Art. 66
    final_k=6 -> Art. 66     final_k=8 -> Art. 4

Same rows, same scores, only the LIMIT varied. That is a production defect —
the rank-1 citation shown to a user depends on how many results were requested
— and across ingests it was worth 0.0132 of hybrid MRR, the last committed
retrieval number still moving after the corpus was pinned.

These tests are offline: they hold the two properties the fix rests on, which
are both checkable without a database. The behavioural check (two runs of the
same query returning the same rows) needs Postgres and runs in CI against the
service container, which needs no provider key now that the query vectors are
committed too.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from app.ingestion.chunker import chunk_unit
from app.ingestion.eurlex import REGULATIONS, load_fixture
from app.retrieval.hybrid import HYBRID_SQL

_EVALS = Path(__file__).resolve().parents[2] / "evals"
sys.path.insert(0, str(_EVALS))

from run_retrieval_eval import TEXT_ONLY_SQL, VECTOR_ONLY_SQL  # noqa: E402

#: The tiebreak columns, in order. `COLLATE "C"` because the default collation
#: is an environment property (ICU version, locale) and a tiebreak that sorts
#: differently on another machine is not a tiebreak.
TIEBREAK = 'd.regulation COLLATE "C", c.article_ref COLLATE "C", c.idx'


def _order_by_clauses(sql: str) -> list[str]:
    """Every ORDER BY in a statement, window ORDER BYs included.

    A window's ordering decides the rank that goes into RRF, so it needs the
    tiebreak exactly as much as the final one does — and a naive regex misses
    that, because `ORDER BY c.embedding <=> CAST(:qvec AS vector)` contains a
    closing paren that does not end the clause. Depth is tracked instead: the
    clause ends at LIMIT, at the paren that closes a group opened before it, or
    at the end of the statement.
    """
    clauses: list[str] = []
    for match in re.finditer(r"ORDER BY", sql):
        depth = 0
        end = len(sql)
        index = match.end()
        while index < len(sql):
            character = sql[index]
            if character == "(":
                depth += 1
            elif character == ")":
                if depth == 0:
                    end = index
                    break
                depth -= 1
            elif depth == 0 and sql.startswith("LIMIT", index):
                end = index
                break
            index += 1
        clauses.append(" ".join(sql[match.end() : end].split()))
    return clauses


def test_every_order_by_in_the_retrieval_sql_is_total() -> None:
    """A partial order anywhere in the chain leaks plan-dependence into the result."""
    for name, statement in (
        ("HYBRID_SQL", HYBRID_SQL),
        ("VECTOR_ONLY_SQL", VECTOR_ONLY_SQL),
        ("TEXT_ONLY_SQL", TEXT_ONLY_SQL),
    ):
        clauses = _order_by_clauses(str(statement))
        assert clauses, f"{name} has no ORDER BY at all"
        for clause in clauses:
            assert clause.endswith(TIEBREAK), f"{name}: ORDER BY without the tiebreak: {clause}"


def test_the_check_above_rejects_the_sql_it_replaced() -> None:
    """Negative control: the pre-fix statement fails the same assertion.

    Without this the test would pass on any string containing the tiebreak
    somewhere, which is not what it claims to check.
    """
    before = """
    SELECT c.id, row_number() OVER (ORDER BY c.embedding <=> CAST(:qvec AS vector)) AS rank
    FROM chunks c JOIN documents d ON d.id = c.document_id
    ORDER BY f.score DESC
    LIMIT :final_k
    """
    clauses = _order_by_clauses(before)
    assert clauses and not all(clause.endswith(TIEBREAK) for clause in clauses)


def test_the_arms_order_before_they_limit() -> None:
    """A LIMIT with no ORDER BY takes an arbitrary subset.

    The text arm used to `LIMIT :arm_size` with no ordering of its own. The
    ranks were computed by a window over every match, so which twenty rows
    survived into fusion was decided by the plan — the same class of defect as
    the fused tie, one level down, and invisible because the plan happened to
    emit them in window order.
    """
    sql = str(HYBRID_SQL)
    for limit in ("LIMIT :arm_size", "LIMIT :final_k"):
        assert limit in sql
        before_limit = sql[: sql.index(limit)]
        # The nearest preceding clause has to be an ORDER BY, not a WHERE.
        assert before_limit.rindex("ORDER BY") > before_limit.rindex("FROM")


def test_the_tiebreak_is_unique_over_the_committed_corpus() -> None:
    """`(regulation, article_ref, idx)` identifies exactly one chunk.

    This is what makes it a *total* order rather than a shorter tie. It is also
    why it is used instead of `chunks.id`: the id is a fresh uuid4 on every
    ingest, so ordering by it would be total and still move between ingests —
    a reproducible-looking fix that reproduces nothing.
    """
    keys = [
        (regulation, chunk.ref, chunk.idx)
        for regulation in REGULATIONS
        for unit in load_fixture(regulation)
        for chunk in chunk_unit(unit)
    ]
    assert len(keys) == 284
    assert len(set(keys)) == len(keys)

    # Negative control: drop `idx` and the key stops being unique, so the
    # assertion above is a property of this key and not of any triple.
    assert len({(regulation, ref) for regulation, ref, _ in keys}) < len(keys)
