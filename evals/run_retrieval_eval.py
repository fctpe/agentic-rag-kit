"""Retrieval-only evaluation: hit-rate@k and MRR against the golden set.

No LLM judges — cheap and deterministic (modulo the embedding call for the
vector arm). For every non-out-of-scope golden question the script retrieves
top-k chunks and checks whether the expected articles ("AI Act Art. 5",
"GDPR Art. 6", ...) appear among them. A chunk counts as a hit for an
expected article when its document regulation AND article_ref both match.

Modes:
  hybrid       app.retrieval.hybrid.hybrid_search (RRF fusion, production path)
  vector_only  pgvector cosine arm only (SQL inlined below)
  text_only    Postgres full-text arm only (SQL inlined below; needs no
               embedding call, so it runs without an OpenAI key)

Run from the backend project so the app package and eval deps resolve:

    cd backend && uv run --group evals python ../evals/run_retrieval_eval.py \
        --mode hybrid --k 6 --regulation-filter on

Metrics per question:
  hit@k           1 if any expected article appears in the top-k chunks
  article_recall  fraction of expected articles found in the top-k
  mrr             1/rank of the first chunk matching any expected article

The regulation filter ablation (--regulation-filter on|off): "on" passes the
regulation implied by the expected articles (only when they all share one
regulation — cross-regulation questions always run unfiltered), "off" always
searches the full corpus.
"""

import argparse
import asyncio
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml
from sqlalchemy import text

EVALS_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = EVALS_DIR / "golden_questions.yaml"
RESULTS_DIR = EVALS_DIR / "results"

sys.path.insert(0, str(EVALS_DIR))
from gate import gate_retrieval  # noqa: E402

_ARTICLE_LABEL = re.compile(r"^(AI Act|GDPR)\s+Art\.?\s*(\S+)$", re.IGNORECASE)
_REGULATION_KEYS = {"ai act": "ai_act", "gdpr": "gdpr"}

# --- single-arm SQL variants -------------------------------------------------
# hybrid_search() has no mode switch, so the two ablation arms replicate the
# corresponding CTE arm of backend/app/retrieval/hybrid.py HYBRID_SQL with the
# other arm dropped. Keep in sync with that file if the schema changes.

VECTOR_ONLY_SQL = text(
    """
SELECT c.id, c.article_ref, d.regulation,
       row_number() OVER (ORDER BY c.embedding <=> CAST(:qvec AS vector)) AS rank
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE (CAST(:regulation AS text) IS NULL OR d.regulation = :regulation)
ORDER BY c.embedding <=> CAST(:qvec AS vector)
LIMIT :final_k
"""
)

TEXT_ONLY_SQL = text(
    """
SELECT c.id, c.article_ref, d.regulation,
       row_number() OVER (ORDER BY ts_rank_cd(c.tsv, query) DESC) AS rank
FROM chunks c
JOIN documents d ON d.id = c.document_id,
plainto_tsquery('english', CAST(:query AS text)) query
WHERE c.tsv @@ query
  AND (CAST(:regulation AS text) IS NULL OR d.regulation = :regulation)
ORDER BY ts_rank_cd(c.tsv, query) DESC
LIMIT :final_k
"""
)


def parse_expected(labels: list[str]) -> list[tuple[str, str]]:
    """ "AI Act Art. 5" -> ("ai_act", "5"). Fails loudly on malformed labels."""
    parsed: list[tuple[str, str]] = []
    for label in labels:
        match = _ARTICLE_LABEL.match(label.strip())
        if not match:
            raise ValueError(f"Cannot parse expected article label: {label!r}")
        parsed.append((_REGULATION_KEYS[match.group(1).lower()], match.group(2)))
    return parsed


def implied_regulation(expected: list[tuple[str, str]]) -> str | None:
    regulations = {regulation for regulation, _ in expected}
    return regulations.pop() if len(regulations) == 1 else None


def article_number(article_ref: str) -> str:
    """chunks.article_ref is "Art. 5" (see app/ingestion/eurlex.py Article.ref)."""
    return article_ref.replace("Art.", "").replace("Article", "").strip()


def score_question(
    expected: list[tuple[str, str]],
    retrieved: list[tuple[str, str]],  # ordered (regulation, article_ref)
) -> dict:
    retrieved_keys = [
        (regulation, article_number(article_ref)) for regulation, article_ref in retrieved
    ]
    matched = [exp for exp in expected if exp in retrieved_keys]
    first_rank = 0
    for rank, key in enumerate(retrieved_keys, start=1):
        if key in expected:
            first_rank = rank
            break
    return {
        "hit": int(bool(matched)),
        "article_recall": (len(matched) / len(expected)) if expected else 0.0,
        "mrr": (1.0 / first_rank) if first_rank else 0.0,
        "first_hit_rank": first_rank or None,
        "matched_articles": [f"{regulation}:{number}" for regulation, number in matched],
        "retrieved": [f"{regulation}:{number}" for regulation, number in retrieved_keys],
    }


async def retrieve(session, mode: str, query: str, regulation: str | None, k: int):
    """Return an ordered list of (regulation, article_ref) for the top-k chunks."""
    if mode == "hybrid":
        from app.retrieval.hybrid import hybrid_search

        chunks = await hybrid_search(session, query, regulation=regulation, final_k=k)
        return [(chunk.regulation, chunk.article_ref) for chunk in chunks]

    if mode == "vector_only":
        from app.retrieval.embedder import embed_query
        from app.retrieval.hybrid import _to_vector_literal

        query_vector = await embed_query(query)
        # Same index-scan relaxation the production path applies.
        await session.execute(text("SET hnsw.iterative_scan = relaxed_order"))
        rows = await session.execute(
            VECTOR_ONLY_SQL,
            {"qvec": _to_vector_literal(query_vector), "regulation": regulation, "final_k": k},
        )
        return [(row["regulation"], row["article_ref"]) for row in rows.mappings()]

    if mode == "text_only":
        rows = await session.execute(
            TEXT_ONLY_SQL,
            {"query": query, "regulation": regulation, "final_k": k},
        )
        return [(row["regulation"], row["article_ref"]) for row in rows.mappings()]

    raise ValueError(f"Unknown mode: {mode}")


def load_questions(smoke: bool, only_ids: list[str] | None) -> list[dict]:
    data = yaml.safe_load(GOLDEN_PATH.read_text())
    questions = [q for q in data["questions"] if q["query_type"] != "out_of_scope"]
    if only_ids:
        wanted = set(only_ids)
        questions = [q for q in questions if q["id"] in wanted]
        missing = wanted - {q["id"] for q in questions}
        if missing:
            raise SystemExit(f"Unknown/out-of-scope question ids: {sorted(missing)}")
    if smoke:
        questions = questions[:5]
    return questions


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--k", type=int, default=6, help="top-k chunks to score (default 6)")
    parser.add_argument("--mode", choices=["hybrid", "vector_only", "text_only"], default="hybrid")
    parser.add_argument(
        "--regulation-filter",
        choices=["on", "off"],
        default="on",
        help="on: filter by the regulation implied by expected_articles; off: full corpus",
    )
    parser.add_argument("--json", type=Path, default=None, help="output JSON path")
    parser.add_argument("--smoke", action="store_true", help="first 5 questions only")
    parser.add_argument("--questions", nargs="+", default=None, help="run only these ids")
    args = parser.parse_args()

    from app.db import dispose_engine, get_session_factory

    questions = load_questions(args.smoke, args.questions)
    use_filter = args.regulation_filter == "on"

    rows: list[dict] = []
    factory = get_session_factory()
    async with factory() as session:
        for question in questions:
            expected = parse_expected(question["expected_articles"])
            regulation = implied_regulation(expected) if use_filter else None
            try:
                retrieved = await retrieve(
                    session, args.mode, question["question"], regulation, args.k
                )
            except Exception as err:
                print(f"[{question['id']}] retrieval failed: {err}", file=sys.stderr)
                raise
            scores = score_question(expected, retrieved)
            rows.append(
                {
                    "id": question["id"],
                    "query_type": question["query_type"],
                    "difficulty": question["difficulty"],
                    "regulation_filter": regulation,
                    "expected": [f"{reg}:{num}" for reg, num in expected],
                    **scores,
                }
            )
    await dispose_engine()

    n = len(rows)
    summary = {
        "mode": args.mode,
        "k": args.k,
        "regulation_filter": args.regulation_filter,
        "smoke": args.smoke,
        "n_questions": n,
        "hit_rate_at_k": round(sum(r["hit"] for r in rows) / n, 4) if n else 0.0,
        "mrr": round(sum(r["mrr"] for r in rows) / n, 4) if n else 0.0,
        "mean_article_recall": round(sum(r["article_recall"] for r in rows) / n, 4) if n else 0.0,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    # markdown report
    print(
        f"\n## Retrieval eval — mode={args.mode}, k={args.k}, "
        f"regulation_filter={args.regulation_filter}, n={n}\n"
    )
    print("| id | type | filter | hit@k | recall | mrr | matched |")
    print("|----|------|--------|-------|--------|-----|---------|")
    for r in rows:
        print(
            f"| {r['id']} | {r['query_type']} | {r['regulation_filter'] or '-'} "
            f"| {r['hit']} | {r['article_recall']:.2f} | {r['mrr']:.3f} "
            f"| {', '.join(r['matched_articles']) or '-'} |"
        )
    print(
        f"\n**hit-rate@{args.k}: {summary['hit_rate_at_k']:.3f}** · "
        f"**MRR: {summary['mrr']:.3f}** · "
        f"**mean article recall: {summary['mean_article_recall']:.3f}**"
    )

    suffix = "" if use_filter else "_nofilter"
    out_path = args.json or RESULTS_DIR / f"retrieval_{args.mode}{suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "questions": rows}, indent=2))
    print(f"\nwrote {out_path}")

    gate = gate_retrieval(summary)
    gate.report()
    if summary.get("smoke"):
        print("\n(smoke run — thresholds not applied)")
        return 0
    return 0 if gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
