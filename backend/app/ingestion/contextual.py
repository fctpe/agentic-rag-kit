"""Context prefixes for chunks (Anthropic-style contextual retrieval).

Every chunk gets a deterministic prefix (regulation, unit ref, heading) —
that alone disambiguates most statutory chunks. The optional LLM pass adds a
one-sentence semantic summary of how the chunk relates to its unit, which
measurably helps recall on paraphrased queries; it is a flag because it costs
one model call per chunk at ingest time.
"""

import asyncio

from langchain.chat_models import init_chat_model

from app.ingestion.chunker import UnitChunk

CONTEXT_PROMPT = (
    "You situate legal text chunks for a retrieval system. In ONE sentence, state what this "
    "chunk of {regulation_title}, {ref} ({heading}) covers, so it can be found by "
    "paraphrased queries. Reply with the sentence only.\n\nCHUNK:\n{content}"
)


def deterministic_prefix(regulation_title: str, chunk: UnitChunk) -> str:
    heading = f" — {chunk.heading}" if chunk.heading else ""
    return f"{regulation_title}, {chunk.ref}{heading}."


async def llm_prefixes(
    regulation_title: str,
    chunks: list[UnitChunk],
    model_name: str,
    concurrency: int = 8,
) -> list[str]:
    model = init_chat_model(model_name, temperature=0)
    semaphore = asyncio.Semaphore(concurrency)

    async def describe(chunk: UnitChunk) -> str:
        prompt = CONTEXT_PROMPT.format(
            regulation_title=regulation_title,
            ref=chunk.ref,
            heading=chunk.heading or "no heading",
            content=chunk.content[:4000],
        )
        async with semaphore:
            response = await model.ainvoke(prompt)
        return str(response.content).strip()

    return list(await asyncio.gather(*(describe(chunk) for chunk in chunks)))
