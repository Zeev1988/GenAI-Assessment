"""Basic tests for chatbot/core/retrieval.py."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from chatbot.core.retrieval import Chunk, Retriever, chunk_html_document


def _run(coro):
    return asyncio.run(coro)


def _make_embedding_client(vectors: list[list[float]]) -> MagicMock:
    """Mock AsyncAzureOpenAI client that yields pre-canned embedding vectors in order."""
    client = MagicMock()
    calls: list[list[float]] = []

    def side_effect(model: str, input: Any):
        n = len(input)
        batch = vectors[len(calls):len(calls) + n]
        calls.extend(batch)
        resp = MagicMock()
        resp.data = [MagicMock(embedding=v) for v in batch]
        return resp

    client.embeddings.create = AsyncMock(side_effect=side_effect)
    return client


_SAMPLE_HTML = """
<h2>מרפאות שיניים</h2>
<p>מרפאות שיניים מציעות מגוון רחב של שירותים.</p>
<table>
  <tr><th>שירות</th><th>מכבי</th></tr>
  <tr><td>ניקוי שיניים</td><td><strong>זהב:</strong> חינם</td></tr>
</table>
"""


def test_chunker_produces_intro_plus_service_rows():
    chunks = chunk_html_document("dentel_services", "מרפאות שיניים", _SAMPLE_HTML)
    assert len(chunks) == 2  # 1 intro + 1 service row (header row skipped)
    assert chunks[0].kind == "intro"
    assert chunks[1].kind == "service"
    assert all(c.topic == "מרפאות שיניים" for c in chunks)


def test_retriever_search_ranks_results_by_cosine():
    chunks = [
        Chunk(topic="T", text="dental", html="<p>dental</p>", kind="service"),
        Chunk(topic="T", text="vision", html="<p>vision</p>", kind="service"),
    ]
    client = _make_embedding_client([
        [1.0, 0.0],   # dental
        [0.0, 1.0],   # vision
        [1.0, 0.0],   # query → matches dental
    ])
    retriever = Retriever(client, embedding_deployment="ada")
    _run(retriever.index(chunks))

    results = _run(retriever.search("teeth cleaning", k=2))
    assert len(results) == 2
    assert results[0][0].text == "dental"
    assert results[0][1] == pytest.approx(1.0)
