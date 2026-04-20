"""
Unit tests for chatbot/core/retrieval.py

Covers:
  - HTML chunker (structure, topic propagation, Hebrew preservation)
  - Cosine similarity math
  - Retriever indexing + top-k search with a mocked embedding client
  - Graceful degradation when the embedding API fails
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from openai import APIError

from chatbot.core.retrieval import (
    Chunk,
    Retriever,
    _cosine,
    chunk_html_document,
)
from chatbot.core.knowledge import TOPIC_TITLES


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_embedding_client(vectors: list[list[float]]) -> MagicMock:
    """Return a mock Azure OpenAI client whose embeddings.create() yields *vectors*.

    The mock is deliberately batch-agnostic: each call returns the pre-canned
    vectors in the same order they were supplied.
    """
    client = MagicMock()
    calls: list[list[float]] = []

    def side_effect(model: str, input):
        n = len(input)
        # Return the first n vectors we haven't yet returned.
        start = len(calls)
        end = start + n
        if end > len(vectors):
            raise AssertionError(
                f"Mock embedding client exhausted: asked for {n} more at index "
                f"{start}, only {len(vectors)} preset vectors"
            )
        batch = vectors[start:end]
        calls.extend(batch)

        resp = MagicMock()
        resp.data = [MagicMock(embedding=vec) for vec in batch]
        return resp

    client.embeddings.create.side_effect = side_effect
    return client


def _make_api_error() -> APIError:
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.headers = {}
    return APIError("Embedding deployment is down", mock_resp, body=None)


# ── Cosine similarity ──────────────────────────────────────────────────────────

class TestCosine:
    def test_identical_vectors_score_one(self):
        assert _cosine([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self):
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_score_minus_one(self):
        assert _cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self):
        assert _cosine([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            _cosine([1.0], [1.0, 2.0])


# ── Chunker ────────────────────────────────────────────────────────────────────

_SAMPLE_HTML = """
<h2>מרפאות שיניים</h2>
<p>מרפאות שיניים מציעות מגוון רחב של שירותים.</p>
<ul><li>בדיקות</li><li>סתימות</li></ul>
<table>
  <tr><th>שירות</th><th>מכבי</th><th>כללית</th></tr>
  <tr>
    <td>בדיקות וניקוי שיניים</td>
    <td><strong>זהב:</strong> חינם פעמיים בשנה</td>
    <td><strong>זהב:</strong> חינם פעם בחודש</td>
  </tr>
  <tr>
    <td>סתימות</td>
    <td><strong>זהב:</strong> 80% הנחה</td>
    <td><strong>זהב:</strong> 70% הנחה</td>
  </tr>
</table>
"""


class TestChunker:
    def test_produces_intro_plus_one_chunk_per_row(self):
        chunks = chunk_html_document("dentel_services", "מרפאות שיניים", _SAMPLE_HTML)
        # 1 intro + 2 service rows  (header row with <th> is skipped)
        assert len(chunks) == 3

    def test_header_row_is_skipped(self):
        chunks = chunk_html_document("dentel_services", "מרפאות שיניים", _SAMPLE_HTML)
        for c in chunks:
            assert "<th" not in c.html.lower(), "header row must not become a chunk"

    def test_each_chunk_carries_topic(self):
        chunks = chunk_html_document("dentel_services", "מרפאות שיניים", _SAMPLE_HTML)
        for c in chunks:
            assert c.topic == "מרפאות שיניים"
            assert "### TOPIC: מרפאות שיניים" in c.prompt_block

    def test_first_chunk_is_intro_with_narrative_text(self):
        chunks = chunk_html_document("dentel_services", "מרפאות שיניים", _SAMPLE_HTML)
        intro = chunks[0]
        assert intro.kind == "intro"
        assert "מציעות" in intro.text       # from the intro <p>
        assert "בדיקות" in intro.text       # from the <ul>

    def test_service_chunks_contain_benefit_data(self):
        chunks = chunk_html_document("dentel_services", "מרפאות שיניים", _SAMPLE_HTML)
        service_chunks = [c for c in chunks if c.kind == "service"]
        # First service row is the check-up row.
        assert any("בדיקות וניקוי שיניים" in c.text for c in service_chunks)
        # Percentages come through in the text used for embedding.
        assert any("80%" in c.text or "80 %" in c.text for c in service_chunks)

    def test_hebrew_preserved_through_chunking(self):
        chunks = chunk_html_document("dentel_services", "מרפאות שיניים", _SAMPLE_HTML)
        combined = " ".join(c.text for c in chunks)
        for word in ("זהב", "מכבי", "כללית", "סתימות"):
            assert word in combined

    def test_chunk_html_is_wrapped_for_llm(self):
        chunks = chunk_html_document("dentel_services", "מרפאות שיניים", _SAMPLE_HTML)
        service = next(c for c in chunks if c.kind == "service")
        # Service rows are wrapped in a <table> so the table structure
        # survives when the chunk is injected into the prompt on its own.
        assert service.html.strip().startswith("<table>")
        assert "<tr" in service.html.lower()

    def test_empty_html_returns_empty_list(self):
        assert chunk_html_document("empty", "Empty", "") == []

    def test_html_without_table_returns_only_intro(self):
        html = "<h2>Title</h2><p>Just prose, no table.</p>"
        chunks = chunk_html_document("prose_only", "Title", html)
        assert len(chunks) == 1
        assert chunks[0].kind == "intro"


# ── Retriever: indexing ────────────────────────────────────────────────────────

def _chunks_for(*texts: str) -> list[Chunk]:
    return [
        Chunk(topic="T", text=t, html=f"<p>{t}</p>", kind="service")
        for t in texts
    ]


class TestRetrieverIndex:
    def test_indexing_stores_vectors(self):
        chunks = _chunks_for("foo", "bar")
        client = _make_embedding_client([[1.0, 0.0], [0.0, 1.0]])
        retriever = Retriever(client, embedding_deployment="ada")
        retriever.index(chunks)

        assert retriever.is_ready()
        assert retriever.chunk_count() == 2

    def test_indexing_passes_batch_to_embedding_api(self):
        chunks = _chunks_for("a", "b", "c")
        client = _make_embedding_client([[1.0], [1.0], [1.0]])
        retriever = Retriever(client, embedding_deployment="ada")
        retriever.index(chunks)

        # The embedder should be called exactly once — a single batch.
        assert client.embeddings.create.call_count == 1
        kwargs = client.embeddings.create.call_args.kwargs
        assert kwargs["model"] == "ada"
        assert len(kwargs["input"]) == 3

    def test_empty_chunk_list_is_a_noop(self):
        client = MagicMock()
        retriever = Retriever(client, embedding_deployment="ada")
        retriever.index([])
        assert not retriever.is_ready()
        client.embeddings.create.assert_not_called()

    def test_api_error_during_indexing_leaves_retriever_not_ready(self):
        client = MagicMock()
        client.embeddings.create.side_effect = _make_api_error()
        retriever = Retriever(client, embedding_deployment="ada")
        retriever.index(_chunks_for("foo"))
        assert not retriever.is_ready()


# ── Retriever: search ──────────────────────────────────────────────────────────

class TestRetrieverSearch:
    def test_returns_chunks_ranked_by_cosine(self):
        chunks = _chunks_for("dental", "vision", "pregnancy")
        # Dental is along x, vision along y, pregnancy along z.
        client = _make_embedding_client(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
             [1.0, 0.0, 0.0]]   # query embedding — pointing at "dental"
        )
        retriever = Retriever(client, embedding_deployment="ada")
        retriever.index(chunks)

        results = retriever.search("teeth cleaning", k=3)
        assert len(results) == 3
        assert results[0][0].text == "dental"
        assert results[0][1] == pytest.approx(1.0)
        # The other two are orthogonal to the query → score 0.
        assert results[1][1] == pytest.approx(0.0)
        assert results[2][1] == pytest.approx(0.0)

    def test_top_k_limits_result_count(self):
        chunks = _chunks_for("a", "b", "c", "d")
        client = _make_embedding_client(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
             [0.9, 0.1, 0.05, 0.01]]
        )
        retriever = Retriever(client, embedding_deployment="ada")
        retriever.index(chunks)
        results = retriever.search("q", k=2)
        assert len(results) == 2

    def test_k_larger_than_corpus_returns_all(self):
        chunks = _chunks_for("a", "b")
        client = _make_embedding_client(
            [[1, 0], [0, 1], [0.5, 0.5]]
        )
        retriever = Retriever(client, embedding_deployment="ada")
        retriever.index(chunks)
        results = retriever.search("q", k=50)
        assert len(results) == 2

    def test_empty_query_returns_no_results(self):
        chunks = _chunks_for("a")
        client = _make_embedding_client([[1.0]])
        retriever = Retriever(client, embedding_deployment="ada")
        retriever.index(chunks)
        # Only the index call should consume embeddings — searches with
        # an empty query must short-circuit before hitting the API.
        assert retriever.search("", k=5) == []
        assert retriever.search("   ", k=5) == []
        assert client.embeddings.create.call_count == 1

    def test_not_ready_retriever_returns_empty(self):
        retriever = Retriever(MagicMock(), embedding_deployment="ada")
        assert retriever.search("anything", k=3) == []

    def test_api_error_during_search_returns_empty(self):
        chunks = _chunks_for("a", "b")
        client = _make_embedding_client([[1, 0], [0, 1]])
        retriever = Retriever(client, embedding_deployment="ada")
        retriever.index(chunks)
        # Now make subsequent calls fail.
        client.embeddings.create.side_effect = _make_api_error()
        assert retriever.search("q", k=2) == []


# ── Integration with the real HTML knowledge base ─────────────────────────────

class TestChunkerOnRealData:
    """Feed the chunker the actual phase-2 HTML files to catch real-world edge cases."""

    @pytest.fixture
    def kb_dir(self) -> Path:
        return Path(__file__).parent / "test_data"

    def test_every_file_produces_at_least_two_chunks(self, kb_dir):
        """One intro + at least one service row per topic."""
        for html_path in kb_dir.glob("*.html"):
            stem = html_path.stem
            title = TOPIC_TITLES.get(stem, stem)
            chunks = chunk_html_document(stem, title, html_path.read_text(encoding="utf-8"))
            assert len(chunks) >= 2, f"{html_path.name}: expected ≥2 chunks, got {len(chunks)}"

    def test_no_chunk_has_empty_text_or_html(self, kb_dir):
        for html_path in kb_dir.glob("*.html"):
            stem = html_path.stem
            title = TOPIC_TITLES.get(stem, stem)
            chunks = chunk_html_document(stem, title, html_path.read_text(encoding="utf-8"))
            for c in chunks:
                assert c.text.strip(), f"{html_path.name}: empty text in a chunk"
                assert c.html.strip(), f"{html_path.name}: empty html in a chunk"
                assert c.topic == title

    def test_service_chunks_mention_tiers_or_hmos(self, kb_dir):
        """Sanity check: service-level chunks should carry benefit data."""
        markers = ("זהב", "כסף", "ארד", "מכבי", "מאוחדת", "כללית")
        for html_path in kb_dir.glob("*.html"):
            stem = html_path.stem
            title = TOPIC_TITLES.get(stem, stem)
            chunks = chunk_html_document(stem, title, html_path.read_text(encoding="utf-8"))
            service_chunks = [c for c in chunks if c.kind == "service"]
            for c in service_chunks:
                assert any(m in c.text for m in markers), (
                    f"{html_path.name}: service chunk missing tier/HMO markers"
                )
