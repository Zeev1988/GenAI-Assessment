"""Semantic retrieval over the HTML knowledge base (Azure OpenAI ADA-002).

The knowledge base is small enough to fit in GPT-4o's context, but the
assignment explicitly lists `text-embedding-ada-002` as an available
resource and retrieval is the production-ready pattern — so we build it.

Design notes
------------
* **Chunk boundaries follow the document structure.**  Each HTML file has a
  single ``<h2>`` heading, a short intro (``<p>`` + ``<ul>``), and a table
  where every ``<tr>`` is one service with columns per HMO.  One ``<tr>``
  is the smallest self-contained unit of meaning, so each becomes a chunk.
  The intro block becomes one additional chunk per topic — that way general
  "what services does X cover?" questions can retrieve the overview rather
  than a random row.

* **Embedding input is cleaned text; the LLM still sees the raw HTML.**
  Embeddings are higher-signal on plain text (angle brackets are noise),
  but GPT-4o handles HTML natively and the tables carry structural meaning
  the LLM needs.  Each chunk therefore stores both a ``text`` field (used
  for the embedding) and an ``html`` field (what goes into the prompt).

* **DOM-based chunking.**  The HTML is parsed with BeautifulSoup so nested
  tables, commented-out rows, and attribute quirks don't confuse the slicer
  the way a regex-based approach would.

* **No external retrieval dependencies.**  Cosine similarity is a one-liner;
  adding faiss / chromadb / numpy would be overkill for a ~50-chunk corpus.
  The native Azure OpenAI SDK is the only network dependency, in line with
  the assignment's "no LangChain / no frameworks" rule.

* **Graceful degradation.**  If the embedding deployment is unavailable at
  startup the retriever logs a warning and ``is_ready()`` returns False;
  callers fall back to stuffing the full knowledge base into the prompt
  (see ``Retriever.fallback_chunks`` and ``api/main.py:_handle_qa``).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag
from openai import APIError, AzureOpenAI

from common import get_logger

logger = get_logger(__name__)


# ── Chunk data model ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Chunk:
    """A single retrievable unit of the knowledge base.

    Attributes:
        topic:  Human-readable topic title (e.g. "מרפאות שיניים").
        text:   Plain-text rendering used to compute the embedding.
        html:   Raw HTML rendering passed to the LLM in the prompt.
        kind:   "intro" or "service" — mostly for logging / debugging.
    """

    topic: str
    text: str
    html: str
    kind: str

    @property
    def prompt_block(self) -> str:
        """The string that actually goes into the system prompt."""
        return f"### TOPIC: {self.topic}\n\n{self.html}"

    @property
    def embedding_input(self) -> str:
        """The string fed to the embedding model (topic + clean text)."""
        return f"{self.topic}\n{self.text}"


# ── HTML chunker ──────────────────────────────────────────────────────────────

# Block-level tags that should produce a line break in the plain-text render.
_BLOCK_TAGS: frozenset[str] = frozenset({
    "br", "tr", "li", "p", "h1", "h2", "h3", "h4", "h5", "h6", "div",
})
# Cell tags separated by " | " to preserve table column boundaries.
_CELL_TAGS: frozenset[str] = frozenset({"td", "th"})


def _node_text(node: Tag | str) -> str:
    """Render a BeautifulSoup node as clean, whitespace-normalised text.

    Block-level tags inject newlines, ``<td>`` / ``<th>`` inject a pipe
    separator so the column structure survives into the embedding input.
    """
    parts: list[str] = []

    def walk(el):
        if isinstance(el, str):
            parts.append(el)
            return
        if not hasattr(el, "name") or el.name is None:
            return
        tag = el.name.lower()
        if tag in _BLOCK_TAGS:
            parts.append("\n")
        elif tag in _CELL_TAGS:
            parts.append(" | ")
        for child in el.children:
            walk(child)

    if isinstance(node, str):
        parts.append(node)
    else:
        for child in node.children:
            walk(child)

    raw = "".join(parts)
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n\s*\n+", "\n", raw)
    return raw.strip()


def chunk_html_document(stem: str, topic_title: str, html: str) -> list[Chunk]:
    """Split a single HTML file into retrievable chunks.

    Uses a DOM parser (BeautifulSoup) so nested tables, commented-out rows,
    and attribute quirks don't confuse the slicer the way a regex would.
    Every ``<tr>`` that holds ``<th>`` cells is skipped; everything outside
    every ``<table>`` becomes the "intro" chunk for the topic.
    """
    chunks: list[Chunk] = []
    soup = BeautifulSoup(html or "", "html.parser")

    # ── Intro (everything outside of tables) ──────────────────────────────
    intro_soup = BeautifulSoup(str(soup), "html.parser")
    for table in intro_soup.find_all("table"):
        table.decompose()
    intro_html = str(intro_soup).strip()
    intro_text = _node_text(intro_soup)
    if intro_text:
        chunks.append(
            Chunk(
                topic=topic_title,
                text=intro_text,
                html=intro_html,
                kind="intro",
            )
        )

    # ── One chunk per service row ────────────────────────────────────────
    # We collect rows per-table so a header row in one table doesn't bleed
    # context into rows of a sibling table.
    for table in soup.find_all("table"):
        header_text = ""
        for tr in table.find_all("tr"):
            if tr.find("th") is not None:
                # Capture the header row so its column labels (HMO names,
                # etc.) can be prepended to every service chunk for context.
                header_text = _node_text(tr)
                continue
            tr_text = _node_text(tr)
            if not tr_text:
                continue
            combined_text = f"{header_text}\n{tr_text}" if header_text else tr_text
            chunks.append(
                Chunk(
                    topic=topic_title,
                    text=combined_text,
                    # Wrap rows in a minimal <table> so the LLM still sees
                    # the tabular structure when the chunk is injected into
                    # the prompt on its own.
                    html=f"<table>{tr!s}</table>",
                    kind="service",
                )
            )

    if not chunks:
        logger.warning("No chunks extracted from %s (%d chars)", stem, len(html))
    return chunks


# ── Cosine similarity (no numpy) ──────────────────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors."""
    if len(a) != len(b):
        raise ValueError(f"Vector length mismatch: {len(a)} vs {len(b)}")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


# ── Retriever ─────────────────────────────────────────────────────────────────

class Retriever:
    """In-memory semantic index backed by Azure OpenAI ADA-002.

    The index is built once, at startup, from a list of Chunks.  Each call
    to :py:meth:`search` embeds the query and ranks the chunks by cosine
    similarity.  For a corpus of O(50) chunks the linear scan costs a
    handful of microseconds — no vector DB required.
    """

    def __init__(
        self,
        client: AzureOpenAI,
        embedding_deployment: str,
    ) -> None:
        self._client = client
        self._deployment = embedding_deployment
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []
        self._ready = False

    # ── Indexing ──────────────────────────────────────────────────────────

    def index(self, chunks: list[Chunk]) -> None:
        """Embed every chunk and store the vectors in memory."""
        if not chunks:
            logger.warning("Retriever.index() called with zero chunks")
            return

        inputs = [c.embedding_input for c in chunks]
        logger.info(
            "Embedding %d chunks with deployment=%s",
            len(inputs),
            self._deployment,
        )
        try:
            response = self._client.embeddings.create(
                model=self._deployment,
                input=inputs,
            )
        except APIError as exc:
            status = getattr(exc, "status_code", "N/A")
            logger.error(
                "Embedding call failed (status=%s): %s — retrieval disabled, "
                "QA handler will fall back to full-context mode.",
                status,
                exc,
            )
            return

        vectors = [item.embedding for item in response.data]
        if len(vectors) != len(chunks):
            logger.error(
                "Embedding count mismatch: got %d vectors for %d chunks",
                len(vectors),
                len(chunks),
            )
            return

        self._chunks = list(chunks)
        self._vectors = vectors
        self._ready = True
        dim = len(vectors[0]) if vectors else 0
        logger.info(
            "Retriever ready — %d chunks indexed, embedding dim=%d",
            len(chunks),
            dim,
        )

    # ── Retrieval ─────────────────────────────────────────────────────────

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        """Return the top-k most similar chunks to *query*.

        Returns an empty list if the index is not ready or *query* is empty.
        """
        if not self._ready:
            logger.debug("Retriever not ready — returning no results")
            return []
        q = (query or "").strip()
        if not q:
            return []
        if k <= 0:
            return []

        try:
            resp = self._client.embeddings.create(
                model=self._deployment,
                input=[q],
            )
        except APIError as exc:
            status = getattr(exc, "status_code", "N/A")
            logger.error(
                "Query embedding failed (status=%s): %s — returning no results",
                status,
                exc,
            )
            return []

        q_vec = resp.data[0].embedding
        scored = [
            (chunk, _cosine(q_vec, vec))
            for chunk, vec in zip(self._chunks, self._vectors)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: min(k, len(scored))]

    # ── Status ────────────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self._ready

    def chunk_count(self) -> int:
        return len(self._chunks)
