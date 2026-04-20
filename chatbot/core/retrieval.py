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

* **No external retrieval dependencies.**  Cosine similarity is a one-liner;
  adding faiss / chromadb / numpy would be overkill for a ~50-chunk corpus.
  The native Azure OpenAI SDK is the only network dependency, in line with
  the assignment's "no LangChain / no frameworks" rule.

* **Graceful degradation.**  If the embedding deployment is unavailable at
  startup the retriever logs a warning and ``is_ready()`` returns False;
  callers fall back to stuffing the full KB into the prompt.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from html.parser import HTMLParser

from openai import AzureOpenAI, APIError

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

class _TextExtractor(HTMLParser):
    """Strip all tags, preserve inline whitespace, collapse extra blanks."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag in {"br", "tr", "li", "p", "h1", "h2", "h3"}:
            self._parts.append("\n")
        elif tag in {"td", "th"}:
            self._parts.append(" | ")

    def get_text(self) -> str:
        raw = "".join(self._parts)
        # Collapse runs of spaces but keep newlines.
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n\s*\n+", "\n", raw)
        return raw.strip()


def _html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return parser.get_text()


_TR_RE = re.compile(r"<tr\b[^>]*>.*?</tr>", re.IGNORECASE | re.DOTALL)
_H2_RE = re.compile(r"<h2\b[^>]*>(.*?)</h2>", re.IGNORECASE | re.DOTALL)
_TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)


def chunk_html_document(stem: str, topic_title: str, html: str) -> list[Chunk]:
    """Split a single HTML file into retrievable chunks.

    The first ``<tr>`` in each table is the header row (``<th>`` cells) and
    is skipped.  Everything outside the table becomes the "intro" chunk for
    the topic.
    """
    chunks: list[Chunk] = []

    # ── Intro (everything before / around the table) ──────────────────────
    intro_html = _TABLE_RE.sub("", html).strip()
    intro_text = _html_to_text(intro_html)
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
    header_html = ""
    header_text = ""
    for tr_html in _TR_RE.findall(html):
        if "<th" in tr_html.lower():
            # Capture the header row so its column labels (HMO names, etc.)
            # can be prepended to every service chunk for context.
            header_html = tr_html
            header_text = _html_to_text(tr_html)
            continue
        tr_text = _html_to_text(tr_html)
        if not tr_text:
            continue
        # Merge header labels into the chunk text so retrieval sees HMO names.
        combined_text = f"{header_text}\n{tr_text}" if header_text else tr_text
        chunks.append(
            Chunk(
                topic=topic_title,
                text=combined_text,
                # Wrap rows in a minimal <table> so the LLM still sees the
                # tabular structure when the chunk is injected into the prompt.
                # Note: header_html is intentionally excluded here — its column
                # labels are already baked into combined_text for retrieval.
                html=f"<table>{tr_html}</table>",
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
