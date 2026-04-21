"""ADA-002 retrieval over the HTML knowledge base.

Each HTML file is split into chunks (one per <tr>, plus an intro chunk
for the non-table content). Embeddings are computed once at startup; a
linear cosine scan ranks chunks per query.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from bs4 import BeautifulSoup
from openai import APIError, AsyncAzureOpenAI

from common import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Chunk:
    topic: str
    text: str   # used for embedding
    html: str   # injected into the prompt
    kind: str   # "intro" | "service"

    @property
    def prompt_block(self) -> str:
        return f"### TOPIC: {self.topic}\n\n{self.html}"

    @property
    def embedding_input(self) -> str:
        return f"{self.topic}\n{self.text}"


def chunk_html_document(stem: str, topic_title: str, html: str) -> list[Chunk]:
    """Split an HTML file: one intro chunk + one chunk per table row."""
    chunks: list[Chunk] = []
    soup = BeautifulSoup(html or "", "html.parser")

    # Intro = everything outside <table>.
    intro_soup = BeautifulSoup(str(soup), "html.parser")
    for table in intro_soup.find_all("table"):
        table.decompose()
    intro_html = str(intro_soup).strip()
    intro_text = intro_soup.get_text(separator="\n", strip=True)
    if intro_text:
        chunks.append(Chunk(topic_title, intro_text, intro_html, "intro"))

    # One chunk per <tr> (skip header rows, prepend header text for context).
    for table in soup.find_all("table"):
        header_text = ""
        for tr in table.find_all("tr"):
            if tr.find("th") is not None:
                header_text = tr.get_text(separator=" | ", strip=True)
                continue
            tr_text = tr.get_text(separator=" | ", strip=True)
            if not tr_text:
                continue
            text = f"{header_text}\n{tr_text}" if header_text else tr_text
            chunks.append(Chunk(topic_title, text, f"<table>{tr!s}</table>", "service"))

    if not chunks:
        logger.warning("No chunks extracted from %s", stem)
    return chunks


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class Retriever:
    """In-memory semantic index over Chunks using ADA-002 embeddings."""

    def __init__(self, client: AsyncAzureOpenAI, embedding_deployment: str) -> None:
        self._client = client
        self._deployment = embedding_deployment
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []
        self._ready = False

    async def index(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        logger.info("Embedding %d chunks", len(chunks))
        try:
            response = await self._client.embeddings.create(
                model=self._deployment,
                input=[c.embedding_input for c in chunks],
            )
        except APIError as exc:
            logger.error("Embedding call failed: %s", exc)
            return

        self._chunks = list(chunks)
        self._vectors = [item.embedding for item in response.data]
        self._ready = True
        logger.info("Retriever ready — %d chunks indexed", len(chunks))

    async def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        q = (query or "").strip()
        if not self._ready or not q or k <= 0:
            return []
        try:
            resp = await self._client.embeddings.create(model=self._deployment, input=[q])
        except APIError as exc:
            logger.error("Query embedding failed: %s", exc)
            return []

        q_vec = resp.data[0].embedding
        scored = [(c, _cosine(q_vec, v)) for c, v in zip(self._chunks, self._vectors)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def is_ready(self) -> bool:
        return self._ready

    def chunk_count(self) -> int:
        return len(self._chunks)
