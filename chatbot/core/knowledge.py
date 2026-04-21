"""Loads HMO knowledge-base HTML files into retrievable Chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from common import get_logger

from .retrieval import Chunk, chunk_html_document

logger = get_logger(__name__)

TOPIC_TITLES: dict[str, str] = {
    "alternative_services": "רפואה משלימה / Alternative Medicine",
    "communication_clinic_services": "מרפאות תקשורת / Communication Clinics",
    "dentel_services": "מרפאות שיניים / Dental Services",
    "optometry_services": "אופטומטריה / Optometry",
    "pragrency_services": "הריון / Pregnancy",
    "workshops_services": "סדנאות בריאות / Health Workshops",
}


class KnowledgeBase:
    def __init__(self) -> None:
        self._content: dict[str, str] = {}
        self._loaded = False

    def load(self, path: Path) -> None:
        if not path.exists():
            logger.error("Knowledge-base directory not found: %s", path)
            return
        html_files = sorted(path.glob("*.html"))
        for f in html_files:
            self._content[f.stem] = f.read_text(encoding="utf-8")
        self._loaded = True
        logger.info("Loaded %d HTML files from %s", len(self._content), path)

    def is_loaded(self) -> bool:
        return self._loaded

    def topic_titles(self) -> list[str]:
        return [TOPIC_TITLES.get(s, s) for s in self._content]

    def topic_count(self) -> int:
        return len(self._content)

    def chunks(self) -> list[Chunk]:
        out: list[Chunk] = []
        for stem, html in self._content.items():
            title = TOPIC_TITLES.get(stem, stem)
            out.extend(chunk_html_document(stem, title, html))
        return out


_kb: Optional[KnowledgeBase] = None


def get_knowledge_base() -> KnowledgeBase:
    global _kb
    if _kb is None:
        from .config import get_settings

        _kb = KnowledgeBase()
        _kb.load(get_settings().knowledge_base_path)
    return _kb
