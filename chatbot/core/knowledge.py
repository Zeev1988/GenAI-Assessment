"""Knowledge-base loader for Israeli health-fund service data.

HTML files (one per service category) are loaded once at startup and
exposed as raw HTML for direct consumption by GPT-4o, which understands
HTML structure natively.  No pre-parsing is done so no structural
information is lost.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .logger import get_logger

logger = get_logger(__name__)

# Human-readable titles for each HTML file stem.
TOPIC_TITLES: dict[str, str] = {
    "alternative_services": "רפואה משלימה / Alternative Medicine",
    "communication_clinic_services": "מרפאות תקשורת / Communication Clinics",
    "dentel_services": "מרפאות שיניים / Dental Services",
    "optometry_services": "אופטומטריה / Optometry",
    "pragrency_services": "הריון / Pregnancy",
    "workshops_services": "סדנאות בריאות / Health Workshops",
}


class KnowledgeBase:
    """Holds and exposes HTML service data for LLM consumption."""

    def __init__(self) -> None:
        # stem → raw HTML string
        self._content: dict[str, str] = {}
        self._loaded = False

    # ── Loading ────────────────────────────────────────────────────────────────

    def load(self, path: Path) -> None:
        """Load all *.html files from *path*."""
        if not path.exists():
            logger.error("Knowledge-base directory not found: %s", path)
            return

        html_files = sorted(path.glob("*.html"))
        if not html_files:
            logger.warning("No HTML files found in %s", path)
            return

        logger.info("Loading %d HTML files from %s", len(html_files), path)
        for f in html_files:
            try:
                html = f.read_text(encoding="utf-8")
                self._content[f.stem] = html
                logger.debug("Loaded: %s (%d chars)", f.name, len(html))
            except OSError as exc:
                logger.error("Failed to read %s: %s", f.name, exc)

        self._loaded = True
        logger.info(
            "Knowledge base ready — %d topics: %s",
            len(self._content),
            ", ".join(self.topic_titles()),
        )

    def is_loaded(self) -> bool:
        return self._loaded

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def all_content(self) -> str:
        """Return the complete knowledge base as a formatted string.

        Each topic is introduced with a clear heading so the LLM can
        orient itself.  The raw HTML is preserved for structural fidelity.
        """
        separator = "\n" + "=" * 80 + "\n"
        parts: list[str] = []
        for stem, html in self._content.items():
            title = TOPIC_TITLES.get(stem, stem)
            parts.append(f"### TOPIC: {title}\n\n{html}")
        return separator.join(parts)

    def topic_titles(self) -> list[str]:
        return [TOPIC_TITLES.get(s, s) for s in self._content]

    def topic_count(self) -> int:
        return len(self._content)


# ── Module-level singleton ─────────────────────────────────────────────────────

_kb: Optional[KnowledgeBase] = None


def get_knowledge_base() -> KnowledgeBase:
    """Return the singleton KnowledgeBase, loading it on first call."""
    global _kb
    if _kb is None:
        from .config import get_settings

        _kb = KnowledgeBase()
        _kb.load(get_settings().knowledge_base_path)
    return _kb
