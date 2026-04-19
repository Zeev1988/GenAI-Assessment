"""
Unit tests for chatbot/core/knowledge.py

Tests are grouped by concern:
  - Loading behaviour (happy path, missing path, partial failures)
  - Content retrieval (all_content format, topic metadata)
  - Singleton factory
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from chatbot.core.knowledge import KnowledgeBase, TOPIC_TITLES, get_knowledge_base


# ── Helpers ────────────────────────────────────────────────────────────────────

def fresh_kb(path: Path) -> KnowledgeBase:
    """Create and load a fresh KnowledgeBase (bypasses the module singleton)."""
    kb = KnowledgeBase()
    kb.load(path)
    return kb


# ── Loading: happy path ────────────────────────────────────────────────────────

class TestLoading:
    def test_loads_all_html_files(self, phase2_data_path):
        kb = fresh_kb(phase2_data_path)
        assert kb.is_loaded()
        assert kb.topic_count() == 6

    def test_all_expected_stems_present(self, phase2_data_path):
        kb = fresh_kb(phase2_data_path)
        expected_stems = {
            "alternative_services",
            "communication_clinic_services",
            "dentel_services",
            "optometry_services",
            "pragrency_services",
            "workshops_services",
        }
        assert expected_stems == set(kb._content.keys())

    def test_content_is_non_empty(self, phase2_data_path):
        kb = fresh_kb(phase2_data_path)
        for stem, html in kb._content.items():
            assert len(html) > 0, f"Content for '{stem}' should not be empty"

    def test_content_contains_html_tags(self, phase2_data_path):
        kb = fresh_kb(phase2_data_path)
        for stem, html in kb._content.items():
            assert "<" in html and ">" in html, (
                f"Content for '{stem}' should contain HTML tags"
            )

    def test_content_contains_hebrew(self, phase2_data_path):
        """Sanity-check that the Hebrew text was read correctly (UTF-8)."""
        kb = fresh_kb(phase2_data_path)
        combined = kb.all_content()
        # Every file mentions at least one HMO name.
        assert "מכבי" in combined
        assert "מאוחדת" in combined
        assert "כללית" in combined

    def test_content_contains_all_tiers(self, phase2_data_path):
        kb = fresh_kb(phase2_data_path)
        combined = kb.all_content()
        assert "זהב" in combined
        assert "כסף" in combined
        assert "ארד" in combined


# ── Loading: failure paths ─────────────────────────────────────────────────────

class TestLoadingFailures:
    def test_missing_directory_does_not_raise(self, tmp_path):
        kb = KnowledgeBase()
        kb.load(tmp_path / "does_not_exist")
        assert not kb.is_loaded()

    def test_empty_directory_does_not_raise(self, tmp_path):
        kb = KnowledgeBase()
        kb.load(tmp_path)           # tmp_path exists but is empty
        # is_loaded is True only when at least one file was loaded
        # (our implementation sets _loaded=True after the loop regardless)
        # but topic_count should be 0.
        assert kb.topic_count() == 0

    def test_unreadable_file_skipped(self, tmp_path):
        """A file that raises OSError should be skipped, not crash the loader."""
        good = tmp_path / "good.html"
        good.write_text("<h2>Good</h2>", encoding="utf-8")
        bad = tmp_path / "bad.html"
        bad.write_text("<h2>Bad</h2>", encoding="utf-8")

        original_read_text = Path.read_text

        def patched_read_text(self, **kwargs):
            if self.name == "bad.html":
                raise OSError("Permission denied")
            return original_read_text(self, **kwargs)

        with patch.object(Path, "read_text", patched_read_text):
            kb = KnowledgeBase()
            kb.load(tmp_path)

        assert "good" in kb._content
        assert "bad" not in kb._content


# ── Content retrieval ──────────────────────────────────────────────────────────

class TestContentRetrieval:
    def test_all_content_contains_topic_headings(self, phase2_data_path):
        kb = fresh_kb(phase2_data_path)
        content = kb.all_content()
        # Each topic is introduced with "### TOPIC: <title>"
        for stem in kb._content:
            title = TOPIC_TITLES.get(stem, stem)
            assert f"### TOPIC: {title}" in content, (
                f"Expected heading for topic '{stem}' in all_content()"
            )

    def test_all_content_separates_topics(self, phase2_data_path):
        kb = fresh_kb(phase2_data_path)
        content = kb.all_content()
        # Topics are separated by a line of '=' characters.
        assert "=" * 40 in content

    def test_topic_titles_length_matches_topic_count(self, phase2_data_path):
        kb = fresh_kb(phase2_data_path)
        assert len(kb.topic_titles()) == kb.topic_count()

    def test_topic_titles_are_strings(self, phase2_data_path):
        kb = fresh_kb(phase2_data_path)
        for title in kb.topic_titles():
            assert isinstance(title, str) and len(title) > 0

    def test_known_title_mapping(self):
        assert TOPIC_TITLES["dentel_services"] == "מרפאות שיניים / Dental Services"
        assert TOPIC_TITLES["optometry_services"] == "אופטומטריה / Optometry"

    def test_all_content_empty_when_not_loaded(self):
        kb = KnowledgeBase()
        assert kb.all_content() == ""


# ── Singleton factory ──────────────────────────────────────────────────────────

class TestSingleton:
    def test_get_knowledge_base_returns_knowledge_base_instance(self):
        kb = get_knowledge_base()
        assert isinstance(kb, KnowledgeBase)

    def test_get_knowledge_base_returns_same_object(self):
        kb1 = get_knowledge_base()
        kb2 = get_knowledge_base()
        assert kb1 is kb2
