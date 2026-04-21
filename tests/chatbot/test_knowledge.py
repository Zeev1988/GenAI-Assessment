"""Basic tests for chatbot/core/knowledge.py."""

from __future__ import annotations

from chatbot.core.knowledge import KnowledgeBase


def test_loads_html_files_from_directory(test_data_path):
    kb = KnowledgeBase()
    kb.load(test_data_path)
    assert kb.is_loaded()
    assert kb.topic_count() > 0


def test_chunks_cover_every_topic(test_data_path):
    kb = KnowledgeBase()
    kb.load(test_data_path)
    topics_in_chunks = {c.topic for c in kb.chunks()}
    assert topics_in_chunks == set(kb.topic_titles())
