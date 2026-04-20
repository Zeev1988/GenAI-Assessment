"""
Unit tests for chatbot/core/prompts.py

Covers:
  - SUBMIT_USER_INFO_TOOL schema structure and constraints
  - COLLECTION_SYSTEM_PROMPT content expectations
  - build_qa_system_prompt() personalisation and scope-refusal rule
"""

from __future__ import annotations

import pytest

from chatbot.core.prompts import (
    COLLECTION_SYSTEM_PROMPT,
    SUBMIT_USER_INFO_TOOL,
    build_qa_system_prompt,
)


# ── Tool schema ────────────────────────────────────────────────────────────────

class TestSubmitUserInfoTool:
    @pytest.fixture(autouse=True)
    def schema(self):
        self.fn = SUBMIT_USER_INFO_TOOL["function"]
        self.params = self.fn["parameters"]
        self.props = self.params["properties"]

    def test_tool_type_is_function(self):
        assert SUBMIT_USER_INFO_TOOL["type"] == "function"

    def test_function_name(self):
        assert self.fn["name"] == "submit_user_info"

    def test_all_required_fields_declared(self):
        expected = {
            "first_name", "last_name", "id_number", "gender",
            "age", "hmo_name", "hmo_card_number", "insurance_tier",
        }
        assert set(self.params["required"]) == expected

    def test_id_number_pattern_is_nine_digits(self):
        assert self.props["id_number"]["pattern"] == "^[0-9]{9}$"

    def test_hmo_card_number_pattern_is_nine_digits(self):
        assert self.props["hmo_card_number"]["pattern"] == "^[0-9]{9}$"

    def test_gender_enum_values(self):
        assert set(self.props["gender"]["enum"]) == {"זכר", "נקבה", "אחר"}

    def test_hmo_name_enum_values(self):
        assert set(self.props["hmo_name"]["enum"]) == {"מכבי", "מאוחדת", "כללית"}

    def test_insurance_tier_enum_values(self):
        assert set(self.props["insurance_tier"]["enum"]) == {"זהב", "כסף", "ארד"}

    def test_age_has_min_max_constraints(self):
        age = self.props["age"]
        assert age["minimum"] == 0
        assert age["maximum"] == 120

    def test_age_type_is_integer(self):
        assert self.props["age"]["type"] == "integer"

    def test_description_mentions_confirmation(self):
        desc = self.fn["description"].lower()
        assert "confirm" in desc


# ── Collection system prompt ───────────────────────────────────────────────────

class TestCollectionSystemPrompt:
    def test_mentions_all_seven_fields(self):
        prompt = COLLECTION_SYSTEM_PROMPT
        # Each field should be referenced — using Hebrew or English terms.
        fields = [
            "שם פרטי", "שם משפחה", "תעודת זהות", "מין", "גיל",
            "קופת חולים", "כרטיס", "מסלול",
        ]
        for field in fields:
            assert field in prompt, f"Expected field '{field}' in collection prompt"

    def test_mentions_valid_hmo_names(self):
        for hmo in ("מכבי", "מאוחדת", "כללית"):
            assert hmo in COLLECTION_SYSTEM_PROMPT

    def test_mentions_valid_tiers(self):
        for tier in ("זהב", "כסף", "ארד"):
            assert tier in COLLECTION_SYSTEM_PROMPT

    def test_references_submit_user_info_tool(self):
        assert "submit_user_info" in COLLECTION_SYSTEM_PROMPT

    def test_instructs_not_to_call_before_confirmation(self):
        prompt = COLLECTION_SYSTEM_PROMPT.upper()
        assert "NEVER" in prompt or "ONLY" in prompt

    def test_language_rule_present(self):
        prompt = COLLECTION_SYSTEM_PROMPT.upper()
        assert "LANGUAGE" in prompt

    def test_instructs_against_hardcoded_questions(self):
        assert "hardcode" in COLLECTION_SYSTEM_PROMPT.lower()

    def test_id_validation_rule_mentioned(self):
        # Must explain 9-digit requirement
        assert "9" in COLLECTION_SYSTEM_PROMPT


# ── Q&A system prompt builder ──────────────────────────────────────────────────

class TestBuildQaSystemPrompt:
    @pytest.fixture
    def user_info(self):
        return {
            "first_name": "ישראל",
            "last_name": "ישראלי",
            "hmo_name": "מכבי",
            "insurance_tier": "זהב",
            "age": 35,
            "gender": "זכר",
        }

    @pytest.fixture
    def kb_content(self):
        return "<h2>שירותי שיניים</h2><p>מכבי זהב: חינם פעמיים בשנה</p>"

    @pytest.fixture
    def prompt(self, user_info, kb_content):
        return build_qa_system_prompt(user_info, kb_content)

    # Member profile injection
    def test_contains_first_name(self, prompt):
        assert "ישראל" in prompt

    def test_contains_last_name(self, prompt):
        assert "ישראלי" in prompt

    def test_contains_hmo(self, prompt):
        assert "מכבי" in prompt

    def test_contains_tier(self, prompt):
        assert "זהב" in prompt

    def test_contains_age(self, prompt):
        assert "35" in prompt

    def test_contains_gender(self, prompt):
        assert "זכר" in prompt

    # Knowledge base injection
    def test_knowledge_base_content_included(self, prompt, kb_content):
        assert kb_content in prompt

    # Scope / off-topic refusal
    def test_off_topic_refusal_rule_present(self, prompt):
        assert "OFF-TOPIC" in prompt or "off-topic" in prompt.lower()

    def test_refusal_rule_instructs_not_to_answer(self, prompt):
        prompt_upper = prompt.upper()
        assert "DO NOT" in prompt_upper or "MUST" in prompt_upper

    def test_language_rule_present(self, prompt):
        assert "LANGUAGE" in prompt.upper()

    def test_focuses_on_member_hmo_and_tier(self, prompt):
        # The prompt should reference the member's specific HMO and tier
        # in the guidelines section.
        assert "מכבי" in prompt and "זהב" in prompt

    # Edge cases
    def test_empty_user_info_does_not_raise(self, kb_content):
        prompt = build_qa_system_prompt({}, kb_content)
        assert isinstance(prompt, str) and len(prompt) > 0

    def test_empty_knowledge_base_does_not_raise(self, user_info):
        prompt = build_qa_system_prompt(user_info, "")
        assert isinstance(prompt, str) and len(prompt) > 0

    def test_returns_string(self, prompt):
        assert isinstance(prompt, str)

    def test_prompt_is_substantial(self, prompt):
        # Should be a real prompt, not just a placeholder.
        assert len(prompt) > 200
