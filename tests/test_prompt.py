"""Tests for prompt templates and builder function."""

from ticket_evaluator.prompt import SYSTEM_PROMPT, build_user_prompt


class TestSystemPrompt:
    """Tests for the system prompt content."""

    def test_contains_scoring_rubric(self) -> None:
        assert "Content Score" in SYSTEM_PROMPT
        assert "Format Score" in SYSTEM_PROMPT

    def test_contains_json_schema_instruction(self) -> None:
        assert "content_score" in SYSTEM_PROMPT
        assert "format_score" in SYSTEM_PROMPT
        assert "JSON" in SYSTEM_PROMPT

    def test_contains_all_score_levels(self) -> None:
        for level in ["1", "2", "3", "4", "5"]:
            assert f"| {level} |" in SYSTEM_PROMPT


class TestBuildUserPrompt:
    """Tests for the user prompt builder."""

    def test_includes_ticket_text(self) -> None:
        prompt = build_user_prompt("My order is late", "Sorry about that")
        assert "My order is late" in prompt

    def test_includes_reply_text(self) -> None:
        prompt = build_user_prompt("My order is late", "Sorry about that")
        assert "Sorry about that" in prompt

    def test_contains_section_headers(self) -> None:
        prompt = build_user_prompt("ticket text", "reply text")
        assert "Customer Ticket" in prompt
        assert "AI-Generated Reply" in prompt
