"""Tests for Pydantic domain models."""

import pytest
from pydantic import ValidationError

from ticket_evaluator.models import EvaluatedTicket, EvaluationResult, Ticket


class TestTicket:
    """Tests for the Ticket model."""

    def test_valid_ticket(self) -> None:
        ticket = Ticket(ticket="Help me", reply="Sure thing")
        assert ticket.ticket == "Help me"
        assert ticket.reply == "Sure thing"

    def test_missing_fields_raises_error(self) -> None:
        with pytest.raises(ValidationError):
            Ticket(ticket="Help me")  # type: ignore[call-arg]


class TestEvaluationResult:
    """Tests for the EvaluationResult model."""

    def test_valid_scores(self) -> None:
        result = EvaluationResult(
            content_score=4,
            content_explanation="Good response",
            format_score=5,
            format_explanation="Well written",
        )
        assert result.content_score == 4
        assert result.format_score == 5

    def test_score_below_minimum_raises_error(self) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            EvaluationResult(
                content_score=0,
                content_explanation="Bad",
                format_score=3,
                format_explanation="OK",
            )

    def test_score_above_maximum_raises_error(self) -> None:
        with pytest.raises(ValidationError, match="less than or equal to 5"):
            EvaluationResult(
                content_score=3,
                content_explanation="OK",
                format_score=6,
                format_explanation="Too high",
            )

    def test_boundary_scores_valid(self) -> None:
        """Scores at boundaries (1 and 5) should be accepted."""
        result = EvaluationResult(
            content_score=1,
            content_explanation="Minimum",
            format_score=5,
            format_explanation="Maximum",
        )
        assert result.content_score == 1
        assert result.format_score == 5


class TestEvaluatedTicket:
    """Tests for EvaluatedTicket model and factory method."""

    def test_from_ticket_and_result(self, sample_ticket: Ticket) -> None:
        result = EvaluationResult(
            content_score=4,
            content_explanation="Good",
            format_score=5,
            format_explanation="Great",
        )
        evaluated = EvaluatedTicket.from_ticket_and_result(sample_ticket, result)

        assert evaluated.ticket == sample_ticket.ticket
        assert evaluated.reply == sample_ticket.reply
        assert evaluated.content_score == 4
        assert evaluated.format_explanation == "Great"

    def test_model_dump_has_correct_keys(self, sample_ticket: Ticket) -> None:
        result = EvaluationResult(
            content_score=3,
            content_explanation="OK",
            format_score=4,
            format_explanation="Good",
        )
        evaluated = EvaluatedTicket.from_ticket_and_result(sample_ticket, result)
        dumped = evaluated.model_dump()

        expected_keys = {
            "ticket",
            "reply",
            "content_score",
            "content_explanation",
            "format_score",
            "format_explanation",
        }
        assert set(dumped.keys()) == expected_keys
