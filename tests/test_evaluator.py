"""Tests for the async LLM ticket evaluator."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from ticket_evaluator.evaluator import TicketEvaluator
from ticket_evaluator.exceptions import EvaluationError
from ticket_evaluator.models import Ticket


@pytest.fixture
def evaluator(mock_openai_client: AsyncMock) -> TicketEvaluator:
    """Create a TicketEvaluator with a mocked OpenAI client."""
    return TicketEvaluator(
        client=mock_openai_client,
        model="gpt-4o-mini",
        max_retries=2,
        max_concurrency=2,
    )


class TestEvaluate:
    """Tests for single ticket evaluation."""

    @pytest.mark.asyncio
    async def test_successful_evaluation(
        self, evaluator: TicketEvaluator, sample_ticket: Ticket
    ) -> None:
        result = await evaluator.evaluate(sample_ticket)

        assert result.content_score == 4
        assert result.format_score == 5
        assert (
            "shipping" in result.content_explanation.lower()
            or len(result.content_explanation) > 0
        )

    @pytest.mark.asyncio
    async def test_empty_response_raises_error(
        self, evaluator: TicketEvaluator, sample_ticket: Ticket
    ) -> None:
        # Override mock to return None content
        mock_message = MagicMock()
        mock_message.content = None
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        evaluator.client.chat.completions.create = AsyncMock(return_value=mock_response)

        with pytest.raises(EvaluationError, match="empty response"):
            await evaluator.evaluate(sample_ticket)

    @pytest.mark.asyncio
    async def test_invalid_json_raises_error(
        self, evaluator: TicketEvaluator, sample_ticket: Ticket
    ) -> None:
        # Override mock to return invalid JSON
        mock_message = MagicMock()
        mock_message.content = "not valid json"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        evaluator.client.chat.completions.create = AsyncMock(return_value=mock_response)

        with pytest.raises(EvaluationError, match="parse"):
            await evaluator.evaluate(sample_ticket)

    @pytest.mark.asyncio
    async def test_invalid_score_in_response_raises_error(
        self, evaluator: TicketEvaluator, sample_ticket: Ticket
    ) -> None:
        # Return valid JSON but with out-of-range score
        bad_response = {
            "content_score": 10,  # Invalid: must be 1-5
            "content_explanation": "test",
            "format_score": 3,
            "format_explanation": "test",
        }
        mock_message = MagicMock()
        mock_message.content = json.dumps(bad_response)
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        evaluator.client.chat.completions.create = AsyncMock(return_value=mock_response)

        with pytest.raises(EvaluationError, match="parse"):
            await evaluator.evaluate(sample_ticket)


class TestEvaluateBatch:
    """Tests for batch ticket evaluation."""

    @pytest.mark.asyncio
    async def test_batch_evaluation(
        self, evaluator: TicketEvaluator, sample_tickets: list[Ticket]
    ) -> None:
        results = await evaluator.evaluate_batch(sample_tickets)

        assert len(results) == len(sample_tickets)
        for result in results:
            assert 1 <= result.content_score <= 5
            assert 1 <= result.format_score <= 5
            assert result.ticket  # Non-empty
            assert result.reply  # Non-empty
