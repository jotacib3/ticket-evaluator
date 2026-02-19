"""Tests for the async LLM ticket evaluator."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import APIConnectionError


from ticket_evaluator.evaluator import TicketEvaluator
from ticket_evaluator.exceptions import EvaluationError
from ticket_evaluator.models import EvaluationResult, Ticket


@pytest.fixture
def evaluator(mock_openai_client: AsyncMock) -> TicketEvaluator:
    """Create a TicketEvaluator with a mocked OpenAI client."""
    return TicketEvaluator(
        client=mock_openai_client,
        model="gpt-5.2",
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
        # Override mock to return None parsed output
        mock_response = MagicMock()
        mock_response.output_parsed = None
        evaluator.client.responses.parse = AsyncMock(return_value=mock_response)

        with pytest.raises(EvaluationError, match="empty response"):
            await evaluator.evaluate(sample_ticket)

    @pytest.mark.asyncio
    async def test_calls_responses_parse(
        self, evaluator: TicketEvaluator, sample_ticket: Ticket
    ) -> None:
        """Verify that the evaluator calls client.responses.parse with correct args."""
        await evaluator.evaluate(sample_ticket)

        evaluator.client.responses.parse.assert_called_once()
        call_kwargs = evaluator.client.responses.parse.call_args[1]
        assert call_kwargs["model"] == "gpt-5.2"
        assert call_kwargs["temperature"] == 0.2
        assert "instructions" in call_kwargs
        assert "text_format" in call_kwargs


class TestEvaluateBatch:
    """Tests for batch ticket evaluation, including fault tolerance."""

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

    @pytest.mark.asyncio
    async def test_batch_partial_failure_returns_successful_results(
        self, mock_openai_client: AsyncMock
    ) -> None:
        """When some tickets fail, the batch still returns successful results."""
        tickets = [
            Ticket(ticket="This ticket will fail.", reply="Failing reply."),
            Ticket(ticket="This ticket will succeed.", reply="Successful reply."),
        ]
        success_response = MagicMock()
        success_response.output_parsed = EvaluationResult(
            content_score=4,
            content_explanation="Good response.",
            format_score=4,
            format_explanation="Well-structured.",
        )

        async def _mock_parse(**kwargs: object) -> MagicMock:
            # Determine success/failure based on the ticket content
            user_content = kwargs["input"][0]["content"]
            if "will fail" in user_content:
                raise APIConnectionError(request=MagicMock())
            return success_response

        mock_openai_client.responses.parse = AsyncMock(side_effect=_mock_parse)
        evaluator = TicketEvaluator(
            client=mock_openai_client, model="gpt-5.2", max_retries=2, max_concurrency=2
        )

        results = await evaluator.evaluate_batch(tickets)

        # The failing ticket is skipped, the successful one is returned
        assert len(results) == 1
        assert results[0].content_score == 4
        assert results[0].ticket == "This ticket will succeed."

    @pytest.mark.asyncio
    async def test_batch_all_failures_returns_empty_list(
        self, evaluator: TicketEvaluator, sample_tickets: list[Ticket]
    ) -> None:
        """When all tickets fail, the batch returns an empty list instead of raising."""
        evaluator.client.responses.parse = AsyncMock(
            side_effect=EvaluationError("Simulated total failure")
        )

        results = await evaluator.evaluate_batch(sample_tickets)

        assert results == []
