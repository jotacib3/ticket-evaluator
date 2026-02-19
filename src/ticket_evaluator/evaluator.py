"""Async LLM-based ticket reply evaluator.

Uses the OpenAI Responses API with Structured Outputs to evaluate
customer support ticket replies on content and format dimensions.
Supports concurrent evaluation with rate-limiting via semaphore.
"""

import asyncio
import logging

from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError

from ticket_evaluator.exceptions import EvaluationError
from ticket_evaluator.models import EvaluatedTicket, EvaluationResult, Ticket
from ticket_evaluator.prompt import SYSTEM_PROMPT, build_user_prompt

logger = logging.getLogger(__name__)


class TicketEvaluator:
    """Evaluates ticket replies using an OpenAI LLM.

    Attributes:
        client: An AsyncOpenAI client instance (injected for testability).
        model: The OpenAI model identifier to use.
        max_retries: Maximum number of retry attempts for transient failures.
        max_concurrency: Maximum number of concurrent API calls.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        max_retries: int = 3,
        max_concurrency: int = 3,
        reasoning_effort: str | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.max_retries = max_retries
        self.reasoning_effort = reasoning_effort
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def evaluate(self, ticket: Ticket) -> EvaluationResult:
        """Evaluate a single ticket reply using the LLM.

        Sends the ticket and reply to the LLM with a detailed scoring rubric
        and parses the structured JSON response.

        Args:
            ticket: The ticket containing the customer message and AI reply.

        Returns:
            EvaluationResult with scores and explanations.

        Raises:
            EvaluationError: If evaluation fails after all retry attempts.
        """
        user_prompt = build_user_prompt(ticket.ticket, ticket.reply)
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                async with self._semaphore:
                    kwargs: dict = {
                        "model": self.model,
                        "instructions": SYSTEM_PROMPT,
                        "input": [
                            {"role": "user", "content": user_prompt},
                        ],
                        "text_format": EvaluationResult,
                        "temperature": 0.0,  # Deterministic for consistent scoring
                    }
                    if self.reasoning_effort:
                        kwargs["reasoning"] = {"effort": self.reasoning_effort}

                    response = await self.client.responses.parse(**kwargs)

                result = response.output_parsed
                if result is None:
                    raise EvaluationError("LLM returned empty response")

                logger.info(
                    "Evaluated ticket (content=%d, format=%d): %.50s...",
                    result.content_score,
                    result.format_score,
                    ticket.ticket,
                )
                return result

            except (RateLimitError, APIConnectionError, APIStatusError) as e:
                last_error = e
                if attempt < self.max_retries:
                    wait_time = 2**attempt  # Exponential backoff: 2s, 4s, 8s
                    logger.warning(
                        "API error (attempt %d/%d), retrying in %ds: %s",
                        attempt,
                        self.max_retries,
                        wait_time,
                        str(e),
                    )
                    await asyncio.sleep(wait_time)

        raise EvaluationError(
            f"Evaluation failed after {self.max_retries} attempts: {last_error}"
        )

    async def evaluate_batch(self, tickets: list[Ticket]) -> list[EvaluatedTicket]:
        """Evaluate multiple tickets concurrently.

        Uses asyncio.gather with a semaphore to limit concurrent API calls
        and avoid rate-limiting issues.

        Args:
            tickets: List of tickets to evaluate.

        Returns:
            List of EvaluatedTicket objects with scores and explanations.
        """

        async def _evaluate_single(ticket: Ticket) -> EvaluatedTicket:
            result = await self.evaluate(ticket)
            return EvaluatedTicket.from_ticket_and_result(ticket, result)

        logger.info("Starting batch evaluation of %d tickets...", len(tickets))

        evaluated = await asyncio.gather(
            *[_evaluate_single(ticket) for ticket in tickets]
        )

        logger.info("Batch evaluation complete. %d tickets processed.", len(evaluated))
        return list(evaluated)
