"""Entry point for the ticket evaluator.

Orchestrates the full evaluation pipeline:
1. Load configuration from environment
2. Read tickets from CSV
3. Evaluate each reply using the OpenAI LLM (async, concurrent)
4. Write results to output CSV

Usage:
    ticket-evaluator
"""

import asyncio
import logging
import sys

from openai import AsyncOpenAI

from ticket_evaluator.config import INPUT_FILE, OUTPUT_FILE, Settings
from ticket_evaluator.csv_handler import read_tickets, write_evaluated_tickets
from ticket_evaluator.evaluator import TicketEvaluator
from ticket_evaluator.exceptions import TicketEvaluatorError

# ─── Logging Configuration ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


async def run(settings: Settings) -> None:
    """Execute the full evaluation pipeline.

    Args:
        settings: Application settings with API key and model configuration.
    """
    # 1. Read tickets
    logger.info("Reading tickets from: %s", INPUT_FILE)
    tickets = read_tickets(INPUT_FILE)
    logger.info("Found %d tickets to evaluate", len(tickets))

    # 2. Initialize OpenAI client and evaluator
    client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
    evaluator = TicketEvaluator(
        client=client,
        model=settings.openai_model,
        max_retries=settings.max_retries,
        max_concurrency=settings.max_concurrency,
        reasoning_effort=settings.reasoning_effort,
    )

    # 3. Evaluate all tickets concurrently
    logger.info("Evaluating tickets using model: %s", settings.openai_model)
    evaluated_tickets = await evaluator.evaluate_batch(tickets)

    # 4. Write results
    write_evaluated_tickets(evaluated_tickets, OUTPUT_FILE)
    logger.info("✅ Evaluation complete! Results saved to: %s", OUTPUT_FILE)

    # Print summary
    if evaluated_tickets:
        avg_content = sum(t.content_score for t in evaluated_tickets) / len(
            evaluated_tickets
        )
        avg_format = sum(t.format_score for t in evaluated_tickets) / len(
            evaluated_tickets
        )
        logger.info(
            "📊 Summary — Tickets: %d | Avg Content: %.1f | Avg Format: %.1f",
            len(evaluated_tickets),
            avg_content,
            avg_format,
        )


def main() -> None:
    """Application entry point. Loads config and runs the evaluation pipeline."""
    try:
        settings = Settings()
        asyncio.run(run(settings))

    except TicketEvaluatorError as e:
        logger.error("❌ %s", e)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        logger.error("❌ Unexpected error: %s", e, exc_info=True)
        sys.exit(1)
