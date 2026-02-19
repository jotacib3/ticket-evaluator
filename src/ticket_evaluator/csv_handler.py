"""CSV file handler for reading tickets and writing evaluation results.

Uses Python's built-in csv module for minimal dependencies.
All data is validated through Pydantic models on read.
"""

import csv
import logging
from pathlib import Path

from ticket_evaluator.exceptions import CSVReadError
from ticket_evaluator.models import EvaluatedTicket, Ticket

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"ticket", "reply"}
OUTPUT_COLUMNS = [
    "ticket",
    "reply",
    "content_score",
    "content_explanation",
    "format_score",
    "format_explanation",
]


def read_tickets(path: Path) -> list[Ticket]:
    """Read and validate tickets from a CSV file.

    Args:
        path: Path to the input CSV file.

    Returns:
        List of validated Ticket objects.

    Raises:
        CSVReadError: If the file is missing, empty, or has invalid structure.
    """
    if not path.exists():
        raise CSVReadError(f"Input file not found: {path}")

    try:
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)

            # Validate required columns exist
            if reader.fieldnames is None:
                raise CSVReadError(f"Empty CSV file: {path}")

            missing = REQUIRED_COLUMNS - set(reader.fieldnames)
            if missing:
                raise CSVReadError(f"Missing required columns: {missing}")

            tickets = []
            for row_num, row in enumerate(reader, start=2):  # Row 1 = header
                ticket_text = row.get("ticket", "").strip()
                reply_text = row.get("reply", "").strip()

                if not ticket_text or not reply_text:
                    logger.warning("Skipping row %d: empty ticket or reply", row_num)
                    continue

                tickets.append(Ticket(ticket=ticket_text, reply=reply_text))

            if not tickets:
                raise CSVReadError("No valid tickets found in the CSV file")

            logger.info("Read %d tickets from %s", len(tickets), path)
            return tickets

    except UnicodeDecodeError as e:
        raise CSVReadError(f"File encoding error in {path}: {e}") from e


def write_evaluated_tickets(tickets: list[EvaluatedTicket], path: Path) -> None:
    """Write evaluated tickets to a CSV file.

    Args:
        tickets: List of evaluated tickets to write.
        path: Path to the output CSV file.
    """
    with open(path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()

        for evaluated in tickets:
            writer.writerow(evaluated.model_dump())

    logger.info("Wrote %d evaluated tickets to %s", len(tickets), path)
