"""Shared pytest fixtures for the ticket evaluator test suite."""

import csv
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ticket_evaluator.models import Ticket

# ─── Sample Data ───

SAMPLE_TICKET_TEXT = "Hi, I'd like to check the shipping status of my order #1234."
SAMPLE_REPLY_TEXT = (
    "Sure, you can check your shipping status on our website. "
    "Your package is scheduled for delivery tomorrow."
)

SAMPLE_LLM_RESPONSE = {
    "content_score": 4,
    "content_explanation": "Addresses the shipping inquiry with a specific delivery timeline.",
    "format_score": 5,
    "format_explanation": "Clear, professional, and well-structured response.",
}


@pytest.fixture
def sample_ticket() -> Ticket:
    """A single sample ticket for testing."""
    return Ticket(ticket=SAMPLE_TICKET_TEXT, reply=SAMPLE_REPLY_TEXT)


@pytest.fixture
def sample_tickets() -> list[Ticket]:
    """Multiple sample tickets for batch testing."""
    return [
        Ticket(
            ticket="Hi, I'd like to check the shipping status of my order #1234.",
            reply="Sure, your package is scheduled for delivery tomorrow.",
        ),
        Ticket(
            ticket="The product I received is defective.",
            reply="We're sorry. Could you please provide a photo of the defect?",
        ),
    ]


@pytest.fixture
def valid_csv_file(tmp_path: Path) -> Path:
    """Create a valid temporary CSV file for testing."""
    csv_path = tmp_path / "test_tickets.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ticket", "reply"])
        writer.writeheader()
        writer.writerow({"ticket": SAMPLE_TICKET_TEXT, "reply": SAMPLE_REPLY_TEXT})
    return csv_path


@pytest.fixture
def empty_csv_file(tmp_path: Path) -> Path:
    """Create an empty CSV file (header only, no data rows)."""
    csv_path = tmp_path / "empty.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ticket", "reply"])
        writer.writeheader()
    return csv_path


@pytest.fixture
def missing_columns_csv(tmp_path: Path) -> Path:
    """Create a CSV file missing required columns."""
    csv_path = tmp_path / "bad_columns.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "message"])
        writer.writeheader()
        writer.writerow({"id": "1", "message": "test"})
    return csv_path


@pytest.fixture
def mock_openai_client() -> AsyncMock:
    """Create a mock AsyncOpenAI client returning a valid JSON response."""
    import json

    client = AsyncMock()

    # Build the mock response chain: client.chat.completions.create()
    mock_message = MagicMock()
    mock_message.content = json.dumps(SAMPLE_LLM_RESPONSE)

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    client.chat.completions.create = AsyncMock(return_value=mock_response)

    return client
