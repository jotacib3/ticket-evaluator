"""Integration test for the full evaluation pipeline.

Tests the end-to-end flow by invoking `main.run()` — the same function the
CLI calls — with monkeypatched file paths and a mocked OpenAI client.
This exercises the complete pipeline: config → CSV read → LLM evaluation →
CSV write, verifying all modules integrate correctly.
"""

import csv
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr

from ticket_evaluator.config import Settings
from ticket_evaluator.models import EvaluationResult


@pytest.fixture
def input_csv(tmp_path: Path) -> Path:
    """Create a realistic input CSV with multiple tickets."""
    csv_path = tmp_path / "tickets.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ticket", "reply"])
        writer.writeheader()
        writer.writerow(
            {
                "ticket": "My order #1234 hasn't arrived yet.",
                "reply": "Your package is on the way and should arrive tomorrow.",
            }
        )
        writer.writerow(
            {
                "ticket": "I was charged twice for my subscription.",
                "reply": "We've issued a refund for the duplicate charge.",
            }
        )
    return csv_path


@pytest.fixture
def mock_responses() -> list[MagicMock]:
    """LLM mock responses with different scores for each ticket."""
    return [
        MagicMock(
            output_parsed=EvaluationResult(
                content_score=4,
                content_explanation="Addresses the issue with a delivery estimate.",
                format_score=4,
                format_explanation="Clear and professional.",
            )
        ),
        MagicMock(
            output_parsed=EvaluationResult(
                content_score=5,
                content_explanation="Fully resolves the double charge.",
                format_score=5,
                format_explanation="Concise and well-structured.",
            )
        ),
    ]


class TestFullPipeline:
    """Integration tests that invoke `main.run()` — the real pipeline entry point."""

    @pytest.mark.asyncio
    async def test_end_to_end_pipeline(
        self,
        input_csv: Path,
        mock_responses: list[MagicMock],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Runs the full pipeline via main.run() and verifies the output CSV."""
        output_csv = tmp_path / "tickets_evaluated.csv"

        # Patch file paths used by main.run()
        monkeypatch.setattr("ticket_evaluator.main.INPUT_FILE", input_csv)
        monkeypatch.setattr("ticket_evaluator.main.OUTPUT_FILE", output_csv)

        # Patch AsyncOpenAI so no real API calls are made
        mock_client = AsyncMock()
        mock_client.responses.parse = AsyncMock(side_effect=mock_responses)
        monkeypatch.setattr(
            "ticket_evaluator.main.AsyncOpenAI", lambda **kwargs: mock_client
        )

        # Build real settings (only openai_api_key is required)
        settings = Settings(
            openai_api_key=SecretStr("test-key"),
            openai_model="gpt-5.2",
            max_concurrency=2,
            max_retries=2,
        )

        # Run the REAL pipeline
        from ticket_evaluator.main import run

        await run(settings)

        # ── Verify output CSV exists and has correct structure ──
        assert output_csv.exists(), "Output CSV was not created"

        with open(output_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert reader.fieldnames == [
            "ticket",
            "reply",
            "content_score",
            "content_explanation",
            "format_score",
            "format_explanation",
        ]

        assert len(rows) == 2

        # ── Verify content: original text preserved, scores correct ──
        assert rows[0]["ticket"] == "My order #1234 hasn't arrived yet."
        assert rows[0]["content_score"] == "4"
        assert rows[0]["format_score"] == "4"

        assert rows[1]["ticket"] == "I was charged twice for my subscription."
        assert rows[1]["content_score"] == "5"
        assert rows[1]["format_score"] == "5"

        # ── Verify the LLM was called exactly once per ticket ──
        assert mock_client.responses.parse.call_count == 2

    @pytest.mark.asyncio
    async def test_pipeline_with_partial_failure(
        self,
        input_csv: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Pipeline completes with partial results when one ticket fails."""
        from openai import APIConnectionError

        output_csv = tmp_path / "tickets_evaluated.csv"

        monkeypatch.setattr("ticket_evaluator.main.INPUT_FILE", input_csv)
        monkeypatch.setattr("ticket_evaluator.main.OUTPUT_FILE", output_csv)

        # First ticket always fails, second succeeds
        success_response = MagicMock(
            output_parsed=EvaluationResult(
                content_score=5,
                content_explanation="Good.",
                format_score=5,
                format_explanation="Clear.",
            )
        )

        async def _mock_parse(**kwargs: object) -> MagicMock:
            user_content = kwargs["input"][0]["content"]
            if "#1234" in user_content:
                raise APIConnectionError(request=MagicMock())
            return success_response

        mock_client = AsyncMock()
        mock_client.responses.parse = AsyncMock(side_effect=_mock_parse)
        monkeypatch.setattr(
            "ticket_evaluator.main.AsyncOpenAI", lambda **kwargs: mock_client
        )

        settings = Settings(
            openai_api_key=SecretStr("test-key"),
            openai_model="gpt-5.2",
            max_concurrency=2,
            max_retries=2,
        )

        from ticket_evaluator.main import run

        await run(settings)

        # Only the successful ticket should appear in output
        with open(output_csv, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 1
        assert rows[0]["ticket"] == "I was charged twice for my subscription."
        assert rows[0]["content_score"] == "5"
