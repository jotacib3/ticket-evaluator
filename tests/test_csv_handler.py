"""Tests for CSV file handler."""

import csv
from pathlib import Path

import pytest

from ticket_evaluator.csv_handler import read_tickets, write_evaluated_tickets
from ticket_evaluator.exceptions import CSVReadError
from ticket_evaluator.models import EvaluatedTicket


class TestReadTickets:
    """Tests for the read_tickets function."""

    def test_read_valid_csv(self, valid_csv_file: Path) -> None:
        tickets = read_tickets(valid_csv_file)
        assert len(tickets) == 1
        assert "shipping status" in tickets[0].ticket

    def test_file_not_found_raises_error(self, tmp_path: Path) -> None:
        with pytest.raises(CSVReadError, match="not found"):
            read_tickets(tmp_path / "nonexistent.csv")

    def test_empty_csv_raises_error(self, empty_csv_file: Path) -> None:
        with pytest.raises(CSVReadError, match="No valid tickets"):
            read_tickets(empty_csv_file)

    def test_missing_columns_raises_error(self, missing_columns_csv: Path) -> None:
        with pytest.raises(CSVReadError, match="Missing required columns"):
            read_tickets(missing_columns_csv)

    def test_skips_rows_with_empty_fields(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "partial.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["ticket", "reply"])
            writer.writeheader()
            writer.writerow({"ticket": "Valid ticket", "reply": "Valid reply"})
            writer.writerow({"ticket": "", "reply": "Reply without ticket"})

        tickets = read_tickets(csv_path)
        assert len(tickets) == 1
        assert tickets[0].ticket == "Valid ticket"


class TestWriteEvaluatedTickets:
    """Tests for the write_evaluated_tickets function."""

    def test_write_creates_valid_csv(self, tmp_path: Path) -> None:
        output_path = tmp_path / "output.csv"
        evaluated = [
            EvaluatedTicket(
                ticket="Test ticket",
                reply="Test reply",
                content_score=4,
                content_explanation="Good content",
                format_score=5,
                format_explanation="Great format",
            )
        ]

        write_evaluated_tickets(evaluated, output_path)

        assert output_path.exists()
        with open(output_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["content_score"] == "4"
        assert rows[0]["format_explanation"] == "Great format"

    def test_output_has_correct_columns(self, tmp_path: Path) -> None:
        output_path = tmp_path / "output.csv"
        evaluated = [
            EvaluatedTicket(
                ticket="T",
                reply="R",
                content_score=3,
                content_explanation="C",
                format_score=3,
                format_explanation="F",
            )
        ]

        write_evaluated_tickets(evaluated, output_path)

        with open(output_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == [
                "ticket",
                "reply",
                "content_score",
                "content_explanation",
                "format_score",
                "format_explanation",
            ]
