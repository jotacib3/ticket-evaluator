"""Custom exceptions for the ticket evaluator."""


class TicketEvaluatorError(Exception):
    """Base exception for all ticket evaluator errors."""


class CSVReadError(TicketEvaluatorError):
    """Raised when the input CSV file cannot be read or is invalid."""


class EvaluationError(TicketEvaluatorError):
    """Raised when the LLM evaluation fails after all retries."""
