"""Domain models for ticket evaluation.

Defines the data structures used throughout the application:
- Ticket: represents a customer support ticket with its AI-generated reply.
- EvaluationResult: the LLM's assessment of a reply (scores + explanations).
- EvaluatedTicket: combines a ticket with its evaluation for CSV output.
"""

from pydantic import BaseModel, Field


class Ticket(BaseModel):
    """Represents a customer support ticket and its AI-generated reply."""

    ticket: str = Field(description="The customer's support message")
    reply: str = Field(description="The AI-generated response to the ticket")


class EvaluationResult(BaseModel):
    """LLM evaluation of a ticket reply on content and format dimensions.

    Scores use a 1-5 scale where:
        1 = Poor/Inadequate
        5 = Excellent/Outstanding
    """

    content_score: int = Field(ge=1, le=5, description="Content quality score (1-5)")
    content_explanation: str = Field(
        description="Brief explanation of the content score"
    )
    format_score: int = Field(ge=1, le=5, description="Format quality score (1-5)")
    format_explanation: str = Field(description="Brief explanation of the format score")


class EvaluatedTicket(BaseModel):
    """A ticket combined with its evaluation result, ready for CSV output."""

    ticket: str
    reply: str
    content_score: int
    content_explanation: str
    format_score: int
    format_explanation: str

    @classmethod
    def from_ticket_and_result(
        cls, ticket: Ticket, result: EvaluationResult
    ) -> "EvaluatedTicket":
        """Create an EvaluatedTicket by combining a Ticket with its EvaluationResult."""
        return cls(
            ticket=ticket.ticket,
            reply=ticket.reply,
            content_score=result.content_score,
            content_explanation=result.content_explanation,
            format_score=result.format_score,
            format_explanation=result.format_explanation,
        )
