# Ticket Evaluator

LLM-powered evaluation of customer support ticket replies. Reads tickets from a CSV, evaluates each AI-generated reply on **Content** and **Format** (1–5 scale), and writes the results to a new CSV.

## Architecture

```
src/ticket_evaluator/
├── main.py           # Pipeline orchestrator
├── models.py         # Pydantic domain models (Ticket, EvaluationResult, EvaluatedTicket)
├── evaluator.py      # Async LLM evaluator with retry + concurrency control
├── prompt.py         # System prompt with scoring rubric + calibration examples
├── csv_handler.py    # CSV read/write with validation
├── config.py         # Settings via environment variables (pydantic-settings)
└── exceptions.py     # Custom exception hierarchy
```

**Design principles**: SRP per module, DIP via constructor-injected OpenAI client (enables mocking in tests), Structured Outputs for guaranteed valid JSON from the LLM.

## Prompt Engineering Approach

The system prompt uses a **rubric-based evaluation** strategy with three key techniques:

1. **Detailed scoring rubric** — explicit criteria for each score level (1–5) on both Content and Format dimensions.
2. **Few-shot calibration** — three annotated examples (strong, weak, average) that anchor the scoring scale and reduce model bias.
3. **Anti-generosity instructions** — explicit directives to use the full 1–5 range and penalize replies that miss the customer's explicit request.

## Prerequisites

- Python ≥ 3.13
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key

## Setup

```bash
# Clone and install
git clone <repo-url> && cd ticket-evaluator
uv sync

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Usage

```bash
# Run evaluation
uv run ticket-evaluator

# Input:  tickets.csv (columns: ticket, reply)
# Output: tickets_evaluated.csv (adds: content_score, content_explanation, format_score, format_explanation)
```

## Testing

```bash
# Run all tests
uv run pytest -v

# With coverage
uv run pytest --cov=ticket_evaluator --cov-report=term-missing

# Lint & format check
uv run ruff check .
uv run ruff format --check .
```

All tests use mocked OpenAI clients — **no real API calls** are made during testing.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | ✅ | — | Your OpenAI API key |
| `OPENAI_MODEL` | ❌ | `gpt-5.2` | Model to use for evaluation |
| `REASONING_EFFORT` | ❌ | `None` | Reasoning effort level (low, medium, high) |
| `MAX_CONCURRENCY` | ❌ | `3` | Max concurrent API calls |
| `MAX_RETRIES` | ❌ | `3` | Max retry attempts per ticket |

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Async evaluation | `AsyncOpenAI` + `asyncio.gather` | Concurrent API calls with semaphore-based rate limiting |
| Output parsing | Structured Outputs (Pydantic) | Type-safe, validated responses — no regex/text parsing |
| Temperature | `0.2` | Low enough for consistency, high enough to avoid mode-seeking bias |
| Fault tolerance | Per-ticket error handling | Failed tickets are logged and skipped; successful results preserved |
| Dependencies | Minimal (`openai`, `pydantic`, `pydantic-settings`) | stdlib `csv` module — no pandas needed |
