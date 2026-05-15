"""
Bench'd Budget Curves — tests accuracy at different token budgets.

Runs the same questions but limits how much context the system can use.
Reveals whether a system is precise (high accuracy at low budget) or
brute-forcing (only accurate with huge context).

Tiers:
  - 500 tokens
  - 1,000 tokens
  - 2,000 tokens
  - 5,000 tokens
  - 10,000 tokens (unlimited)
"""

from typing import List
from benchd_harness.benchmarks.base import BaseBenchmark, BenchmarkItem


# Core questions that work at any budget level
BUDGET_QUESTIONS = [
    {
        "id": "budget_001",
        "turns": [
            {"role": "user", "content": "I'm a software engineer working at a startup in San Francisco. We use React for the frontend and Python for the backend."},
            {"role": "assistant", "content": "That's a solid tech stack! What kind of startup?"},
            {"role": "user", "content": "We build developer tools. My team has 5 people. I've been there 2 years."},
        ],
        "query": "What tech stack does the user's company use?",
        "expected": "react",
    },
    {
        "id": "budget_002",
        "turns": [
            {"role": "user", "content": "My daughter Emma just turned 7. She loves painting and soccer."},
            {"role": "assistant", "content": "She sounds very active and creative!"},
            {"role": "user", "content": "Yes, she just won a painting competition at school. We're very proud."},
        ],
        "query": "How old is Emma?",
        "expected": "7",
    },
    {
        "id": "budget_003",
        "turns": [
            {"role": "user", "content": "I just bought a Tesla Model 3. My old car was a Honda Civic."},
            {"role": "assistant", "content": "Nice upgrade! How are you liking it?"},
            {"role": "user", "content": "The autopilot is amazing. Paid $42,000 for it."},
        ],
        "query": "How much did the user pay for their car?",
        "expected": "42,000",
    },
    {
        "id": "budget_004",
        "turns": [
            {"role": "user", "content": "I run 5K every morning at 6am. Started running 3 years ago."},
        ],
        "query": "What time does the user run?",
        "expected": "6am",
    },
    {
        "id": "budget_005",
        "turns": [
            {"role": "user", "content": "I'm reading 'Project Hail Mary' by Andy Weir. About halfway through."},
            {"role": "assistant", "content": "Great book! What do you think so far?"},
            {"role": "user", "content": "It's incredible. Best sci-fi I've read in years."},
        ],
        "query": "What book is the user reading?",
        "expected": "project hail mary",
    },
]

BUDGET_TIERS = [500, 1000, 2000, 5000, 10000]


class BudgetCurvesBenchmark(BaseBenchmark):
    """Tests accuracy at different context token budgets."""

    @property
    def slug(self) -> str:
        return "budget-curves-v1"

    @property
    def name(self) -> str:
        return "Budget Curves"

    @property
    def version(self) -> str:
        return "1.0"

    @property
    def description(self) -> str:
        return "Tests accuracy at 500/1K/2K/5K/10K token budgets. Reveals precision vs brute-force."

    def __init__(self, max_items: int | None = None):
        self._max_items = max_items

    def load_items(self) -> List[BenchmarkItem]:
        items = []

        for budget in BUDGET_TIERS:
            for q in BUDGET_QUESTIONS:
                items.append(BenchmarkItem(
                    question_id=f"{q['id']}_budget_{budget}",
                    dimension=f"budget_{budget}",
                    ingest_history=q["turns"],
                    recall_query=q["query"],
                    expected_answer=q["expected"],
                    scoring_method="exact",
                    max_score=1.0,
                    metadata={"token_budget": budget},
                ))

        if self._max_items:
            items = items[:self._max_items]

        return items

    def get_dimensions(self) -> List[str]:
        return [f"budget_{b}" for b in BUDGET_TIERS]
