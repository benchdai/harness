"""Smoke benchmark — a harness test fixture with 10 realistic memory scenarios."""

from typing import List

from .base import BaseBenchmark, BenchmarkItem


class SmokeBenchmark(BaseBenchmark):
    """
    Harness test fixture (NOT a real benchmark).

    Ten questions across three dimensions that exercise the harness plumbing:
    recall (4), temporal (3), reasoning (3).
    """

    @property
    def slug(self) -> str:
        return "smoke-memory-v0"

    @property
    def name(self) -> str:
        return "Smoke Memory v0"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def description(self) -> str:
        return (
            "Harness test fixture with 10 realistic memory scenarios. "
            "Verifies ingestion, recall, temporal reasoning, and LLM scoring plumbing."
        )

    def load_items(self) -> List[BenchmarkItem]:
        return [
            self._q1_user_name(),
            self._q2_project_budget(),
            self._q3_workplace(),
            self._q4_email(),
            self._q5_language_change(),
            self._q6_relocation(),
            self._q7_deadline_change(),
            self._q8_vacation_preference(),
            self._q9_dietary_restrictions(),
            self._q10_career_situation(),
        ]

    # ------------------------------------------------------------------
    # Recall dimension (4 questions, exact match)
    # ------------------------------------------------------------------

    @staticmethod
    def _q1_user_name() -> BenchmarkItem:
        return BenchmarkItem(
            question_id="smoke-recall-001",
            dimension="recall",
            scoring_method="exact",
            recall_query="What is the user's name?",
            expected_answer="Marcus Chen",
            ingest_history=[
                {
                    "role": "user",
                    "content": (
                        "Hey, I just got back from vacation and my inbox is a disaster. "
                        "Can you help me draft a quick OOO follow-up to my team?"
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "Sure, I can help with that! Could you tell me your name so I "
                        "can address it properly?"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "It's Marcus Chen. Just something casual is fine \u2014 we're a "
                        "pretty informal team."
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "Got it, Marcus. Here's a draft:\n\n"
                        '"Hi team, Marcus here \u2014 back from PTO and working through '
                        "the backlog. I'll be caught up by EOD tomorrow. Flag anything "
                        'urgent and I\'ll prioritize it. Thanks!"'
                    ),
                },
                {
                    "role": "user",
                    "content": "Perfect, that sounds like me. Send it off.",
                },
            ],
        )

    @staticmethod
    def _q2_project_budget() -> BenchmarkItem:
        return BenchmarkItem(
            question_id="smoke-recall-002",
            dimension="recall",
            scoring_method="exact",
            recall_query="What was the budget for the Q2 project?",
            expected_answer="$150,000",
            ingest_history=[
                {
                    "role": "user",
                    "content": (
                        "I need to put together a slide for the Q2 project review. "
                        "Can you help me outline the key talking points?"
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "Absolutely. What are the main areas you'd like to cover? "
                        "Timeline, deliverables, budget?"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Yeah, all of those. The budget was $150,000 and we came in "
                        "slightly under, which is great. Deliverables were the new "
                        "ingestion pipeline and the dashboard redesign."
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "Nice work coming in under budget. I'll frame it around three "
                        "pillars: timeline adherence, budget efficiency at $150k, and "
                        "the two major deliverables. Want me to draft the slide copy?"
                    ),
                },
                {
                    "role": "user",
                    "content": "Yes please, keep it concise \u2014 three bullet points max per section.",
                },
            ],
        )

    @staticmethod
    def _q3_workplace() -> BenchmarkItem:
        return BenchmarkItem(
            question_id="smoke-recall-003",
            dimension="recall",
            scoring_method="exact",
            recall_query="Where does the user work?",
            expected_answer="Anthropic",
            ingest_history=[
                {
                    "role": "user",
                    "content": (
                        "I've been swamped with alignment research papers this week. "
                        "Can you summarize the key takeaways from the latest "
                        "Constitutional AI paper?"
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "Sure! Are you looking at this from a research perspective or "
                        "more for practical implementation?"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Research perspective \u2014 I'm a research engineer at Anthropic, "
                        "so I need to understand the technical details for an internal "
                        "reading group I'm running."
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "Got it, I'll keep it technical then. The paper introduces a "
                        "two-phase approach to alignment: first a supervised phase "
                        "using human feedback, then a reinforcement phase guided by "
                        "a set of constitutional principles..."
                    ),
                },
            ],
        )

    @staticmethod
    def _q4_email() -> BenchmarkItem:
        return BenchmarkItem(
            question_id="smoke-recall-004",
            dimension="recall",
            scoring_method="exact",
            recall_query="What is the user's email address?",
            expected_answer="marcus.chen@example.com",
            ingest_history=[
                {
                    "role": "user",
                    "content": (
                        "I want to sign up for that ML newsletter you mentioned last time. "
                        "What's the link again?"
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "The newsletter is 'The Gradient' \u2014 you can sign up at "
                        "thegradient.pub. Want me to walk you through it?"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "No, I'll handle it. Just use marcus.chen@example.com for "
                        "anything you need to send my way. That's my main address."
                    ),
                },
                {
                    "role": "assistant",
                    "content": "Noted \u2014 I'll keep that on file for future reference.",
                },
                {
                    "role": "user",
                    "content": "Thanks. Also, can you remind me tomorrow to check the submission portal?",
                },
            ],
        )

    # ------------------------------------------------------------------
    # Temporal dimension (3 questions, regex)
    # ------------------------------------------------------------------

    @staticmethod
    def _q5_language_change() -> BenchmarkItem:
        return BenchmarkItem(
            question_id="smoke-temporal-001",
            dimension="temporal",
            scoring_method="regex",
            recall_query="What programming language does the user currently prefer?",
            expected_answer=r"[Rr]ust",
            ingest_history=[
                # --- earlier session ---
                {
                    "role": "user",
                    "content": (
                        "I've been writing everything in Python for the last five years. "
                        "It's my go-to for prototyping and data work."
                    ),
                    "metadata": {"session": "session-a"},
                },
                {
                    "role": "assistant",
                    "content": (
                        "Python is hard to beat for iteration speed. Are you using it "
                        "for production systems too?"
                    ),
                    "metadata": {"session": "session-a"},
                },
                {
                    "role": "user",
                    "content": (
                        "Mostly, yeah. Though I've been curious about Rust for the "
                        "performance-critical stuff."
                    ),
                    "metadata": {"session": "session-a"},
                },
                # --- later session ---
                {
                    "role": "user",
                    "content": (
                        "Okay, I've officially made the switch. I've been writing Rust "
                        "full-time for the past two months and I'm not going back. "
                        "The type system alone is worth it."
                    ),
                    "metadata": {"session": "session-b"},
                },
                {
                    "role": "assistant",
                    "content": (
                        "That's a big shift! What pushed you over the edge \u2014 the "
                        "performance or the safety guarantees?"
                    ),
                    "metadata": {"session": "session-b"},
                },
                {
                    "role": "user",
                    "content": (
                        "Both, honestly. But the compiler catching bugs before runtime "
                        "is what sold me. Rust is my primary language now."
                    ),
                    "metadata": {"session": "session-b"},
                },
            ],
        )

    @staticmethod
    def _q6_relocation() -> BenchmarkItem:
        return BenchmarkItem(
            question_id="smoke-temporal-002",
            dimension="temporal",
            scoring_method="regex",
            recall_query="Where does the user currently live?",
            expected_answer=r"[Aa]ustin",
            ingest_history=[
                # --- earlier session ---
                {
                    "role": "user",
                    "content": (
                        "Traffic on the Bay Bridge was brutal this morning. "
                        "I love San Francisco but the commute is killing me."
                    ),
                    "metadata": {"session": "session-a"},
                },
                {
                    "role": "assistant",
                    "content": (
                        "Have you considered remote work or moving closer to the office?"
                    ),
                    "metadata": {"session": "session-a"},
                },
                {
                    "role": "user",
                    "content": (
                        "Actually, my partner and I have been talking about relocating. "
                        "Austin keeps coming up \u2014 lower cost of living, good tech scene."
                    ),
                    "metadata": {"session": "session-a"},
                },
                # --- later session ---
                {
                    "role": "user",
                    "content": (
                        "We did it \u2014 moved to Austin last month. Still getting settled "
                        "but it already feels right. The extra space is amazing."
                    ),
                    "metadata": {"session": "session-b"},
                },
                {
                    "role": "assistant",
                    "content": "Congrats on the move! How's the transition going work-wise?",
                    "metadata": {"session": "session-b"},
                },
                {
                    "role": "user",
                    "content": (
                        "Smooth \u2014 I'm fully remote now so it didn't affect much. "
                        "Austin's been great so far."
                    ),
                    "metadata": {"session": "session-b"},
                },
            ],
        )

    @staticmethod
    def _q7_deadline_change() -> BenchmarkItem:
        return BenchmarkItem(
            question_id="smoke-temporal-003",
            dimension="temporal",
            scoring_method="regex",
            recall_query="What is the current project deadline?",
            expected_answer=r"[Aa]pril\s*1",
            ingest_history=[
                {
                    "role": "user",
                    "content": (
                        "We need to have the v2 migration done by March 15. "
                        "Can you help me build out a task breakdown?"
                    ),
                    "metadata": {"session": "session-a"},
                },
                {
                    "role": "assistant",
                    "content": (
                        "Sure \u2014 March 15 is tight. Let me sketch out the critical path. "
                        "How many engineers are on this?"
                    ),
                    "metadata": {"session": "session-a"},
                },
                {
                    "role": "user",
                    "content": (
                        "Three, including me. We already have the schema migration "
                        "drafted but the data backfill is the risky part."
                    ),
                    "metadata": {"session": "session-a"},
                },
                # --- later session: deadline slipped ---
                {
                    "role": "user",
                    "content": (
                        "Update on the migration \u2014 we hit some snags with the backfill "
                        "and leadership agreed to push the deadline to April 1. "
                        "So we have a bit more breathing room."
                    ),
                    "metadata": {"session": "session-b"},
                },
                {
                    "role": "assistant",
                    "content": (
                        "That's a relief. Want me to update the task breakdown with "
                        "the new April 1 deadline?"
                    ),
                    "metadata": {"session": "session-b"},
                },
                {
                    "role": "user",
                    "content": "Yes, please. And flag anything that's still at risk.",
                    "metadata": {"session": "session-b"},
                },
            ],
        )

    # ------------------------------------------------------------------
    # Reasoning dimension (3 questions, LLM-scored)
    # ------------------------------------------------------------------

    @staticmethod
    def _q8_vacation_preference() -> BenchmarkItem:
        return BenchmarkItem(
            question_id="smoke-reasoning-001",
            dimension="reasoning",
            scoring_method="llm",
            recall_query=(
                "Based on the user's stated preferences, would they prefer a "
                "beach vacation or a mountain hiking trip?"
            ),
            expected_answer="beach vacation",
            ingest_history=[
                {
                    "role": "user",
                    "content": (
                        "I'm trying to plan something for August. I keep going back "
                        "and forth between a few ideas."
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "What kind of vibe are you going for \u2014 active adventure, "
                        "relaxation, cultural exploration?"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Definitely relaxation. I've been running on fumes and I just "
                        "want to decompress. Somewhere warm where I can read a book "
                        "and not think about work."
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "Sounds like a beach destination might be a good fit. "
                        "Any preferences on domestic vs. international?"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "International would be cool but honestly I'd be happy with "
                        "anywhere with warm water and good food. I loved Tulum last "
                        "year \u2014 that kind of pace. Nothing too rugged or outdoorsy, "
                        "my knees can't take it anymore."
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "Got it \u2014 warm, laid-back, good food, easy on the body. "
                        "I'll put together a few options."
                    ),
                },
            ],
        )

    @staticmethod
    def _q9_dietary_restrictions() -> BenchmarkItem:
        return BenchmarkItem(
            question_id="smoke-reasoning-002",
            dimension="reasoning",
            scoring_method="llm",
            recall_query=(
                "What dietary considerations should be kept in mind when "
                "recommending restaurants to this user?"
            ),
            expected_answer=(
                "The user is allergic to shellfish and prefers vegetarian options"
            ),
            ingest_history=[
                {
                    "role": "user",
                    "content": (
                        "We're doing a team dinner next week and I got put in charge "
                        "of picking the restaurant. Any ideas for downtown?"
                    ),
                    "metadata": {"session": "session-a"},
                },
                {
                    "role": "assistant",
                    "content": (
                        "Sure! Any dietary restrictions I should keep in mind for "
                        "the group or for you personally?"
                    ),
                    "metadata": {"session": "session-a"},
                },
                {
                    "role": "user",
                    "content": (
                        "For the group, no idea \u2014 I'll ask around. For me, I'm "
                        "allergic to shellfish, so seafood-heavy places are tricky. "
                        "I can usually find something but it's stressful."
                    ),
                    "metadata": {"session": "session-a"},
                },
                # --- later session ---
                {
                    "role": "user",
                    "content": (
                        "Hey, I've been trying to eat more plant-based lately. "
                        "Can you suggest some vegetarian recipes that are actually "
                        "filling? I keep ending up hungry an hour later."
                    ),
                    "metadata": {"session": "session-b"},
                },
                {
                    "role": "assistant",
                    "content": (
                        "Definitely \u2014 the trick is protein and fat. Lentil stews, "
                        "chickpea curries, and grain bowls with tahini are great. "
                        "Are you going fully vegetarian?"
                    ),
                    "metadata": {"session": "session-b"},
                },
                {
                    "role": "user",
                    "content": (
                        "Mostly, yeah. I still eat eggs occasionally but I've been "
                        "preferring vegetarian options for a couple months now. "
                        "Feels better overall."
                    ),
                    "metadata": {"session": "session-b"},
                },
            ],
        )

    @staticmethod
    def _q10_career_situation() -> BenchmarkItem:
        return BenchmarkItem(
            question_id="smoke-reasoning-003",
            dimension="reasoning",
            scoring_method="llm",
            recall_query=(
                "Summarize the user's career situation and what they're looking for."
            ),
            expected_answer=(
                "The user is a research engineer feeling limited by their current "
                "role and exploring senior positions at startups"
            ),
            ingest_history=[
                {
                    "role": "user",
                    "content": (
                        "I've been thinking about my career a lot lately. "
                        "I like what I do but I feel like I've hit a ceiling."
                    ),
                    "metadata": {"session": "session-a"},
                },
                {
                    "role": "assistant",
                    "content": (
                        "That's a common feeling after a few years in one role. "
                        "What's driving the frustration \u2014 scope, compensation, growth?"
                    ),
                    "metadata": {"session": "session-a"},
                },
                {
                    "role": "user",
                    "content": (
                        "Scope, mostly. I'm a research engineer but I spend half my "
                        "time on infrastructure grunt work that doesn't move the needle. "
                        "I want to be closer to the actual research decisions."
                    ),
                    "metadata": {"session": "session-a"},
                },
                # --- later session ---
                {
                    "role": "user",
                    "content": (
                        "I had coffee with a friend who just joined a Series A startup "
                        "as a senior research engineer. The way she described the role "
                        "\u2014 small team, direct impact, choosing your own research "
                        "direction \u2014 it sounds like exactly what I want."
                    ),
                    "metadata": {"session": "session-b"},
                },
                {
                    "role": "assistant",
                    "content": (
                        "Startups can definitely offer that kind of autonomy. "
                        "Are you actively looking or just exploring?"
                    ),
                    "metadata": {"session": "session-b"},
                },
                {
                    "role": "user",
                    "content": (
                        "Somewhere in between. I've been browsing senior roles at "
                        "startups and I updated my resume last weekend. Not in a rush "
                        "but I'm keeping my eyes open for the right fit."
                    ),
                    "metadata": {"session": "session-b"},
                },
            ],
        )
