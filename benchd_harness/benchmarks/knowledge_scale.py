"""
Bench'd Knowledge Scale Benchmark — tests knowledge brains at increasing page counts.

Tests retrieval accuracy and speed as the knowledge base grows:
- 10 pages (baseline)
- 50 pages (small vault)
- 100 pages (medium vault)

Each tier ingests N pages then queries for specific facts buried in them.
Measures: accuracy, latency degradation, false positive rate.
"""

import uuid
from typing import List

from benchd_harness.benchmarks.base import BaseBenchmark, BenchmarkItem


# Generate filler pages with unique identifiable facts
TEAMS = [
    ("Alpha", "payments", "Maria", "React", "PostgreSQL", "AWS"),
    ("Beta", "search", "Tom", "Vue", "MongoDB", "GCP"),
    ("Gamma", "auth", "Lisa", "Angular", "Redis", "Azure"),
    ("Delta", "analytics", "James", "Svelte", "ClickHouse", "Vercel"),
    ("Epsilon", "notifications", "Nina", "Next.js", "SQLite", "Fly.io"),
    ("Zeta", "billing", "Oscar", "Remix", "MySQL", "Railway"),
    ("Eta", "onboarding", "Priya", "Solid", "DynamoDB", "Render"),
    ("Theta", "reporting", "Chen", "Astro", "CockroachDB", "Netlify"),
    ("Iota", "integrations", "Alex", "Nuxt", "Firestore", "DigitalOcean"),
    ("Kappa", "security", "Dana", "Qwik", "ScyllaDB", "Hetzner"),
]

PRODUCTS = [
    ("Widget Pro", "$29/mo", "10K users", "launched Jan 2025"),
    ("DataSync", "$99/mo", "2K users", "launched Mar 2025"),
    ("CloudBridge", "$199/mo", "500 users", "launched Jun 2025"),
    ("APIGuard", "$49/mo", "8K users", "launched Sep 2025"),
    ("FormFlow", "$19/mo", "25K users", "launched Dec 2025"),
]

PEOPLE = [
    ("Alice Johnson", "VP Engineering", "alice@company.com", "likes hiking"),
    ("Bob Martinez", "CTO", "bob@company.com", "plays guitar"),
    ("Carol Wu", "Head of Product", "carol@company.com", "runs marathons"),
    ("Dave Kim", "Director of Design", "dave@company.com", "collects vinyl"),
    ("Eve Patel", "Head of Data", "eve@company.com", "speaks 4 languages"),
]

def _generate_pages(count: int) -> list[dict]:
    """Generate N distinct pages with unique queryable facts."""
    pages = []

    # Team pages
    for i, (name, service, lead, frontend, db, cloud) in enumerate(TEAMS):
        if len(pages) >= count:
            break
        pages.append({
            "content": f"Team {name} works on the {service} service. Team lead: {lead}. "
                      f"Tech stack: {frontend} frontend, {db} database, deployed on {cloud}. "
                      f"Team size: {i + 3} engineers. Founded Q{(i % 4) + 1} 2025.",
            "fact_query": f"Who leads team {name}?",
            "fact_answer": lead.lower(),
        })

    # Product pages
    for name, price, users, launched in PRODUCTS:
        if len(pages) >= count:
            break
        pages.append({
            "content": f"Product: {name}. Price: {price}. Active users: {users}. {launched.capitalize()}. "
                      f"Built by the product team. Available worldwide.",
            "fact_query": f"How much does {name} cost?",
            "fact_answer": price.lower().replace("/mo", ""),
        })

    # People pages
    for name, title, email, hobby in PEOPLE:
        if len(pages) >= count:
            break
        pages.append({
            "content": f"Employee profile: {name}, {title}. Email: {email}. "
                      f"Hobby: {hobby}. Joined the company in 2024.",
            "fact_query": f"What is {name}'s title?",
            "fact_answer": title.lower(),
        })

    # Filler pages to reach count
    while len(pages) < count:
        idx = len(pages)
        pages.append({
            "content": f"Meeting notes #{idx}: Discussed quarterly planning. Action items include "
                      f"reviewing the {TEAMS[idx % len(TEAMS)][1]} service roadmap and updating "
                      f"documentation for {PRODUCTS[idx % len(PRODUCTS)][0]}. Budget: ${idx * 1000 + 5000}.",
            "fact_query": f"What was the budget discussed in meeting #{idx}?",
            "fact_answer": str(idx * 1000 + 5000),
        })

    return pages


# Scale tiers
SCALE_TIERS = [
    {"name": "small", "page_count": 10, "query_count": 5},
    {"name": "medium", "page_count": 50, "query_count": 5},
    {"name": "large", "page_count": 100, "query_count": 5},
]


class KnowledgeScaleBenchmark(BaseBenchmark):
    """Tests knowledge brain accuracy as vault size increases."""

    @property
    def slug(self) -> str:
        return "knowledge-scale-v1"

    @property
    def name(self) -> str:
        return "Knowledge Scale"

    @property
    def version(self) -> str:
        return "1.0"

    @property
    def description(self) -> str:
        return "Tests retrieval accuracy at 10, 50, and 100 stored pages."

    def __init__(self, max_items: int | None = None):
        self._max_items = max_items

    def load_items(self) -> List[BenchmarkItem]:
        items = []

        for tier in SCALE_TIERS:
            pages = _generate_pages(tier["page_count"])

            # All pages get ingested as conversation turns
            all_turns = [
                {"role": "user", "content": p["content"]}
                for p in pages
            ]

            # Query a subset of pages
            query_pages = pages[:tier["query_count"]]

            for i, page in enumerate(query_pages):
                items.append(BenchmarkItem(
                    question_id=f"scale_{tier['name']}_{i}",
                    dimension=f"scale_{tier['name']}",
                    ingest_history=all_turns,
                    recall_query=page["fact_query"],
                    expected_answer=page["fact_answer"],
                    scoring_method="exact",
                    max_score=1.0,
                    metadata={"page_count": tier["page_count"]},
                ))

        if self._max_items:
            items = items[:self._max_items]

        return items

    def get_dimensions(self) -> List[str]:
        return ["scale_small", "scale_medium", "scale_large"]
