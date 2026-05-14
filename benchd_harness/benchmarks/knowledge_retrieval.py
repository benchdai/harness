"""
Bench'd Knowledge Retrieval Benchmark — tests for knowledge brain systems.

Evaluates document storage, semantic search, fact retrieval from pages,
and knowledge update tracking. Designed for systems like gbrain, Graphiti,
Microsoft GraphRAG, Obsidian Smart Connections, and similar tools.

These systems store documents/pages rather than conversation turns,
so the test format is different from LongMemEval.
"""

from typing import List

from benchd_harness.benchmarks.base import BaseBenchmark, BenchmarkItem


ITEMS = [
    # --- Document Retrieval: store a page, find it by query ---
    {
        "id": "doc_001",
        "dimension": "document_retrieval",
        "turns": [
            {"role": "user", "content": "Project Alpha uses React for the frontend, Express for the backend, and PostgreSQL for the database. The team lead is Sarah Chen. Deployment is on AWS us-east-1."},
        ],
        "query": "What database does Project Alpha use?",
        "expected": "postgresql",
    },
    {
        "id": "doc_002",
        "dimension": "document_retrieval",
        "turns": [
            {"role": "user", "content": "Meeting notes from March 15: We decided to migrate from MongoDB to PostgreSQL. Timeline is 6 weeks. Budget approved for $45,000. Jake will lead the migration."},
        ],
        "query": "Who is leading the database migration?",
        "expected": "jake",
    },
    {
        "id": "doc_003",
        "dimension": "document_retrieval",
        "turns": [
            {"role": "user", "content": "Customer feedback summary: Users love the dark mode feature. Main complaints are about slow search performance and occasional crashes on mobile. NPS score is 42."},
        ],
        "query": "What is the NPS score?",
        "expected": "42",
    },
    {
        "id": "doc_004",
        "dimension": "document_retrieval",
        "turns": [
            {"role": "user", "content": "API documentation: The /users endpoint accepts GET and POST methods. Authentication is via Bearer token. Rate limit is 100 requests per minute. Response format is JSON."},
        ],
        "query": "What is the rate limit for the users API?",
        "expected": "100",
    },
    {
        "id": "doc_005",
        "dimension": "document_retrieval",
        "turns": [
            {"role": "user", "content": "Team roster: Alice (frontend, started Jan 2025), Bob (backend, started Mar 2024), Carol (DevOps, started Sep 2024), Dave (ML engineer, started Jun 2025)."},
        ],
        "query": "When did Bob start?",
        "expected": "mar 2024",
    },

    # --- Semantic Search Precision: find relevant content among noise ---
    {
        "id": "sem_001",
        "dimension": "semantic_search",
        "turns": [
            {"role": "user", "content": "Recipe for chocolate cake: 2 cups flour, 1 cup sugar, 3/4 cup cocoa powder, 2 eggs, 1 cup milk, 1/2 cup oil. Bake at 350F for 30 minutes."},
            {"role": "user", "content": "Recipe for banana bread: 3 ripe bananas, 1/3 cup melted butter, 3/4 cup sugar, 1 egg, 1 teaspoon vanilla, 1 teaspoon baking soda, 1 cup flour. Bake at 350F for 60 minutes."},
            {"role": "user", "content": "Recipe for pasta carbonara: 1 pound spaghetti, 4 eggs, 1 cup parmesan, 8 oz pancetta, black pepper. Cook pasta, mix eggs and cheese, combine with hot pasta and pancetta."},
        ],
        "query": "How long do you bake banana bread?",
        "expected": "60 minutes",
    },
    {
        "id": "sem_002",
        "dimension": "semantic_search",
        "turns": [
            {"role": "user", "content": "Server config for production: 8 vCPUs, 32GB RAM, 500GB SSD, Ubuntu 22.04, nginx reverse proxy, pm2 process manager."},
            {"role": "user", "content": "Server config for staging: 4 vCPUs, 16GB RAM, 200GB SSD, Ubuntu 22.04, nginx reverse proxy, pm2 process manager."},
            {"role": "user", "content": "Server config for development: 2 vCPUs, 8GB RAM, 100GB SSD, Ubuntu 22.04, built-in dev server."},
        ],
        "query": "How much RAM does the staging server have?",
        "expected": "16gb",
    },
    {
        "id": "sem_003",
        "dimension": "semantic_search",
        "turns": [
            {"role": "user", "content": "Q1 revenue: $2.3M. Q2 revenue: $2.8M. Q3 revenue: $3.1M. Q4 revenue: $3.7M. Total annual revenue: $11.9M."},
        ],
        "query": "What was Q3 revenue?",
        "expected": "3.1",
    },
    {
        "id": "sem_004",
        "dimension": "semantic_search",
        "turns": [
            {"role": "user", "content": "Python is great for data science and ML. JavaScript dominates web development. Rust excels at systems programming with memory safety. Go is preferred for cloud-native microservices."},
        ],
        "query": "Which language is best for systems programming?",
        "expected": "rust",
    },
    {
        "id": "sem_005",
        "dimension": "semantic_search",
        "turns": [
            {"role": "user", "content": "The company has offices in San Francisco (HQ, 200 employees), New York (sales, 50 employees), London (EMEA, 30 employees), and Tokyo (APAC, 15 employees)."},
        ],
        "query": "How many employees are in the London office?",
        "expected": "30",
    },

    # --- Knowledge Update: store a fact, update it, get the latest ---
    {
        "id": "upd_001",
        "dimension": "knowledge_update",
        "turns": [
            {"role": "user", "content": "Current tech stack: React 17, Node 14, PostgreSQL 13."},
            {"role": "user", "content": "Tech stack updated: React 18, Node 20, PostgreSQL 16. Migration completed successfully."},
        ],
        "query": "What version of Node are we using?",
        "expected": "20",
    },
    {
        "id": "upd_002",
        "dimension": "knowledge_update",
        "turns": [
            {"role": "user", "content": "CEO is John Smith. Founded in 2020."},
            {"role": "user", "content": "Leadership change: Jane Doe has been appointed as the new CEO, replacing John Smith effective immediately."},
        ],
        "query": "Who is the current CEO?",
        "expected": "jane doe",
    },
    {
        "id": "upd_003",
        "dimension": "knowledge_update",
        "turns": [
            {"role": "user", "content": "Office address: 123 Main Street, Suite 400, San Francisco, CA 94105."},
            {"role": "user", "content": "We moved offices. New address: 456 Market Street, Floor 12, San Francisco, CA 94107."},
        ],
        "query": "What is the current office address?",
        "expected": "456 market",
    },
    {
        "id": "upd_004",
        "dimension": "knowledge_update",
        "turns": [
            {"role": "user", "content": "Pricing: Starter plan is $29/month, Pro plan is $99/month, Enterprise is custom."},
            {"role": "user", "content": "Pricing update effective today: Starter is now $19/month, Pro is $79/month, Enterprise remains custom."},
        ],
        "query": "How much is the Starter plan?",
        "expected": "19",
    },
    {
        "id": "upd_005",
        "dimension": "knowledge_update",
        "turns": [
            {"role": "user", "content": "The project deadline is December 15, 2026."},
            {"role": "user", "content": "Deadline extended. New deadline is January 31, 2027 due to scope changes."},
        ],
        "query": "What is the project deadline?",
        "expected": "january 31",
    },

    # --- Multi-page reasoning: find info across multiple stored pages ---
    {
        "id": "multi_001",
        "dimension": "multi_page",
        "turns": [
            {"role": "user", "content": "Team A works on the payments service. Team lead: Maria. 5 engineers."},
            {"role": "user", "content": "Team B works on the search service. Team lead: Tom. 3 engineers."},
            {"role": "user", "content": "Team C works on the auth service. Team lead: Lisa. 4 engineers."},
        ],
        "query": "Who leads the search service?",
        "expected": "tom",
    },
    {
        "id": "multi_002",
        "dimension": "multi_page",
        "turns": [
            {"role": "user", "content": "Bug report #1234: Login fails on Safari. Priority: High. Assigned to: Alex."},
            {"role": "user", "content": "Bug report #1235: Dashboard charts not loading. Priority: Medium. Assigned to: Kim."},
            {"role": "user", "content": "Bug report #1236: Email notifications delayed. Priority: Low. Assigned to: Sam."},
        ],
        "query": "What is bug #1235 about?",
        "expected": "dashboard chart",
    },
    {
        "id": "multi_003",
        "dimension": "multi_page",
        "turns": [
            {"role": "user", "content": "January metrics: 10,000 users, 95% uptime, $50K MRR."},
            {"role": "user", "content": "February metrics: 12,000 users, 99% uptime, $58K MRR."},
            {"role": "user", "content": "March metrics: 15,000 users, 99.5% uptime, $67K MRR."},
        ],
        "query": "What was the uptime in February?",
        "expected": "99%",
    },
    {
        "id": "multi_004",
        "dimension": "multi_page",
        "turns": [
            {"role": "user", "content": "Vendor A: AWS. Cost: $12K/month. Contract ends Dec 2026."},
            {"role": "user", "content": "Vendor B: Stripe. Cost: 2.9% + $0.30 per transaction. No contract end date."},
            {"role": "user", "content": "Vendor C: Datadog. Cost: $3K/month. Contract ends Mar 2027."},
        ],
        "query": "How much does Datadog cost per month?",
        "expected": "3k",
    },
    {
        "id": "multi_005",
        "dimension": "multi_page",
        "turns": [
            {"role": "user", "content": "Interview notes - Candidate A (Sarah): Strong React skills, 5 years experience, asking $150K. Recommendation: Hire."},
            {"role": "user", "content": "Interview notes - Candidate B (Mike): Good backend skills, 3 years experience, asking $120K. Recommendation: Second interview."},
            {"role": "user", "content": "Interview notes - Candidate C (Emma): Excellent full-stack, 7 years experience, asking $180K. Recommendation: Strong hire."},
        ],
        "query": "What salary is Candidate C asking for?",
        "expected": "180k",
    },
]


class KnowledgeRetrievalBenchmark(BaseBenchmark):
    """
    Knowledge retrieval benchmark for knowledge brain systems.

    Tests document storage, semantic search, knowledge updates,
    and multi-page reasoning.
    """

    @property
    def slug(self) -> str:
        return "knowledge-retrieval-v1"

    @property
    def name(self) -> str:
        return "Knowledge Retrieval"

    @property
    def version(self) -> str:
        return "1.0"

    @property
    def description(self) -> str:
        return (
            "Tests knowledge brain systems on document storage, semantic search, "
            "knowledge updates, and multi-page reasoning. Designed for systems "
            "like gbrain, Graphiti, and Microsoft GraphRAG."
        )

    def __init__(self, max_items: int | None = None):
        self._max_items = max_items

    def load_items(self) -> List[BenchmarkItem]:
        items = ITEMS
        if self._max_items:
            items = items[: self._max_items]

        return [
            BenchmarkItem(
                question_id=item["id"],
                dimension=item["dimension"],
                ingest_history=item["turns"],
                recall_query=item["query"],
                expected_answer=item["expected"],
                scoring_method="exact",
                max_score=1.0,
            )
            for item in items
        ]

    def get_dimensions(self) -> List[str]:
        return ["document_retrieval", "semantic_search", "knowledge_update", "multi_page"]
