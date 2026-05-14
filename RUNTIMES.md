# Bench'd Runtime Architecture v0.1

Three separate contracts govern how a system is evaluated:

## 1. Track Protocol (WHAT is being tested)

Defines the benchmark domain and scoring dimensions.

| Track | Input | Query | Metrics |
|-------|-------|-------|---------|
| Conversational Memory | Chat turns | Natural language recall | Recall, Temporal, Reasoning, Reliability |
| Knowledge Base | Documents/pages | Semantic search | Retrieval, Citation, Update Freshness, Scale |
| Agent Workflow | Tool calls, decisions | Task context retrieval | Resume, Decision Recall, Regression Avoidance |
| Graph/RAG | Text corpus | Entity/relationship queries | Entity Resolution, Multi-hop, Attribution |
| Coding Memory | Commits, diffs, logs | Code context queries | Architecture Recall, Change Rationale, Handoff |

## 2. Adapter Contract (WHAT operations to call)

Track-specific logical interface.

### Conversational Memory Contract
```python
reset() → ingest(turns) → recall(query)
```

### Knowledge Base Contract
```python
reset() → index(documents) → retrieve(query)
```

### Agent Workflow Contract
```python
reset() → ingest_events(events) → retrieve_context(task)
```

## 3. Runtime Contract (HOW the system is launched and kept alive)

| Runtime Type | Description | Lifecycle | Example Systems |
|-------------|-------------|-----------|-----------------|
| `python_library` | Import and call directly | per_run | Mem0, LlamaIndex, LangChain |
| `mcp_stdio` | Long-lived MCP process via stdin/stdout | per_run | gbrain, OpenMemory |
| `http_server` | Long-lived HTTP server | per_run | Letta, Khoj |
| `docker_service` | Docker container with health check | per_run | Neo4j (for Graphiti), Postgres (for Khoj) |
| `cli_command` | Subprocess per operation | per_query | Simple tools (not recommended) |
| `hosted_endpoint` | Vendor's production API | hosted | Mem0 managed, Zep Cloud |

### Runtime Lifecycle

```
prepare() → start() → healthcheck() → check_isolation()
    ↓
[reset → load → query → query → query...]  (benchmark loop)
    ↓
cleanup() → stop()
```

## 4. Isolation Verification

Before any benchmark data is loaded, a canary check runs:

1. Query for data that should never exist ("benchd_canary_xyz789")
2. If the system returns non-empty results → **isolation_failed**
3. Run is flagged as "not score eligible" — not published as 0%

This catches the gbrain-class problem: stale data from previous runs
contaminating results. The check is automatic and applies to ALL systems.

### Isolation Strategies

| Strategy | Description | Used by |
|----------|-------------|---------|
| `full_database_wipe` | Delete all storage before each run | gbrain, systems with persistent local DBs |
| `fresh_workspace` | New directory/container per run | Docker-based systems |
| `namespace_scope` | Logical isolation within same DB | Multi-tenant systems |
| `adapter_reset` | Trust adapter.reset() | Python library imports |

## 5. Failure Classification

Not all 0% scores mean the same thing:

| Failure Type | Meaning | Published? |
|-------------|---------|------------|
| `runtime_start_failed` | System couldn't launch | No — infrastructure issue |
| `healthcheck_failed` | System started but not responsive | No — configuration issue |
| `isolation_failed` | Stale data detected | No — contaminated run |
| `benchmark_completed` | Ran successfully | Yes — real score |
| `benchmark_partial` | Ran but didn't finish all questions | Yes — with caveat |

## 6. Manifest Extensions

Every run manifest now records:

```json
{
  "runtime": {
    "type": "mcp_stdio",
    "lifecycle": "per_run_long_lived",
    "startup_ms": 12400,
    "isolation_strategy": "full_database_wipe",
    "isolation_check": "passed",
    "failure_type": "benchmark_completed"
  }
}
```

---

**Version:** 0.1.0
**Effective:** 2026-05-14
