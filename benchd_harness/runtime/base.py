"""
Base classes for the Bench'd Runtime Layer.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class RuntimeType(str, Enum):
    PYTHON_LIBRARY = "python_library"
    MCP_STDIO = "mcp_stdio"
    HTTP_SERVER = "http_server"
    DOCKER_SERVICE = "docker_service"
    CLI_COMMAND = "cli_command"
    HOSTED_ENDPOINT = "hosted_endpoint"


class Lifecycle(str, Enum):
    PER_RUN = "per_run_long_lived"       # Start once, live for entire benchmark
    PER_SCENARIO = "per_scenario"         # Restart per benchmark question
    PER_QUERY = "per_query"              # New process per query (least efficient)
    HOSTED = "hosted_persistent"          # Always running (vendor endpoint)


class IsolationStrategy(str, Enum):
    FULL_WIPE = "full_database_wipe"      # Delete all data
    FRESH_WORKSPACE = "fresh_workspace"    # New directory/container
    NAMESPACE = "namespace_scope"          # Logical isolation within same DB
    ADAPTER_RESET = "adapter_reset"        # Trust adapter.reset()
    NONE = "none"                          # No isolation (for hosted endpoints)


class FailureType(str, Enum):
    RUNTIME_START_FAILED = "runtime_start_failed"
    HEALTHCHECK_FAILED = "healthcheck_failed"
    ISOLATION_FAILED = "isolation_failed"
    CAPABILITY_UNSUPPORTED = "capability_unsupported"
    BENCHMARK_COMPLETED = "benchmark_completed"
    BENCHMARK_PARTIAL = "benchmark_partial"


@dataclass
class RuntimeConfig:
    """Configuration for a runtime executor, typically from benchd.yaml."""
    type: str                                    # Runtime type
    command: str = ""                            # Command to start the system
    lifecycle: str = "per_run_long_lived"         # How long the process lives
    startup_timeout_seconds: int = 120            # Max wait for system to start
    query_timeout_seconds: int = 60               # Max wait per query
    isolation_strategy: str = "adapter_reset"     # How to guarantee clean state
    wipe_paths: list[str] = field(default_factory=list)  # Paths to delete for full_wipe
    healthcheck_url: str = ""                     # URL to check for HTTP/Docker
    healthcheck_timeout: int = 30                 # Max wait for health
    docker_image: str = ""                        # Docker image name
    docker_ports: list[str] = field(default_factory=list)
    docker_env: dict[str, str] = field(default_factory=dict)
    cwd: str = ""                                # Working directory
    env: dict[str, str] = field(default_factory=dict)  # Extra env vars


@dataclass
class RuntimeResult:
    """Result of a runtime lifecycle operation."""
    success: bool
    failure_type: Optional[FailureType] = None
    startup_ms: float = 0
    message: str = ""
    logs: str = ""


@dataclass
class IsolationProbe:
    """Result of the canary isolation check."""
    clean: bool
    evidence: str = ""
    stale_data_found: bool = False


class RuntimeExecutor(ABC):
    """
    Abstract base for all runtime types.

    Every runtime implements the full lifecycle:
    prepare → start → healthcheck → [operations] → cleanup → stop

    The adapter sits ON TOP of the runtime — it translates
    Bench'd operations (ingest/recall) into system-specific calls
    that go through the runtime.
    """

    def __init__(self, config: RuntimeConfig):
        self.config = config

    @abstractmethod
    def prepare(self) -> RuntimeResult:
        """Install or validate dependencies."""
        ...

    @abstractmethod
    def start(self) -> RuntimeResult:
        """Launch the target system."""
        ...

    @abstractmethod
    def healthcheck(self) -> RuntimeResult:
        """Verify the system is ready to accept operations."""
        ...

    @abstractmethod
    def stop(self) -> RuntimeResult:
        """Stop the target system and release resources."""
        ...

    def cleanup(self) -> RuntimeResult:
        """Remove run artifacts. Default: no-op."""
        return RuntimeResult(success=True)

    def check_isolation(self) -> IsolationProbe:
        """
        Canary check: verify the system has no stale data.

        Run BEFORE any benchmark data is loaded. Query for something
        that should never exist. If it returns results, isolation failed.
        """
        return IsolationProbe(clean=True, evidence="No canary check implemented")

    def wipe(self) -> RuntimeResult:
        """
        Full state wipe for isolation strategies that need it.
        Default implementation deletes configured wipe_paths.
        """
        import shutil
        for path in self.config.wipe_paths:
            try:
                shutil.rmtree(path, ignore_errors=True)
            except Exception:
                pass
        return RuntimeResult(success=True)
