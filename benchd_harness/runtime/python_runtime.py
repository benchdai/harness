"""
Python Library Runtime — the default for most adapters.

The system is a Python package imported directly. No subprocess needed.
This is how Mem0, LlamaIndex, LangChain, CrewAI etc. run.
"""

import time
from .base import RuntimeExecutor, RuntimeConfig, RuntimeResult, IsolationProbe


class PythonLibraryRuntime(RuntimeExecutor):
    """Runtime for systems that are Python libraries imported directly."""

    def prepare(self) -> RuntimeResult:
        """Python libraries are already installed via pip."""
        return RuntimeResult(success=True, message="Python library — no preparation needed")

    def start(self) -> RuntimeResult:
        """No process to start — adapter handles initialization in setup()."""
        return RuntimeResult(success=True, startup_ms=0, message="In-process — no startup needed")

    def healthcheck(self) -> RuntimeResult:
        """Adapter's setup() serves as healthcheck."""
        return RuntimeResult(success=True)

    def stop(self) -> RuntimeResult:
        """Adapter's teardown() handles cleanup."""
        return RuntimeResult(success=True)

    def check_isolation(self) -> IsolationProbe:
        """
        For Python libraries, isolation depends on the adapter's reset().
        We trust it but flag that it's adapter-managed, not environment-guaranteed.
        """
        return IsolationProbe(
            clean=True,
            evidence="Isolation managed by adapter.reset() — not environment-guaranteed",
        )
