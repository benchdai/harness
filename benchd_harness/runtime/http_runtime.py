"""
HTTP Server Runtime — for systems that run as HTTP/REST servers.

Used by Letta, Khoj, and any system with a health endpoint.
"""

import time
from typing import Optional

import requests

from .base import RuntimeExecutor, RuntimeConfig, RuntimeResult, IsolationProbe


class HTTPServerRuntime(RuntimeExecutor):
    """Runtime for HTTP-based memory systems."""

    def __init__(self, config: RuntimeConfig):
        super().__init__(config)
        self._session: Optional[requests.Session] = None

    def prepare(self) -> RuntimeResult:
        if not self.config.healthcheck_url:
            return RuntimeResult(success=False, message="No healthcheck_url configured")
        self._session = requests.Session()
        return RuntimeResult(success=True)

    def start(self) -> RuntimeResult:
        """HTTP servers are expected to already be running (via Docker or manually)."""
        return RuntimeResult(success=True, message="HTTP server assumed running")

    def healthcheck(self) -> RuntimeResult:
        """Hit the health endpoint."""
        if not self._session:
            return RuntimeResult(success=False, message="Session not initialized")

        start = time.time()
        deadline = start + self.config.healthcheck_timeout

        while time.time() < deadline:
            try:
                resp = self._session.get(self.config.healthcheck_url, timeout=5)
                if resp.status_code < 500:
                    ms = (time.time() - start) * 1000
                    return RuntimeResult(success=True, startup_ms=ms, message=f"HTTP {resp.status_code}")
            except Exception:
                pass
            time.sleep(2)

        return RuntimeResult(success=False, message=f"Health check timed out after {self.config.healthcheck_timeout}s")

    def stop(self) -> RuntimeResult:
        if self._session:
            self._session.close()
        return RuntimeResult(success=True)

    def check_isolation(self) -> IsolationProbe:
        return IsolationProbe(
            clean=True,
            evidence="HTTP server — isolation managed by adapter.reset()",
        )

    @property
    def session(self) -> Optional[requests.Session]:
        return self._session
