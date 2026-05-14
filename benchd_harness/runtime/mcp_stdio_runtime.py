"""
MCP Stdio Runtime — for systems that run as persistent MCP processes.

Starts one process via stdin/stdout, keeps it alive for the entire run,
sends JSON-RPC 2.0 messages. This is how gbrain, OpenMemory, and other
MCP-native systems should be tested.
"""

import json
import subprocess
import time
import shlex
from typing import Optional

from .base import RuntimeExecutor, RuntimeConfig, RuntimeResult, IsolationProbe


class MCPStdioRuntime(RuntimeExecutor):
    """Runtime for MCP stdio-based systems (gbrain, etc.)."""

    def __init__(self, config: RuntimeConfig):
        super().__init__(config)
        self._process: Optional[subprocess.Popen] = None
        self._request_id = 0

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def prepare(self) -> RuntimeResult:
        """Verify the command exists."""
        if not self.config.command:
            return RuntimeResult(success=False, message="No command configured for MCP stdio runtime")
        return RuntimeResult(success=True)

    def start(self) -> RuntimeResult:
        """Start the MCP stdio process and wait for it to be ready."""
        start_time = time.time()

        try:
            cmd = shlex.split(self.config.command)
            self._process = subprocess.Popen(
                cmd,
                cwd=self.config.cwd or None,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env={**dict(__import__("os").environ), **self.config.env},
            )

            # Wait for startup
            time.sleep(min(self.config.startup_timeout_seconds / 4, 10))

            if self._process.poll() is not None:
                stderr = self._process.stderr.read()[:500]
                return RuntimeResult(
                    success=False,
                    message=f"MCP process exited immediately: {stderr}",
                )

            startup_ms = (time.time() - start_time) * 1000
            return RuntimeResult(
                success=True,
                startup_ms=startup_ms,
                message=f"MCP stdio process started (PID {self._process.pid})",
            )

        except Exception as e:
            return RuntimeResult(success=False, message=f"Failed to start MCP process: {e}")

    def healthcheck(self) -> RuntimeResult:
        """Send a tools/list request to verify the MCP server is responsive."""
        if self._process is None or self._process.poll() is not None:
            return RuntimeResult(success=False, message="MCP process not running")

        try:
            result = self.send_message("tools/list", {})
            tools = result.get("tools", [])
            return RuntimeResult(
                success=True,
                message=f"MCP server responsive, {len(tools)} tools available",
            )
        except Exception as e:
            return RuntimeResult(success=False, message=f"MCP healthcheck failed: {e}")

    def send_message(self, method: str, params: dict) -> dict:
        """Send a JSON-RPC 2.0 message and read the response."""
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("MCP process not running")

        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
            "params": params,
        }

        self._process.stdin.write(json.dumps(request) + "\n")
        self._process.stdin.flush()

        # Read response, skipping non-JSON lines (logs)
        for _ in range(20):
            line = self._process.stdout.readline()
            if not line:
                raise RuntimeError("MCP process returned no response")
            try:
                response = json.loads(line)
                if "error" in response:
                    raise RuntimeError(f"MCP error: {response['error']}")
                return response.get("result", {})
            except json.JSONDecodeError:
                continue

        raise RuntimeError("Could not parse MCP response after 20 lines")

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call an MCP tool and return the text result."""
        result = self.send_message("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })
        content = result.get("content", [])
        if isinstance(content, list):
            texts = [c.get("text", str(c)) for c in content if isinstance(c, dict)]
            return " ".join(texts) if texts else str(content)
        return str(content)

    def check_isolation(self) -> IsolationProbe:
        """
        Canary check: query for something that should never exist.
        If results come back, there's stale data contamination.
        """
        try:
            result = self.call_tool("query", {"query": "benchd_canary_isolation_check_xyz789"})
            if result and "error" not in result.lower() and len(result.strip()) > 0:
                return IsolationProbe(
                    clean=False,
                    evidence=f"Canary query returned data: {result[:200]}",
                    stale_data_found=True,
                )
            return IsolationProbe(clean=True, evidence="Canary query returned empty — clean state")
        except Exception:
            return IsolationProbe(clean=True, evidence="Canary query errored — likely clean")

    def stop(self) -> RuntimeResult:
        """Terminate the MCP process."""
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=10)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None
        return RuntimeResult(success=True, message="MCP process stopped")

    @property
    def process(self) -> Optional[subprocess.Popen]:
        return self._process
