"""
CLI Command Runtime — for systems called via subprocess per operation.

This is the least efficient runtime type. Each operation spawns a new process.
Only use when no other option exists. Most systems should use python_library,
mcp_stdio, or http_server instead.
"""

import subprocess
import shlex
from typing import Optional

from .base import RuntimeExecutor, RuntimeConfig, RuntimeResult, IsolationProbe


class CLICommandRuntime(RuntimeExecutor):
    """Runtime for CLI-based systems (subprocess per call)."""

    def prepare(self) -> RuntimeResult:
        if not self.config.command:
            return RuntimeResult(success=False, message="No command configured")
        return RuntimeResult(success=True)

    def start(self) -> RuntimeResult:
        """No persistent process for CLI runtime."""
        return RuntimeResult(success=True, message="CLI runtime — no persistent process")

    def healthcheck(self) -> RuntimeResult:
        """Try running the command with --help or --version."""
        try:
            cmd = shlex.split(self.config.command)
            result = subprocess.run(
                cmd + ["--version"],
                capture_output=True, text=True, timeout=30,
                cwd=self.config.cwd or None,
            )
            return RuntimeResult(success=True, message=f"CLI available: {result.stdout[:100]}")
        except Exception as e:
            return RuntimeResult(success=False, message=f"CLI not available: {e}")

    def run_command(self, *args: str, stdin: Optional[str] = None) -> str:
        """Run a CLI command and return stdout."""
        cmd = shlex.split(self.config.command) + list(args)
        result = subprocess.run(
            cmd,
            capture_output=True, text=True,
            input=stdin,
            timeout=self.config.query_timeout_seconds,
            cwd=self.config.cwd or None,
        )
        return result.stdout.strip()

    def stop(self) -> RuntimeResult:
        return RuntimeResult(success=True)

    def check_isolation(self) -> IsolationProbe:
        return IsolationProbe(
            clean=True,
            evidence="CLI runtime — isolation depends on adapter.reset()",
        )
