"""
Bench'd Runtime Layer — manages how target systems are launched, kept alive, and torn down.

Runtime types:
  - python_library: Import and call directly (most adapters)
  - mcp_stdio: Long-lived MCP process via stdin/stdout
  - http_server: Long-lived HTTP server (Letta, Khoj)
  - docker_service: Docker container with health check
  - cli_command: CLI subprocess per operation (simple tools)
  - hosted_endpoint: Vendor's production API

Each runtime type implements the RuntimeExecutor interface:
  prepare() → start() → healthcheck() → [reset/load/query...] → cleanup() → stop()
"""

from .base import RuntimeExecutor, RuntimeConfig, RuntimeResult
from .python_runtime import PythonLibraryRuntime
from .mcp_stdio_runtime import MCPStdioRuntime
from .http_runtime import HTTPServerRuntime
from .docker_runtime import DockerServiceRuntime
from .cli_runtime import CLICommandRuntime

__all__ = [
    "RuntimeExecutor",
    "RuntimeConfig",
    "RuntimeResult",
    "PythonLibraryRuntime",
    "MCPStdioRuntime",
    "HTTPServerRuntime",
    "DockerServiceRuntime",
    "CLICommandRuntime",
    "get_runtime",
]

RUNTIME_TYPES = {
    "python_library": PythonLibraryRuntime,
    "mcp_stdio": MCPStdioRuntime,
    "http_server": HTTPServerRuntime,
    "docker_service": DockerServiceRuntime,
    "cli_command": CLICommandRuntime,
}


def get_runtime(config: RuntimeConfig) -> RuntimeExecutor:
    """Factory: get a runtime executor by config type."""
    cls = RUNTIME_TYPES.get(config.type)
    if cls is None:
        available = ", ".join(sorted(RUNTIME_TYPES))
        raise ValueError(f"Unknown runtime type {config.type!r}. Available: {available}")
    return cls(config)
