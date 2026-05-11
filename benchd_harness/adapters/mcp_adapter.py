"""
Generic MCP adapter for Bench'd harness.

Connects to ANY memory system that exposes an MCP server.
Auto-discovers available tools and maps them to ingest/recall/reset.

Usage:
  benchd run --adapter mcp --benchmark longmemeval-v1 \
    --adapter-config '{"endpoint": "http://localhost:3000/mcp", "api_key": "..."}'

Environment variables:
  MCP_ENDPOINT  — MCP server URL (required)
  MCP_API_KEY   — API key for authentication (optional)
  MCP_INGEST_TOOL — Override: name of the ingest tool (auto-detected if not set)
  MCP_QUERY_TOOL  — Override: name of the query tool (auto-detected if not set)
  MCP_RESET_TOOL  — Override: name of the reset tool (auto-detected if not set)
"""

import os
import json
from typing import Any, Dict, List, Optional

import requests

from benchd_harness.adapters.base import BaseAdapter


# Tool name patterns for auto-discovery
INGEST_PATTERNS = [
    "ingest", "add", "store", "remember", "save", "put", "write",
    "memory_ingest", "add_memory",
]
QUERY_PATTERNS = [
    "query", "search", "recall", "retrieve", "get", "find",
    "memory_query", "search_memory",
]
RESET_PATTERNS = [
    "reset", "clear", "delete", "forget", "remove",
    "memory_delete", "delete_all", "clear_memory",
]


class MCPAdapter(BaseAdapter):
    """
    Generic adapter that connects to any MCP-compatible memory server.

    Auto-discovers tools and maps them to Bench'd's ingest/recall/reset interface.
    """

    def __init__(self, adapter_config: Optional[Dict[str, Any]] = None):
        self._config = adapter_config or {}
        self._endpoint: Optional[str] = None
        self._api_key: Optional[str] = None
        self._session: Optional[requests.Session] = None
        self._tools: Dict[str, Any] = {}  # discovered tools
        self._ingest_tool: Optional[str] = None  # resolved tool name for ingestion
        self._query_tool: Optional[str] = None   # resolved tool name for queries
        self._reset_tool: Optional[str] = None   # resolved tool name for reset
        self._request_id = 0

    @property
    def name(self) -> str:
        return "mcp"

    @property
    def version(self) -> Optional[str]:
        return "generic"

    def setup(self) -> None:
        """Connect to MCP server and discover available tools."""
        # adapter_config takes priority, then environment variables
        self._endpoint = (
            self._config.get("endpoint")
            or os.environ.get("MCP_ENDPOINT")
        )
        if not self._endpoint:
            raise RuntimeError(
                "MCP adapter requires an endpoint.\n"
                "Set MCP_ENDPOINT or pass --adapter-config '{\"endpoint\": \"...\"}'"
            )

        self._api_key = (
            self._config.get("api_key")
            or os.environ.get("MCP_API_KEY")
        )

        self._session = requests.Session()
        if self._api_key:
            self._session.headers["Authorization"] = f"Bearer {self._api_key}"
        self._session.headers["Content-Type"] = "application/json"

        # Discover tools
        self._discover_tools()

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _discover_tools(self) -> None:
        """List available MCP tools and map them to our operations."""
        try:
            # MCP uses JSON-RPC 2.0
            resp = self._call_mcp("tools/list", {})
            tools = resp.get("tools", [])

            self._tools = {t["name"]: t for t in tools}
            tool_names = list(self._tools.keys())

            # Auto-detect or use overrides from config / env
            self._ingest_tool = (
                self._config.get("ingest_tool")
                or os.environ.get("MCP_INGEST_TOOL")
                or self._find_tool(tool_names, INGEST_PATTERNS)
            )
            self._query_tool = (
                self._config.get("query_tool")
                or os.environ.get("MCP_QUERY_TOOL")
                or self._find_tool(tool_names, QUERY_PATTERNS)
            )
            self._reset_tool = (
                self._config.get("reset_tool")
                or os.environ.get("MCP_RESET_TOOL")
                or self._find_tool(tool_names, RESET_PATTERNS)
            )

            if not self._ingest_tool:
                raise RuntimeError(
                    f"No ingest tool found among: {tool_names}. "
                    "Set MCP_INGEST_TOOL or pass ingest_tool in adapter config."
                )
            if not self._query_tool:
                raise RuntimeError(
                    f"No query tool found among: {tool_names}. "
                    "Set MCP_QUERY_TOOL or pass query_tool in adapter config."
                )
            # reset is optional — some systems don't support it

        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Cannot connect to MCP server at {self._endpoint}")

    @staticmethod
    def _find_tool(tool_names: List[str], patterns: List[str]) -> Optional[str]:
        """Find the first tool name matching any pattern."""
        for name in tool_names:
            name_lower = name.lower()
            for pattern in patterns:
                if pattern in name_lower:
                    return name
        return None

    def _call_mcp(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send a JSON-RPC 2.0 request to the MCP server."""
        assert self._session is not None
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
            "params": params,
        }
        resp = self._session.post(
            self._endpoint,  # type: ignore[arg-type]
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            err = data["error"]
            code = err.get("code", "?")
            msg = err.get("message", str(err))
            raise RuntimeError(f"MCP error {code}: {msg}")
        return data.get("result", {})

    def _call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Call an MCP tool and return the text result."""
        result = self._call_mcp("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })
        # MCP tool results can be in various formats
        content = result.get("content", [])
        if isinstance(content, list):
            texts = [
                c.get("text", str(c))
                for c in content
                if isinstance(c, dict)
            ]
            return " ".join(texts) if texts else str(content)
        return str(content)

    def teardown(self) -> None:
        """Close the HTTP session."""
        if self._session:
            self._session.close()

    def reset(self) -> None:
        """Clear memory if a reset tool is available."""
        if self._reset_tool:
            try:
                self._call_tool(self._reset_tool, {})
            except Exception:
                pass  # Reset is best-effort

    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        """Send conversation turns to the MCP memory server."""
        # Format turns as a conversation transcript
        transcript = "\n".join(
            f"[{t.get('role', 'user')}]: {t.get('content', '')}"
            for t in turns
        )

        # Try to match the tool's expected parameters
        tool_schema = self._tools.get(self._ingest_tool, {})  # type: ignore[arg-type]
        input_schema = tool_schema.get("inputSchema", {}).get("properties", {})

        # Build arguments based on the tool's declared input schema
        args = self._build_args(input_schema, transcript, turns)
        if not args:
            args["content"] = transcript

        # Add source_type if supported
        if "source_type" in input_schema:
            args["source_type"] = "conversation"

        self._call_tool(self._ingest_tool, args)  # type: ignore[arg-type]

    def recall(self, query: str) -> str:
        """Query the MCP memory server."""
        tool_schema = self._tools.get(self._query_tool, {})  # type: ignore[arg-type]
        input_schema = tool_schema.get("inputSchema", {}).get("properties", {})

        args: Dict[str, Any] = {}
        # Try common query parameter names
        for candidate in ("query", "query_text", "text", "q", "search", "input"):
            if candidate in input_schema:
                args[candidate] = query
                break

        if not args:
            # Fallback: use first string parameter
            for param_name, param_def in input_schema.items():
                if isinstance(param_def, dict) and param_def.get("type") == "string":
                    args[param_name] = query
                    break
            if not args:
                args["query"] = query

        try:
            return self._call_tool(self._query_tool, args)  # type: ignore[arg-type]
        except Exception as e:
            return f"[MCP query error: {e}]"

    @staticmethod
    def _build_args(
        input_schema: Dict[str, Any],
        transcript: str,
        turns: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build tool arguments by matching the input schema to our data."""
        args: Dict[str, Any] = {}

        if "content" in input_schema:
            args["content"] = transcript
        elif "messages" in input_schema:
            args["messages"] = [
                {"role": t.get("role", "user"), "content": t.get("content", "")}
                for t in turns
            ]
        elif "text" in input_schema:
            args["text"] = transcript
        elif "data" in input_schema:
            args["data"] = transcript
        elif "input" in input_schema:
            args["input"] = transcript
        else:
            # Fallback: use first string parameter
            for param_name, param_def in input_schema.items():
                if isinstance(param_def, dict) and param_def.get("type") == "string":
                    args[param_name] = transcript
                    break

        return args
