"""
gbrain adapter for Bench'd harness.

Uses gbrain's MCP stdio transport — starts one persistent process,
sends JSON-RPC messages for ingest and recall. This is how people
actually use gbrain (via Claude Code, Cursor, etc.).

This same approach applies to ANY MCP stdio-based memory system.

Requires:
  gbrain installed:
    git clone https://github.com/garrytan/gbrain /tmp/gbrain
    cd /tmp/gbrain && bun install
    gbrain init --pglite

  export GBRAIN_PATH=/tmp/gbrain
"""

import json
import os
import subprocess
import uuid
from typing import Any, Dict, List, Optional

from benchd_harness.adapters.base import BaseAdapter


class GBrainAdapter(BaseAdapter):
    """
    Adapter for gbrain via persistent MCP stdio process.

    Starts gbrain serve (stdio MCP) once in setup(), sends JSON-RPC
    messages for put/query/delete. Same pattern as any MCP stdio system.
    """

    @property
    def name(self) -> str:
        return "gbrain"

    @property
    def version(self) -> Optional[str]:
        return "0.33"

    def __init__(self):
        self._gbrain_path: Optional[str] = None
        self._process: Optional[subprocess.Popen] = None
        self._request_id = 0
        self._page_slugs: list[str] = []

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _send_jsonrpc(self, method: str, params: dict) -> dict:
        """Send a JSON-RPC message to the running gbrain MCP process."""
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("gbrain process not running")

        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
            "params": params,
        }

        line = json.dumps(request) + "\n"
        self._process.stdin.write(line)
        self._process.stdin.flush()

        # Read response line
        response_line = self._process.stdout.readline()
        if not response_line:
            raise RuntimeError("gbrain process returned empty response")

        try:
            response = json.loads(response_line)
        except json.JSONDecodeError:
            # Sometimes gbrain outputs non-JSON lines (logs). Skip them.
            for _ in range(10):
                response_line = self._process.stdout.readline()
                if not response_line:
                    break
                try:
                    response = json.loads(response_line)
                    break
                except json.JSONDecodeError:
                    continue
            else:
                raise RuntimeError(f"Could not parse gbrain response")

        if "error" in response:
            raise RuntimeError(f"gbrain error: {response['error']}")

        return response.get("result", {})

    def setup(self) -> None:
        """Start persistent gbrain MCP stdio process."""
        self._gbrain_path = os.environ.get("GBRAIN_PATH", "/tmp/gbrain")

        if not os.path.exists(os.path.join(self._gbrain_path, "src", "cli.ts")):
            raise RuntimeError(
                f"gbrain not found at {self._gbrain_path}.\n"
                "Install: git clone https://github.com/garrytan/gbrain /tmp/gbrain\n"
                "         cd /tmp/gbrain && bun install && gbrain init --pglite"
            )

        # Start gbrain serve (MCP stdio mode)
        try:
            self._process = subprocess.Popen(
                ["bun", "run", "src/cli.ts", "serve"],
                cwd=self._gbrain_path,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            # Wait a moment for initialization
            import time
            time.sleep(5)

            if self._process.poll() is not None:
                stderr = self._process.stderr.read()
                raise RuntimeError(f"gbrain serve exited: {stderr[:500]}")

        except FileNotFoundError:
            raise RuntimeError("bun not found. Install: curl -fsSL https://bun.sh/install | bash")

        self._page_slugs = []

    def _fallback_cli(self, *args: str, stdin: str | None = None, timeout: int = 120) -> str:
        """Fallback: run gbrain CLI if MCP stdio doesn't work."""
        cmd = ["bun", "run", "src/cli.ts", *args]
        result = subprocess.run(
            cmd,
            cwd=self._gbrain_path,
            capture_output=True,
            text=True,
            input=stdin,
            timeout=timeout,
        )
        return result.stdout.strip()

    def reset(self) -> None:
        """Delete pages created during this question."""
        for slug in self._page_slugs:
            try:
                self._fallback_cli("delete", slug, timeout=30)
            except Exception:
                pass
        self._page_slugs = []

    def teardown(self) -> None:
        """Stop the persistent gbrain process."""
        self.reset()
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=10)
            except Exception:
                self._process.kill()
            self._process = None

    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        """Store conversation as a gbrain page."""
        if self._gbrain_path is None:
            raise RuntimeError("Adapter not initialized.")

        lines = []
        for turn in turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            lines.append(f"**{role}**: {content}")

        text = "\n\n".join(lines)
        slug = f"benchd-{uuid.uuid4().hex[:8]}"

        try:
            # Try MCP first
            if self._process and self._process.poll() is None:
                self._send_jsonrpc("tools/call", {
                    "name": "put",
                    "arguments": {"slug": slug, "content": text},
                })
            else:
                # Fallback to CLI
                self._fallback_cli("put", slug, stdin=text, timeout=120)
            self._page_slugs.append(slug)
        except Exception:
            # CLI fallback
            try:
                self._fallback_cli("put", slug, stdin=text, timeout=120)
                self._page_slugs.append(slug)
            except Exception as e:
                raise RuntimeError(f"gbrain ingest failed: {e}")

    def recall(self, query: str) -> str:
        """Query gbrain for relevant memories."""
        if self._gbrain_path is None:
            raise RuntimeError("Adapter not initialized.")

        try:
            # Try MCP first
            if self._process and self._process.poll() is None:
                result = self._send_jsonrpc("tools/call", {
                    "name": "query",
                    "arguments": {"query": query},
                })
                # Extract text from MCP response
                content = result.get("content", [])
                if isinstance(content, list):
                    texts = [c.get("text", str(c)) for c in content if isinstance(c, dict)]
                    return " ".join(texts) if texts else str(content)
                return str(content)
            else:
                # CLI fallback
                result = self._fallback_cli("query", query, timeout=120)
                lines = []
                for line in result.split("\n"):
                    if " -- " in line:
                        lines.append(line.split(" -- ", 1)[1])
                    elif line.strip() and not line.startswith("["):
                        lines.append(line)
                return "\n".join(lines) if lines else result
        except Exception as e:
            return f"[recall error: {e}]"
