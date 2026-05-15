"""
gbrain adapter for Bench'd harness.

Uses gbrain's MCP stdio transport — starts one persistent process in setup(),
keeps it alive for all queries. NO CLI subprocess per operation.

This is how people actually use gbrain (via Claude Code, Cursor MCP).
Same approach applies to ANY MCP stdio-based memory system.

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
import time
import uuid
from typing import Any, Dict, List, Optional

from benchd_harness.adapters.base import BaseAdapter


class GBrainAdapter(BaseAdapter):
    """
    Adapter for gbrain via persistent MCP stdio process.

    Starts gbrain serve once in setup(). Sends JSON-RPC for every
    put/query/delete. No subprocess per call. No cold starts.
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
        self._ready = False

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _send(self, method: str, params: dict) -> dict:
        """Send JSON-RPC 2.0 message to the persistent MCP process."""
        if not self._process or self._process.poll() is not None:
            raise RuntimeError("gbrain MCP process not running")

        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
            "params": params,
        }

        self._process.stdin.write(json.dumps(request) + "\n")
        self._process.stdin.flush()

        # Read response, skip non-JSON log lines
        for _ in range(50):
            line = self._process.stdout.readline()
            if not line:
                raise RuntimeError("gbrain returned empty response")
            line = line.strip()
            if not line:
                continue
            try:
                response = json.loads(line)
                if "error" in response:
                    err = response["error"]
                    msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                    raise RuntimeError(f"gbrain error: {msg}")
                return response.get("result", {})
            except json.JSONDecodeError:
                continue

        raise RuntimeError("Could not parse gbrain response after 50 lines")

    def _call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call an MCP tool and return text result."""
        result = self._send("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })
        content = result.get("content", [])
        if isinstance(content, list):
            texts = [c.get("text", str(c)) for c in content if isinstance(c, dict)]
            return " ".join(texts) if texts else str(content)
        return str(content)

    def setup(self) -> None:
        """Start persistent gbrain MCP stdio process with clean database."""
        self._gbrain_path = os.environ.get("GBRAIN_PATH", "/tmp/gbrain")

        cli_path = os.path.join(self._gbrain_path, "src", "cli.ts")
        if not os.path.exists(cli_path):
            raise RuntimeError(
                f"gbrain not found at {self._gbrain_path}.\n"
                "Install: git clone https://github.com/garrytan/gbrain /tmp/gbrain\n"
                "         cd /tmp/gbrain && bun install && bun run src/cli.ts init --pglite"
            )

        # Wipe PGLite database for clean isolation
        import shutil
        pglite_path = os.path.expanduser("~/.gbrain/brain.pglite")
        if os.path.exists(pglite_path):
            shutil.rmtree(pglite_path, ignore_errors=True)

        # Re-initialize PGLite
        subprocess.run(
            ["bun", "run", "src/cli.ts", "init", "--pglite"],
            cwd=self._gbrain_path,
            capture_output=True,
            timeout=60,
        )

        # Start gbrain serve (MCP stdio)
        self._process = subprocess.Popen(
            ["bun", "run", "src/cli.ts", "serve"],
            cwd=self._gbrain_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Wait for process to initialize
        time.sleep(8)

        if self._process.poll() is not None:
            stderr = self._process.stderr.read()[:500] if self._process.stderr else ""
            raise RuntimeError(f"gbrain serve exited: {stderr}")

        # Verify it responds
        try:
            result = self._send("tools/list", {})
            tools = result.get("tools", [])
            tool_names = [t.get("name", "") for t in tools] if isinstance(tools, list) else []
            self._ready = True
        except Exception as e:
            raise RuntimeError(f"gbrain MCP not responsive: {e}")

        self._page_slugs = []

    def reset(self) -> None:
        """Delete pages created during this benchmark question."""
        for slug in self._page_slugs:
            try:
                self._call_tool("delete_page", {"slug": slug})
            except Exception:
                pass
        self._page_slugs = []

    def teardown(self) -> None:
        """Stop the persistent MCP process."""
        self.reset()
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
        self._ready = False

    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        """Store conversation as a gbrain page via MCP."""
        if not self._ready:
            raise RuntimeError("gbrain not initialized. Call setup() first.")

        lines = []
        for turn in turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            lines.append(f"**{role}**: {content}")

        text = "\n\n".join(lines)
        slug = f"benchd-{uuid.uuid4().hex[:8]}"

        self._call_tool("put_page", {"slug": slug, "content": text})
        self._page_slugs.append(slug)

    def recall(self, query: str) -> str:
        """Query gbrain via MCP."""
        if not self._ready:
            raise RuntimeError("gbrain not initialized. Call setup() first.")

        try:
            result = self._call_tool("query", {"query": query})
            return result
        except Exception as e:
            return f"[recall error: {e}]"
