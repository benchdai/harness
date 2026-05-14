"""
gbrain adapter for Bench'd harness.

gbrain is a personal knowledge brain with PGLite/Postgres + pgvector,
hybrid search, and knowledge graph. Uses the CLI for ingest (put) and
recall (query).

Requires:
  gbrain installed and initialized:
    git clone https://github.com/garrytan/gbrain /tmp/gbrain
    cd /tmp/gbrain && bun install && bun link
    gbrain init --pglite

  export GBRAIN_PATH=/tmp/gbrain  (path to gbrain repo)
"""

import os
import subprocess
import uuid
from typing import Any, Dict, List, Optional

from benchd_harness.adapters.base import BaseAdapter


class GBrainAdapter(BaseAdapter):
    """
    Adapter for gbrain — personal knowledge brain by Garry Tan.

    Uses the gbrain CLI to store conversation turns as pages and
    query them via hybrid search.
    """

    @property
    def name(self) -> str:
        return "gbrain"

    @property
    def version(self) -> Optional[str]:
        try:
            result = subprocess.run(
                ["bun", "run", "src/cli.ts", "--version"],
                cwd=self._gbrain_path,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout.strip() or "unknown"
        except Exception:
            return "unknown"

    def __init__(self):
        self._gbrain_path: Optional[str] = None
        self._page_slugs: list[str] = []

    def _run_gbrain(self, *args: str, stdin: str | None = None, timeout: int = 30) -> str:
        """Run a gbrain CLI command and return stdout."""
        cmd = ["bun", "run", "src/cli.ts", *args]
        result = subprocess.run(
            cmd,
            cwd=self._gbrain_path,
            capture_output=True,
            text=True,
            input=stdin,
            timeout=timeout,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if stderr and "error" in stderr.lower():
                raise RuntimeError(f"gbrain error: {stderr}")
        return result.stdout.strip()

    def setup(self) -> None:
        self._gbrain_path = os.environ.get("GBRAIN_PATH", "/tmp/gbrain")

        if not os.path.exists(os.path.join(self._gbrain_path, "src", "cli.ts")):
            raise RuntimeError(
                f"gbrain not found at {self._gbrain_path}.\n"
                "Install with:\n"
                "  git clone https://github.com/garrytan/gbrain /tmp/gbrain\n"
                "  cd /tmp/gbrain && bun install && bun link\n"
                "  gbrain init --pglite\n"
                "Or set GBRAIN_PATH to your gbrain directory."
            )

        self._page_slugs = []

    def reset(self) -> None:
        """Delete all pages created during this benchmark question."""
        for slug in self._page_slugs:
            try:
                self._run_gbrain("delete", slug, timeout=10)
            except Exception:
                pass
        self._page_slugs = []

    def teardown(self) -> None:
        self.reset()

    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        if self._gbrain_path is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        # Build conversation as markdown
        lines = []
        for turn in turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            timestamp = turn.get("timestamp", "")
            prefix = f"[{timestamp}] " if timestamp else ""
            lines.append(f"{prefix}**{role}**: {content}")

        text = "\n\n".join(lines)
        slug = f"benchd-{uuid.uuid4().hex[:8]}"

        try:
            self._run_gbrain("put", slug, stdin=text, timeout=30)
            self._page_slugs.append(slug)
        except Exception as e:
            raise RuntimeError(f"gbrain ingest failed: {e}")

    def recall(self, query: str) -> str:
        if self._gbrain_path is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        try:
            result = self._run_gbrain("query", query, timeout=30)
            # gbrain query returns lines like:
            # [0.3964] slug -- content
            # Extract the content parts
            lines = []
            for line in result.split("\n"):
                if " -- " in line:
                    content = line.split(" -- ", 1)[1]
                    lines.append(content)
                elif line.strip() and not line.startswith("["):
                    lines.append(line)
            return "\n".join(lines) if lines else result
        except Exception as e:
            return f"[recall error: {e}]"
