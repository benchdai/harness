"""
Quivr adapter for Bench'd harness.

Uses quivr-core Python library. Creates a fresh Brain from ingested
content for each benchmark question.

Requires:
  pip install quivr-core
  export OPENAI_API_KEY=sk-...
"""

import os
import tempfile
from typing import Any, Dict, List, Optional

from benchd_harness.adapters.base import BaseAdapter


class QuivrAdapter(BaseAdapter):

    @property
    def name(self) -> str:
        return "quivr"

    @property
    def version(self) -> Optional[str]:
        try:
            import quivr_core
            return getattr(quivr_core, "__version__", "unknown")
        except ImportError:
            return None

    def __init__(self):
        self._brain = None
        self._Brain = None

    def setup(self) -> None:
        try:
            from quivr_core import Brain
            self._Brain = Brain
        except ImportError:
            raise RuntimeError("quivr-core not installed. pip install quivr-core")

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Quivr requires OPENAI_API_KEY.")

    def reset(self) -> None:
        self._brain = None

    def teardown(self) -> None:
        self._brain = None

    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        if not self._Brain:
            raise RuntimeError("Not initialized.")

        lines = []
        for turn in turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            lines.append(f"[{role}]: {content}")

        text = "\n\n".join(lines)

        # Create brain directly from the content file
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        tmp.write(text)
        tmp.close()

        try:
            self._brain = self._Brain.from_files(
                name="benchd-eval",
                file_paths=[tmp.name],
            )
        finally:
            os.unlink(tmp.name)

    def recall(self, query: str) -> str:
        if not self._brain:
            return ""

        try:
            result = self._brain.ask(query)
            if hasattr(result, "answer"):
                return str(result.answer)
            return str(result)
        except Exception as e:
            return f"[recall error: {e}]"
