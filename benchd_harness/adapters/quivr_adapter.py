"""
Quivr adapter for Bench'd harness.

Quivr is an open-source RAG brain that stores and retrieves documents.
Runs as a Docker service with REST API.

Requires:
  docker run -d --name quivr -p 5050:5050 quivrhq/quivr:latest
  Or: pip install quivr-core

  export OPENAI_API_KEY=sk-...
"""

import os
from typing import Any, Dict, List, Optional

from benchd_harness.adapters.base import BaseAdapter


class QuivrAdapter(BaseAdapter):
    """Adapter for Quivr using quivr-core Python library."""

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

    def setup(self) -> None:
        try:
            from quivr_core import Brain
        except ImportError:
            raise RuntimeError(
                "quivr-core not installed. Install with:\n"
                "  pip install quivr-core"
            )

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Quivr requires OPENAI_API_KEY.")

        import tempfile as _tf
        _seed = _tf.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        _seed.write("Bench'd evaluation seed document.")
        _seed.close()
        self._brain = Brain.from_files(name="benchd-eval", file_paths=[_seed.name])
        os.unlink(_seed.name)

    def reset(self) -> None:
        if self._brain:
            try:
                from quivr_core import Brain
                import tempfile as _tf
        _seed = _tf.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        _seed.write("Bench'd evaluation seed document.")
        _seed.close()
        self._brain = Brain.from_files(name="benchd-eval", file_paths=[_seed.name])
        os.unlink(_seed.name)
            except Exception:
                pass

    def teardown(self) -> None:
        self._brain = None

    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        if not self._brain:
            raise RuntimeError("Not initialized.")

        import tempfile, os
        lines = []
        for turn in turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            lines.append(f"[{role}]: {content}")

        text = "\n\n".join(lines)

        # Write to temp file and ingest
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        tmp.write(text)
        tmp.close()

        try:
            self._brain.add_file(tmp.name)
        except Exception:
            try:
                from quivr_core import Brain
                self._brain = Brain.from_files(name="benchd-eval", file_paths=[tmp.name])
            except Exception:
                pass
        finally:
            os.unlink(tmp.name)

    def recall(self, query: str) -> str:
        if not self._brain:
            raise RuntimeError("Not initialized.")

        try:
            result = self._brain.ask(query)
            if hasattr(result, "answer"):
                return str(result.answer)
            return str(result)
        except Exception as e:
            return f"[recall error: {e}]"
