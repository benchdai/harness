"""
Bench'd Adapter SDK — scaffolding and validation for custom adapters.

Usage:
  benchd adapter init my-system
  benchd adapter test my-system
  benchd adapter validate my-system
"""

import os
from pathlib import Path


ADAPTER_TEMPLATE = '''"""
{name} adapter for Bench'd harness.

Requires:
  pip install {package}

  export {env_key}=...
"""

import os
from typing import Any, Dict, List, Optional

from benchd_harness.adapters.base import BaseAdapter


class {class_name}(BaseAdapter):
    """Adapter for {name}."""

    @property
    def name(self) -> str:
        return "{slug}"

    @property
    def version(self) -> Optional[str]:
        return "0.1.0"

    def __init__(self):
        self._client = None

    def setup(self) -> None:
        """Initialize your memory system client."""
        # TODO: Import and configure your system
        # Example:
        # from my_system import Client
        # self._client = Client(api_key=os.environ.get("{env_key}"))
        pass

    def reset(self) -> None:
        """Clear memory between benchmark questions."""
        # TODO: Clear all stored memories
        pass

    def teardown(self) -> None:
        """Clean up resources."""
        self._client = None

    def ingest(self, turns: List[Dict[str, Any]]) -> None:
        """
        Feed conversation turns into your memory system.

        Each turn has: role ("user"/"assistant"/"system"), content (str),
        timestamp (optional ISO string), metadata (optional dict).
        """
        if self._client is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        for turn in turns:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            # TODO: Store this turn in your memory system
            # Example:
            # self._client.add_memory(content=f"[{{role}}]: {{content}}")

    def recall(self, query: str) -> str:
        """
        Query your memory system and return a plain string response.
        """
        if self._client is None:
            raise RuntimeError("Adapter not initialized. Call setup() first.")

        try:
            # TODO: Search your memory system
            # Example:
            # results = self._client.search(query)
            # return results.text
            return ""
        except Exception as e:
            return f"[recall error: {{e}}]"
'''

SYSTEM_MANIFEST_TEMPLATE = '''name: {slug}
adapter: {slug}
install: pip install {package}
env:
  {env_key}: required
docker: null
benchmarks:
  - longmemeval-v1
  - reliability-v1
'''

REGISTER_SNIPPET = '''
# Add to benchd_harness/adapters/__init__.py:
try:
    from .{filename} import {class_name}
    _BUILTIN_ADAPTERS["{slug}"] = {class_name}
    __all__ = [*__all__, "{class_name}"]
except ImportError:
    pass
'''


def init_adapter(name: str, output_dir: Path | None = None) -> dict:
    """
    Scaffold a new adapter with template files.

    Returns dict with paths of created files.
    """
    slug = name.lower().replace(" ", "-").replace("_", "-")
    class_name = "".join(word.capitalize() for word in slug.split("-")) + "Adapter"
    filename = slug.replace("-", "_") + "_adapter"
    package = slug
    env_key = slug.upper().replace("-", "_") + "_API_KEY"

    base_dir = output_dir or Path(".")

    # Write adapter file
    adapter_path = base_dir / "benchd_harness" / "adapters" / f"{filename}.py"
    if not adapter_path.parent.exists():
        adapter_path = base_dir / f"{filename}.py"

    adapter_code = ADAPTER_TEMPLATE.format(
        name=name,
        slug=slug,
        class_name=class_name,
        package=package,
        env_key=env_key,
    )
    adapter_path.write_text(adapter_code)

    # Write system manifest
    manifest_path = base_dir / "systems" / f"{slug}.yaml"
    if not manifest_path.parent.exists():
        manifest_path = base_dir / f"{slug}.yaml"

    manifest_code = SYSTEM_MANIFEST_TEMPLATE.format(
        slug=slug,
        package=package,
        env_key=env_key,
    )
    manifest_path.write_text(manifest_code)

    # Generate register snippet
    register = REGISTER_SNIPPET.format(
        filename=filename,
        class_name=class_name,
        slug=slug,
    )

    return {
        "adapter_path": str(adapter_path),
        "manifest_path": str(manifest_path),
        "register_snippet": register,
        "class_name": class_name,
        "slug": slug,
    }


def validate_adapter(adapter_name: str) -> list[str]:
    """
    Validate an adapter implements the required interface.

    Returns list of issues (empty = valid).
    """
    from benchd_harness.adapters import get_adapter

    issues = []

    try:
        adapter = get_adapter(adapter_name)
    except ValueError as e:
        return [f"Adapter not found: {e}"]

    # Check required methods
    for method in ["setup", "reset", "teardown", "ingest", "recall"]:
        if not hasattr(adapter, method):
            issues.append(f"Missing required method: {method}")
        elif not callable(getattr(adapter, method)):
            issues.append(f"{method} is not callable")

    # Check properties
    if not hasattr(adapter, "name") or not adapter.name:
        issues.append("Missing or empty 'name' property")

    # Try setup
    try:
        adapter.setup()
    except Exception as e:
        issues.append(f"setup() failed: {e}")
        return issues  # Can't continue without setup

    # Try ingest
    try:
        adapter.ingest([{"role": "user", "content": "test message"}])
    except Exception as e:
        issues.append(f"ingest() failed: {e}")

    # Try recall
    try:
        result = adapter.recall("test query")
        if not isinstance(result, str):
            issues.append(f"recall() returned {type(result)}, expected str")
    except Exception as e:
        issues.append(f"recall() failed: {e}")

    # Try reset
    try:
        adapter.reset()
    except Exception as e:
        issues.append(f"reset() failed: {e}")

    # Teardown
    try:
        adapter.teardown()
    except Exception:
        pass

    return issues
