"""Adapter layer for Bench'd harness — pluggable memory system backends."""

from .base import BaseAdapter
from .echo import EchoAdapter
from .null import NullAdapter

__all__ = ["BaseAdapter", "EchoAdapter", "NullAdapter", "get_adapter"]

_BUILTIN_ADAPTERS: dict[str, type[BaseAdapter]] = {
    "echo": EchoAdapter,
    "null": NullAdapter,
}

# Optional adapters — registered only when their dependencies are installed.
try:
    from .mem0_adapter import Mem0Adapter

    _BUILTIN_ADAPTERS["mem0-local"] = Mem0Adapter
    __all__ = [*__all__, "Mem0Adapter"]
except ImportError:
    pass

try:
    from .llm_baseline import LLMBaselineAdapter

    _BUILTIN_ADAPTERS["llm-baseline"] = LLMBaselineAdapter
    __all__ = [*__all__, "LLMBaselineAdapter"]
except ImportError:
    pass

try:
    from .letta_adapter import LettaAdapter

    _BUILTIN_ADAPTERS["letta"] = LettaAdapter
    __all__ = [*__all__, "LettaAdapter"]
except ImportError:
    pass

try:
    from .zep_adapter import ZepAdapter

    _BUILTIN_ADAPTERS["zep"] = ZepAdapter
    __all__ = [*__all__, "ZepAdapter"]
except ImportError:
    pass

try:
    from .cognee_adapter import CogneeAdapter

    _BUILTIN_ADAPTERS["cognee"] = CogneeAdapter
    __all__ = [*__all__, "CogneeAdapter"]
except ImportError:
    pass

try:
    from .graphiti_adapter import GraphitiAdapter

    _BUILTIN_ADAPTERS["graphiti"] = GraphitiAdapter
    __all__ = [*__all__, "GraphitiAdapter"]
except ImportError:
    pass

try:
    from .langchain_memory_adapter import LangChainMemoryAdapter

    _BUILTIN_ADAPTERS["langchain-memory"] = LangChainMemoryAdapter
    __all__ = [*__all__, "LangChainMemoryAdapter"]
except ImportError:
    pass

try:
    from .llamaindex_memory_adapter import LlamaIndexMemoryAdapter

    _BUILTIN_ADAPTERS["llamaindex-memory"] = LlamaIndexMemoryAdapter
    __all__ = [*__all__, "LlamaIndexMemoryAdapter"]
except ImportError:
    pass

try:
    from .verifiedstate_adapter import VerifiedStateAdapter

    _BUILTIN_ADAPTERS["verifiedstate"] = VerifiedStateAdapter
    __all__ = [*__all__, "VerifiedStateAdapter"]
except ImportError:
    pass

try:
    from .honcho_adapter import HonchoAdapter

    _BUILTIN_ADAPTERS["honcho"] = HonchoAdapter
    __all__ = [*__all__, "HonchoAdapter"]
except ImportError:
    pass

try:
    from .memobase_adapter import MemobaseAdapter

    _BUILTIN_ADAPTERS["memobase"] = MemobaseAdapter
    __all__ = [*__all__, "MemobaseAdapter"]
except ImportError:
    pass

try:
    from .crewai_adapter import CrewAIMemoryAdapter

    _BUILTIN_ADAPTERS["crewai-memory"] = CrewAIMemoryAdapter
    __all__ = [*__all__, "CrewAIMemoryAdapter"]
except ImportError:
    pass

try:
    from .langmem_adapter import LangMemAdapter

    _BUILTIN_ADAPTERS["langmem"] = LangMemAdapter
    __all__ = [*__all__, "LangMemAdapter"]
except ImportError:
    pass

try:
    from .khoj_adapter import KhojAdapter

    _BUILTIN_ADAPTERS["khoj"] = KhojAdapter
    __all__ = [*__all__, "KhojAdapter"]
except ImportError:
    pass

try:
    from .graphrag_adapter import GraphRAGAdapter

    _BUILTIN_ADAPTERS["microsoft-graphrag"] = GraphRAGAdapter
    __all__ = [*__all__, "GraphRAGAdapter"]
except ImportError:
    pass

try:
    from .autogpt_adapter import AutoGPTMemoryAdapter

    _BUILTIN_ADAPTERS["autogpt-memory"] = AutoGPTMemoryAdapter
    __all__ = [*__all__, "AutoGPTMemoryAdapter"]
except ImportError:
    pass

try:
    from .julep_adapter import JulepAdapter

    _BUILTIN_ADAPTERS["julep"] = JulepAdapter
    __all__ = [*__all__, "JulepAdapter"]
except ImportError:
    pass

try:
    from .openmemory_adapter import OpenMemoryAdapter

    _BUILTIN_ADAPTERS["openmemory"] = OpenMemoryAdapter
    __all__ = [*__all__, "OpenMemoryAdapter"]
except ImportError:
    pass

try:
    from .gbrain_adapter import GBrainAdapter

    _BUILTIN_ADAPTERS["gbrain"] = GBrainAdapter
    __all__ = [*__all__, "GBrainAdapter"]
except ImportError:
    pass

try:
    from .memoripy_adapter import MemoripyAdapter

    _BUILTIN_ADAPTERS["memoripy"] = MemoripyAdapter
    __all__ = [*__all__, "MemoripyAdapter"]
except ImportError:
    pass

from .mcp_adapter import MCPAdapter

_BUILTIN_ADAPTERS["mcp"] = MCPAdapter
__all__ = [*__all__, "MCPAdapter"]


def get_adapter(name: str, adapter_config: dict | None = None) -> BaseAdapter:
    """
    Factory function that returns an adapter instance by name.

    Args:
        name: Adapter identifier (e.g., 'echo', 'null', 'mem0-local').

    Returns:
        An instantiated BaseAdapter subclass.

    Raises:
        ValueError: If the adapter name is not recognized.
    """
    cls = _BUILTIN_ADAPTERS.get(name)
    if cls is None:
        # Provide a targeted hint when mem0 is requested but not installed.
        if name == "mem0-local":
            raise ValueError(
                "Adapter 'mem0-local' requires the mem0ai package. "
                "Install it with: pip install mem0ai"
            )
        if name == "llm-baseline":
            raise ValueError(
                "Adapter 'llm-baseline' requires the openai package. "
                "Install it with: pip install openai"
            )
        if name == "letta":
            raise ValueError(
                "Adapter 'letta' requires the letta-client package. "
                "Install it with: pip install letta-client"
            )
        if name == "zep":
            raise ValueError(
                "Adapter 'zep' requires the zep-python package. "
                "Install it with: pip install zep-python"
            )
        if name == "cognee":
            raise ValueError(
                "Adapter 'cognee' requires the cognee package. "
                "Install it with: pip install cognee"
            )
        if name == "graphiti":
            raise ValueError(
                "Adapter 'graphiti' requires the graphiti-core package and Neo4j. "
                "Install it with: pip install graphiti-core\n"
                "Neo4j quick-start: docker run -d --name neo4j -p 7687:7687 "
                "-e NEO4J_AUTH=neo4j/password neo4j:5"
            )
        if name == "langchain-memory":
            raise ValueError(
                "Adapter 'langchain-memory' requires LangChain packages. "
                "Install with: pip install langchain langchain-openai langchain-community"
            )
        if name == "llamaindex-memory":
            raise ValueError(
                "Adapter 'llamaindex-memory' requires LlamaIndex packages. "
                "Install with: pip install llama-index llama-index-llms-openai"
            )
        if name == "verifiedstate":
            raise ValueError(
                "Adapter 'verifiedstate' requires the requests package and VS_API_KEY. "
                "Install with: pip install requests\n"
                "Then set: export VS_API_KEY=vs_live_..."
            )
        if name == "honcho":
            raise ValueError(
                "Adapter 'honcho' requires the honcho-ai package. "
                "Install it with: pip install honcho-ai\n"
                "Then set: export HONCHO_API_KEY=..."
            )
        if name == "memobase":
            raise ValueError(
                "Adapter 'memobase' requires the memobase package. "
                "Install it with: pip install memobase\n"
                "Then set: export MEMOBASE_API_KEY=..."
            )
        available = ", ".join(sorted(_BUILTIN_ADAPTERS))
        raise ValueError(
            f"Unknown adapter {name!r}. Available adapters: {available}"
        )
    # Pass adapter_config to adapters that accept it
    import inspect
    sig = inspect.signature(cls.__init__)
    if "adapter_config" in sig.parameters:
        return cls(adapter_config=adapter_config)
    return cls()
