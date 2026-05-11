"""Tests for benchd_harness.adapters."""

import pytest

from benchd_harness.adapters import get_adapter
from benchd_harness.adapters.echo import EchoAdapter
from benchd_harness.adapters.null import NullAdapter


class TestEchoAdapter:
    def test_echo_adapter_stores_and_recalls(self):
        adapter = EchoAdapter()
        adapter.ingest([
            {"role": "user", "content": "My name is Marcus Chen."},
            {"role": "assistant", "content": "Nice to meet you, Marcus."},
            {"role": "user", "content": "I work at Anthropic on alignment research."},
        ])
        result = adapter.recall("What is the user's name?")
        # The echo adapter does keyword matching; the turn mentioning "name"
        # should be returned, and it contains "Marcus Chen".
        assert "Marcus" in result

    def test_echo_adapter_reset(self):
        adapter = EchoAdapter()
        adapter.ingest([
            {"role": "user", "content": "My name is Marcus Chen."},
        ])
        adapter.reset()
        result = adapter.recall("What is the user's name?")
        assert result == ""

    def test_echo_adapter_recall_empty(self):
        adapter = EchoAdapter()
        result = adapter.recall("anything")
        assert result == ""


class TestNullAdapter:
    def test_null_adapter_returns_empty(self):
        adapter = NullAdapter()
        adapter.ingest([
            {"role": "user", "content": "My name is Marcus Chen."},
        ])
        result = adapter.recall("What is the user's name?")
        assert result == ""


class TestAdapterFactory:
    def test_adapter_factory_echo(self):
        adapter = get_adapter("echo")
        assert isinstance(adapter, EchoAdapter)

    def test_adapter_factory_null(self):
        adapter = get_adapter("null")
        assert isinstance(adapter, NullAdapter)

    def test_adapter_factory_unknown(self):
        with pytest.raises(ValueError, match="Unknown adapter"):
            get_adapter("unknown")
