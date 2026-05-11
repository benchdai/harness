"""Tests for benchd_harness.runner."""

import json

from benchd_harness.adapters.echo import EchoAdapter
from benchd_harness.adapters.null import NullAdapter
from benchd_harness.benchmarks.smoke import SmokeBenchmark
from benchd_harness.runner import BenchmarkRunner
from benchd_harness.signing.local import LocalSigner, SignedManifest, verify_signature


class TestRunnerEchoAdapter:
    def test_runner_echo_adapter(self):
        runner = BenchmarkRunner(
            adapter=EchoAdapter(),
            benchmark=SmokeBenchmark(),
        )
        result = runner.run()
        manifest = result.manifest

        # Echo adapter should pass at least some recall questions
        passed = manifest["summary"]["passed"]
        assert passed > 0, "Echo adapter should pass at least some benchmark items"

    def test_runner_echo_adapter_scores_structure(self):
        runner = BenchmarkRunner(
            adapter=EchoAdapter(),
            benchmark=SmokeBenchmark(),
        )
        result = runner.run()
        manifest = result.manifest

        assert manifest["summary"]["total_questions"] == 10
        assert "verified" in manifest["scores"]
        assert "nuance" in manifest["scores"]


class TestRunnerNullAdapter:
    def test_runner_null_adapter(self):
        runner = BenchmarkRunner(
            adapter=NullAdapter(),
            benchmark=SmokeBenchmark(),
        )
        result = runner.run()
        manifest = result.manifest

        # Null adapter returns "" for everything, so all scored items should fail.
        # The 7 exact/regex items should all be scored and failed.
        # The 3 llm items should be pending.
        scored = manifest["summary"]["scored_questions"]
        passed = manifest["summary"]["passed"]
        pending = manifest["summary"]["pending_questions"]

        assert scored == 7
        assert passed == 0, "Null adapter should pass zero scored items"
        assert pending == 3


class TestRunnerSignedManifest:
    def test_runner_produces_signed_manifest(self):
        runner = BenchmarkRunner(
            adapter=EchoAdapter(),
            benchmark=SmokeBenchmark(),
        )
        result = runner.run()
        assert isinstance(result, SignedManifest)
        assert verify_signature(result.to_dict()) is True

    def test_runner_saves_to_output_dir(self, tmp_path):
        runner = BenchmarkRunner(
            adapter=EchoAdapter(),
            benchmark=SmokeBenchmark(),
            output_dir=tmp_path,
        )
        result = runner.run()
        run_id = result.manifest["run_id"]

        manifest_path = tmp_path / run_id / "manifest.signed.json"
        assert manifest_path.exists()

        # Verify the saved file is valid JSON and matches the result
        saved = json.loads(manifest_path.read_text())
        assert saved["manifest"]["run_id"] == run_id
        assert verify_signature(saved) is True
