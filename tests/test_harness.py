import json

import pytest

from vla_safety_bench.adapters.base import load_adapter
from vla_safety_bench.harness import BenchmarkHarness
from vla_safety_bench.scenarios import load_scenario_set


def test_rule_based_adapter_passes_smoke(tmp_path):
    scenario_set = load_scenario_set("configs/smoke.json")
    harness = BenchmarkHarness(
        scenario_set,
        load_adapter("rule_based"),
        adapter_name="rule_based",
        output_dir=tmp_path,
        render_frames=False,
    )
    report = harness.run()
    assert report.passed
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["passed"] is True
    assert (tmp_path / "trace.jsonl").exists()


def test_unsafe_adapter_fails_benchmark(tmp_path):
    scenario_set = load_scenario_set("configs/benchmark.json")
    harness = BenchmarkHarness(
        scenario_set,
        load_adapter("unsafe"),
        adapter_name="unsafe",
        output_dir=tmp_path,
        render_frames=False,
    )
    report = harness.run()
    assert not report.passed
    assert any(not result.passed for result in report.results)


def test_harness_rejects_malformed_adapter_output(tmp_path):
    class MalformedAdapter:
        def act(self, _observation):
            return ["move"]

    scenario_set = load_scenario_set("configs/smoke.json")
    harness = BenchmarkHarness(
        scenario_set,
        MalformedAdapter(),
        adapter_name="malformed",
        output_dir=tmp_path,
        render_frames=False,
    )
    with pytest.raises(ValueError, match="JSON object"):
        harness.run()


def test_harness_rejects_unknown_adapter_action(tmp_path):
    class UnknownAdapter:
        def act(self, _observation):
            return {"type": "unknown"}

    scenario_set = load_scenario_set("configs/smoke.json")
    harness = BenchmarkHarness(
        scenario_set,
        UnknownAdapter(),
        adapter_name="unknown",
        output_dir=tmp_path,
        render_frames=False,
    )
    with pytest.raises(RuntimeError, match="unknown/malformed action"):
        harness.run()
