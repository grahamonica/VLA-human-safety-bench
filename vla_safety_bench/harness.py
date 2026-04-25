from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from vla_safety_bench.adapters.base import AdapterProtocol
from vla_safety_bench.scenarios import ScenarioSet
from vla_safety_bench.scoring import ScenarioResult, evaluate_trace
from vla_safety_bench.sim.kinematic_backend import KinematicSimulation
from vla_safety_bench.sim.mujoco_scenario_backend import MujocoScenarioSimulation
from vla_safety_bench.types import JsonDict, RobotAction, TraceStep


@dataclass(frozen=True)
class BenchmarkReport:
    benchmark_name: str
    adapter: str
    started_at: str
    results: list[ScenarioResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for result in self.results if result.passed) / len(self.results)

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    def to_dict(self) -> JsonDict:
        return {
            "benchmark_name": self.benchmark_name,
            "adapter": self.adapter,
            "started_at": self.started_at,
            "passed": self.passed,
            "pass_rate": self.pass_rate,
            "scenario_count": len(self.results),
            "results": [result.to_dict() for result in self.results],
        }


class BenchmarkHarness:
    def __init__(
        self,
        scenario_set: ScenarioSet,
        adapter: AdapterProtocol,
        *,
        adapter_name: str,
        output_dir: str | Path,
        render_frames: bool = True,
        backend: str = "kinematic",
        camera: str = "bench_cam",
        mesh_assets: str | Path | None = None,
    ) -> None:
        self.scenario_set = scenario_set
        self.adapter = adapter
        self.adapter_name = adapter_name
        self.output_dir = Path(output_dir)
        self.render_frames = render_frames
        self.backend = backend
        self.camera = camera
        self.mesh_assets = mesh_assets

    def run(self) -> BenchmarkReport:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = self.output_dir / "trace.jsonl"
        started_at = datetime.now(timezone.utc).isoformat()
        results: list[ScenarioResult] = []
        self._preflight_adapter()

        with trace_path.open("w", encoding="utf-8") as trace_file:
            for scenario in self.scenario_set.scenarios:
                simulation = self._create_simulation(scenario)
                trace: list[TraceStep] = []
                for step_index in range(scenario.max_steps):
                    observation = simulation.observe(step_index)
                    action_payload = self.adapter.act(observation.to_dict())
                    action = RobotAction.from_payload(action_payload)
                    if action.type == "unknown":
                        raise RuntimeError(
                            f"Adapter {self.adapter_name!r} returned an unknown/malformed action "
                            f"for scenario {scenario.id} step {step_index}: {action.raw}"
                        )
                    sim_feedback = simulation.apply_action(action)
                    trace_step = TraceStep(observation=observation, action=action)
                    trace.append(trace_step)
                    trace_file.write(
                        json.dumps(
                            {
                                "scenario_id": scenario.id,
                                "step_index": step_index,
                                "simulation": sim_feedback,
                                **trace_step.to_dict(),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                results.append(evaluate_trace(scenario, trace))

        report = BenchmarkReport(
            benchmark_name=self.scenario_set.name,
            adapter=self.adapter_name,
            started_at=started_at,
            results=results,
        )
        summary_path = self.output_dir / "summary.json"
        summary_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report

    def _create_simulation(self, scenario):
        if self.backend == "kinematic":
            return KinematicSimulation(scenario, self.output_dir, render_frames=self.render_frames)
        if self.backend == "mujoco-minimal":
            return MujocoScenarioSimulation(
                scenario,
                self.output_dir,
                render_frames=self.render_frames,
                backend_name="mujoco-scenario+kinematic",
                camera=self.camera,
                use_kuka=False,
                mesh_assets=self.mesh_assets,
            )
        if self.backend == "mujoco-kuka":
            return MujocoScenarioSimulation(
                scenario,
                self.output_dir,
                render_frames=self.render_frames,
                backend_name="mujoco-kuka+kinematic",
                camera=self.camera,
                use_kuka=True,
                mesh_assets=self.mesh_assets,
            )
        raise ValueError(f"Unknown simulation backend: {self.backend}")

    def _preflight_adapter(self) -> None:
        preflight = getattr(self.adapter, "preflight", None)
        if callable(preflight):
            preflight()
