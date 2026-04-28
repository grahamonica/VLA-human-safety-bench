from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from vla_safety_bench.adapters.base import AdapterProtocol
from vla_safety_bench.hardware.hardware_io import HardwareIO
from vla_safety_bench.hardware.injection_backend import HardwareInjectionSimulation
from vla_safety_bench.scenarios import ScenarioSet
from vla_safety_bench.scoring import ScenarioResult, evaluate_trace
from vla_safety_bench.sim.kinematic_backend import KinematicSimulation
from vla_safety_bench.sim.mujoco_scenario_backend import MujocoScenarioSimulation
from vla_safety_bench.types import JsonDict, RobotAction, TraceStep
from vla_safety_bench.video import write_camera_slideshow


@dataclass(frozen=True)
class BenchmarkReport:
    benchmark_name: str
    adapter: str
    started_at: str
    results: list[ScenarioResult] = field(default_factory=list)
    video_artifacts: list[JsonDict] = field(default_factory=list)

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
            "video_artifacts": list(self.video_artifacts),
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
        hardware_io: HardwareIO | None = None,
        create_videos: bool = True,
        video_cameras: Sequence[str] | None = None,
    ) -> None:
        self.scenario_set = scenario_set
        self.adapter = adapter
        self.adapter_name = adapter_name
        self.output_dir = Path(output_dir)
        self.render_frames = render_frames
        self.backend = backend
        self.camera = camera
        self.mesh_assets = mesh_assets
        self.hardware_io = hardware_io
        self.create_videos = create_videos
        self.video_cameras = tuple(video_cameras) if video_cameras is not None else None

    def run(self) -> BenchmarkReport:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = self.output_dir / "trace.jsonl"
        started_at = datetime.now(timezone.utc).isoformat()
        results: list[ScenarioResult] = []
        video_artifacts: list[JsonDict] = []
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
                video_artifact = self._write_scenario_video(scenario.id, trace)
                if video_artifact is not None:
                    video_artifacts.append(video_artifact)

        report = BenchmarkReport(
            benchmark_name=self.scenario_set.name,
            adapter=self.adapter_name,
            started_at=started_at,
            results=results,
            video_artifacts=video_artifacts,
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
                video_cameras=self._video_cameras_for_backend(),
            )
        if self.backend == "mujoco-kuka":
            return MujocoScenarioSimulation(
                scenario,
                self.output_dir,
                render_frames=self.render_frames,
                backend_name="mujoco-kuka+physics",
                camera=self.camera,
                use_kuka=True,
                mesh_assets=self.mesh_assets,
                video_cameras=self._video_cameras_for_backend(),
            )
        if self.backend == "hardware-injection":
            if self.hardware_io is None:
                raise ValueError(
                    "hardware-injection backend requires a HardwareIO implementation. "
                    "Pass hardware_io=... to BenchmarkHarness or use the harness API "
                    "from Python; the CLI cannot construct one because it needs a "
                    "concrete robot driver."
                )
            return HardwareInjectionSimulation(
                scenario,
                self.output_dir,
                hardware_io=self.hardware_io,
                camera=self.camera,
                mesh_assets=self.mesh_assets,
            )
        raise ValueError(f"Unknown simulation backend: {self.backend}")

    def _preflight_adapter(self) -> None:
        preflight = getattr(self.adapter, "preflight", None)
        if callable(preflight):
            preflight()

    def _write_scenario_video(self, scenario_id: str, trace: list[TraceStep]) -> JsonDict | None:
        if not self.render_frames or not self.create_videos:
            return None
        frames_by_step: list[dict[str, str]] = []
        for step in trace:
            camera_frames = step.observation.metadata.get("camera_frames", {})
            if isinstance(camera_frames, dict) and camera_frames:
                frames_by_step.append(
                    {
                        str(camera): str(path)
                        for camera, path in camera_frames.items()
                        if path is not None
                    }
                )
            elif step.observation.image_path:
                frames_by_step.append({self.camera: step.observation.image_path})

        artifact = write_camera_slideshow(
            scenario_id=scenario_id,
            frames_by_step=frames_by_step,
            output_path=self.output_dir / "videos" / f"{scenario_id}.gif",
        )
        return None if artifact is None else artifact.to_dict()

    def _video_cameras_for_backend(self) -> tuple[str, ...] | None:
        if self.video_cameras is not None:
            return self.video_cameras
        if self.backend == "mujoco-kuka":
            defaults = ("bench_cam", "overhead_cam", "wrist_cam")
            return defaults if self.camera in defaults else (self.camera, *defaults)
        if self.backend == "mujoco-minimal":
            defaults = ("bench_cam", "overhead_cam")
            return defaults if self.camera in defaults else (self.camera, *defaults)
        if self.backend == "hardware-injection":
            return (self.camera,)
        return None
