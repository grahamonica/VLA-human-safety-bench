from __future__ import annotations

from pathlib import Path

from vla_safety_bench.scenarios import ScenarioSpec
from vla_safety_bench.sim.kinematic_backend import KinematicSimulation
from vla_safety_bench.sim.mujoco_backend import (
    kuka_render_joint_positions,
    kuka_scenario_mujoco_xml,
    render_scene_png,
    scenario_mujoco_xml,
)
from vla_safety_bench.sim.mesh_assets import MeshAssetLibrary, load_mesh_asset_library
from vla_safety_bench.types import Observation


class MujocoScenarioSimulation(KinematicSimulation):
    """Kinematic safety stepping with MuJoCo-rendered camera observations."""

    def __init__(
        self,
        scenario: ScenarioSpec,
        output_dir: Path,
        *,
        render_frames: bool = True,
        backend_name: str = "mujoco-scenario+kinematic",
        camera: str = "bench_cam",
        use_kuka: bool = False,
        mesh_assets: MeshAssetLibrary | str | Path | None = None,
    ) -> None:
        super().__init__(
            scenario,
            output_dir,
            render_frames=render_frames,
            backend_name=backend_name,
        )
        self.camera = camera
        self.use_kuka = use_kuka
        self.mesh_assets = load_mesh_asset_library(mesh_assets)

    def observe(self, step_index: int) -> Observation:
        self.world.humans = self.scenario.humans_at(step_index)
        image_path = None
        if self.render_frames:
            image_path = self._render_mujoco_frame(step_index)
        return Observation(
            scenario_id=self.scenario.id,
            prompt=self.scenario.prompt,
            step_index=step_index,
            humans=list(self.world.humans),
            objects=list(self.world.objects.values()),
            image_path=image_path,
            metadata={
                "backend": self.backend_name,
                "category": self.scenario.category,
                "tags": self.scenario.tags,
                "mujoco_render": image_path is not None,
                "camera": self.camera,
                "robot_model": "kuka_iiwa_14_menagerie" if self.use_kuka else "primitive_proxy",
                "mesh_assets": str(self.mesh_assets.manifest_path) if self.mesh_assets else None,
                "kuka_render_path": kuka_render_joint_positions(step_index, self.scenario.max_steps)
                if self.use_kuka
                else None,
                "world_state": self.world.to_dict(),
                "safety_context": self._safety_context(),
            },
        )

    def _render_mujoco_frame(self, step_index: int) -> str | None:
        import mujoco

        xml = (
            kuka_scenario_mujoco_xml(
                objects=list(self.world.objects.values()),
                humans=list(self.world.humans),
                mesh_assets=self.mesh_assets,
            )
            if self.use_kuka
            else scenario_mujoco_xml(
                objects=list(self.world.objects.values()),
                humans=list(self.world.humans),
                mesh_assets=self.mesh_assets,
            )
        )
        model = mujoco.MjModel.from_xml_string(xml)
        frame_path = self.output_dir / "frames" / self.scenario.id / f"{step_index:03d}.png"
        return render_scene_png(
            model,
            frame_path,
            camera=self.camera,
            joint_positions=kuka_render_joint_positions(step_index, self.scenario.max_steps)
            if self.use_kuka
            else None,
        )
