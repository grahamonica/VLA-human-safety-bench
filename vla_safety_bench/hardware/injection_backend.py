from __future__ import annotations

from pathlib import Path

from vla_safety_bench.hardware.hardware_io import HardwareIO
from vla_safety_bench.hardware.pixel_injection import (
    composite_overlay,
    render_human_overlay,
    save_frame_png,
)
from vla_safety_bench.scenarios import ScenarioSpec
from vla_safety_bench.sim.kinematic_backend import KinematicSimulation
from vla_safety_bench.sim.mesh_assets import MeshAssetLibrary, load_mesh_asset_library
from vla_safety_bench.types import JsonDict, Observation, RobotAction


class HardwareInjectionSimulation(KinematicSimulation):
    """Real-hardware backend: composite synthetic human pixels into a real cam feed.

    For each step:
      1. Read the live wrist-cam frame, intrinsics, and pose from `hardware_io`.
      2. Render the scenario's synthetic human(s) from that exact cam pose using
         a small MuJoCo scene (the same human body/mesh definition used by the
         simulation backends).
      3. Alpha-composite the rendered human onto the real frame and save the
         result; that path is what the VLA sees as `image_path`.
      4. Forward the VLA's action to `hardware_io.submit_action(...)` and record
         the controller's feedback in the trace.

    No real human is ever present in the workspace. The benchmark measures
    whether the VLA reacts to the perceived human in pixels.
    """

    def __init__(
        self,
        scenario: ScenarioSpec,
        output_dir: Path,
        *,
        hardware_io: HardwareIO,
        camera: str = "wrist_cam",
        mesh_assets: MeshAssetLibrary | str | Path | None = None,
        backend_name: str = "hardware-injection",
    ) -> None:
        super().__init__(
            scenario,
            output_dir,
            render_frames=True,
            backend_name=backend_name,
        )
        self.hardware_io = hardware_io
        self.camera = camera
        self.mesh_assets = load_mesh_asset_library(mesh_assets)
        self._last_action_feedback: JsonDict | None = None
        self._last_cam_pose: JsonDict | None = None

    def observe(self, step_index: int) -> Observation:
        self.world.humans = self.scenario.humans_at(step_index)
        self._safety_context_cache = None

        real_frame = self.hardware_io.read_camera_frame(self.camera)
        intrinsics = self.hardware_io.read_camera_intrinsics(self.camera)
        cam_pose = self.hardware_io.read_camera_pose(self.camera)
        self._last_cam_pose = {
            "position_m": list(cam_pose.position_m),
            "rotation_matrix": [list(row) for row in cam_pose.rotation_matrix],
            "intrinsics": {
                "width": intrinsics.width,
                "height": intrinsics.height,
                "fovy_deg": intrinsics.fovy_deg,
            },
        }

        overlay_rgb, overlay_alpha = render_human_overlay(
            cam_pose,
            intrinsics,
            self.world.humans,
            mesh_assets=self.mesh_assets,
        )
        composited = composite_overlay(real_frame, overlay_rgb, overlay_alpha)
        frame_path = self.output_dir / "frames" / self.scenario.id / f"{step_index:03d}.png"
        image_path = save_frame_png(composited, frame_path)

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
                "camera": self.camera,
                "hardware_camera_pose": self._last_cam_pose,
                "hardware_robot_state": self.hardware_io.read_robot_state(),
                "mesh_assets": str(self.mesh_assets.manifest_path) if self.mesh_assets else None,
                "world_state": self.world.to_dict(),
                "safety_context": self._safety_context(),
            },
        )

    def apply_action(self, action: RobotAction) -> JsonDict:
        feedback = self.hardware_io.submit_action(action.to_dict())
        self._last_action_feedback = feedback
        result = super().apply_action(action)
        result["hardware_feedback"] = feedback
        return result
