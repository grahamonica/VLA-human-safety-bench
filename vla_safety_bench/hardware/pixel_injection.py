from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from vla_safety_bench.hardware.hardware_io import CameraIntrinsics, CameraPose
from vla_safety_bench.sim.mesh_assets import MeshAssetLibrary
from vla_safety_bench.sim.mujoco_backend import _human_body_xml, _mesh_asset_xml
from vla_safety_bench.types import HumanState, Vector3


def render_human_overlay(
    cam_pose: CameraPose,
    cam_intrinsics: CameraIntrinsics,
    humans: Iterable[HumanState],
    mesh_assets: MeshAssetLibrary | None = None,
) -> tuple[Any, Any]:
    """Render the manifest-backed human mesh from the given camera pose.

    Builds a focused MuJoCo scene that contains only the human body/bodies plus a
    single camera placed at the supplied world-frame pose, and returns
    `(rgb, alpha)` where rgb is a HxWx3 uint8 numpy array and alpha is a HxW
    float32 mask in [0, 1]. Pixels that did not hit any human geometry have
    alpha 0; pixels that hit a human have alpha 1. The mask is derived from the
    depth buffer against MuJoCo's far plane, so it is rendered geometry, not a
    naive bounding box.
    """

    import mujoco
    import numpy as np

    humans_list = list(humans)
    if not humans_list:
        rgb = np.zeros((cam_intrinsics.height, cam_intrinsics.width, 3), dtype=np.uint8)
        alpha = np.zeros((cam_intrinsics.height, cam_intrinsics.width), dtype=np.float32)
        return rgb, alpha
    if mesh_assets is None:
        raise ValueError("Hardware injection requires mesh assets for rendered humans.")

    xml = _build_human_only_xml(cam_pose, cam_intrinsics, humans_list, mesh_assets)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=cam_intrinsics.height, width=cam_intrinsics.width)
    try:
        renderer.disable_depth_rendering()
        renderer.update_scene(data, camera="injection_cam")
        rgb = renderer.render().copy()

        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera="injection_cam")
        depth = renderer.render().copy()
    finally:
        renderer.close()

    far_threshold = float(model.stat.extent) * float(model.vis.map.zfar) * 0.99
    alpha = (depth < far_threshold).astype(np.float32)
    return rgb, alpha


def composite_overlay(real_frame: Any, overlay_rgb: Any, overlay_alpha: Any) -> Any:
    """Alpha-composite overlay_rgb onto real_frame using overlay_alpha.

    All inputs must agree on (height, width). real_frame and overlay_rgb are
    HxWx3 uint8; overlay_alpha is HxW float in [0, 1]. Returns HxWx3 uint8.
    """

    import numpy as np

    if real_frame.shape != overlay_rgb.shape:
        raise ValueError(
            f"Real frame {real_frame.shape} does not match overlay shape {overlay_rgb.shape}"
        )
    if overlay_alpha.shape[:2] != real_frame.shape[:2]:
        raise ValueError(
            f"Alpha shape {overlay_alpha.shape} does not match frame shape {real_frame.shape[:2]}"
        )
    real = real_frame.astype(np.float32)
    overlay = overlay_rgb.astype(np.float32)
    alpha = np.clip(overlay_alpha.astype(np.float32), 0.0, 1.0)[:, :, None]
    blended = overlay * alpha + real * (1.0 - alpha)
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def save_frame_png(frame: Any, output_path: Path) -> str:
    from PIL import Image

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(output_path)
    return str(output_path)


def _build_human_only_xml(
    cam_pose: CameraPose,
    cam_intrinsics: CameraIntrinsics,
    humans: list[HumanState],
    mesh_assets: MeshAssetLibrary,
) -> str:
    mesh_asset_xml = _mesh_asset_xml(mesh_assets)
    human_xml = "\n".join(_human_body_xml(human, mesh_assets) for human in humans)
    return f"""
<mujoco model="vla_human_overlay">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.01" gravity="0 0 0"/>
  <visual>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.4 0.4 0.4" specular="0.05 0.05 0.05"/>
    <map zfar="50" znear="0.05"/>
  </visual>
  <asset>
{mesh_asset_xml}
  </asset>
  <worldbody>
    <light name="overlay_key" pos="0 -1.4 2.2" dir="0 0 -1" directional="true"/>
    <light name="overlay_fill" pos="1.5 1.2 1.8" dir="-1 -1 -1" directional="true" diffuse="0.35 0.35 0.35"/>
{_camera_xml(cam_pose, cam_intrinsics)}
{human_xml}
  </worldbody>
</mujoco>
""".strip()


def _camera_xml(cam_pose: CameraPose, cam_intrinsics: CameraIntrinsics) -> str:
    pos = _vec(cam_pose.position_m)
    x_axis = _vec(cam_pose.x_axis())
    y_axis = _vec(cam_pose.y_axis())
    return (
        f'    <camera name="injection_cam" pos="{pos}" '
        f'xyaxes="{x_axis} {y_axis}" fovy="{cam_intrinsics.fovy_deg:g}"/>'
    )


def _vec(value: Vector3) -> str:
    return f"{value[0]:.9g} {value[1]:.9g} {value[2]:.9g}"
