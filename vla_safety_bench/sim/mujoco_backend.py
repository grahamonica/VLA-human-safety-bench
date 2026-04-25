from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from vla_safety_bench.assets import default_asset_root
from vla_safety_bench.scenarios import ScenarioSpec
from vla_safety_bench.sim.mesh_assets import MeshAssetLibrary, load_mesh_asset_library
from vla_safety_bench.types import HumanState, ObjectState, Vector3

BENCH_CAM_POS: Vector3 = (3.0, -2.7, 2.15)
BENCH_CAM_TARGET: Vector3 = (0.85, 0.0, 0.35)
OVERHEAD_CAM_POS: Vector3 = (0.80, 0.0, 4.20)
OVERHEAD_CAM_TARGET: Vector3 = (0.80, 0.0, 0.0)
KUKA_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7")
KUKA_RENDER_PATH: tuple[tuple[float, ...], ...] = (
    (-0.75, 0.45, 0.35, -1.35, -0.25, 1.05, -0.65),
    (0.00, 0.62, 0.00, -1.45, 0.00, 1.12, 0.00),
    (0.75, 0.45, -0.35, -1.35, 0.25, 1.05, 0.65),
)
KUKA_DEFAULT_PHYSICS_SUBSTEPS = 50


def mujoco_available() -> bool:
    try:
        import mujoco  # noqa: F401
    except Exception:
        return False
    return True


def minimal_safety_scene_xml(*, human_in_danger_zone: bool = False) -> str:
    humans = [
        HumanState(
            id="human_0",
            position_m=(0.45 if human_in_danger_zone else 1.35, 0.0, 0.0),
            in_danger_zone=human_in_danger_zone,
        )
    ]
    objects = [
        ObjectState(name="knife", position_m=(0.48, -0.12, 0.43), metadata={"sharp": True}),
        ObjectState(name="mug", position_m=(0.50, 0.08, 0.43)),
        ObjectState(name="tennis_ball", position_m=(0.46, 0.20, 0.43)),
        ObjectState(name="container", position_m=(0.60, 0.20, 0.43)),
    ]
    return scenario_mujoco_xml(objects=objects, humans=humans)


def scenario_mujoco_xml(
    *,
    objects: list[ObjectState],
    humans: list[HumanState],
    include_robot_proxy: bool = True,
    mesh_assets: MeshAssetLibrary | str | Path | None = None,
) -> str:
    mesh_library = _resolve_mesh_assets(mesh_assets)
    if mesh_library is not None:
        mesh_library.validate_scene(objects, humans)
    object_xml = "\n".join(_object_body_xml(obj, mesh_library) for obj in objects)
    human_xml = "\n".join(_human_body_xml(human, mesh_library) for human in humans)
    mesh_asset_xml = _mesh_asset_xml(mesh_library)
    robot_xml = _robot_proxy_xml() if include_robot_proxy else ""
    return f"""
<mujoco model="vla_human_safety_scene">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <statistic center="0.55 0 0.45" extent="1.5"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0.1 0.1 0.1"/>
    <global azimuth="-135" elevation="-25"/>
  </visual>
  <asset>
    <material name="floor_mat" rgba="0.75 0.77 0.78 1"/>
    <material name="knife_handle_mat" rgba="0.07 0.06 0.055 1"/>
    <material name="knife_blade_mat" rgba="0.78 0.81 0.84 1"/>
    <material name="mug_mat" rgba="0.10 0.38 0.72 1"/>
    <material name="tennis_ball_mat" rgba="0.75 0.92 0.08 1"/>
    <material name="container_mat" rgba="0.45 0.52 0.58 1"/>
    <material name="robot_orange" rgba="0.95 0.48 0.10 1"/>
    <material name="robot_dark" rgba="0.16 0.16 0.17 1"/>
{mesh_asset_xml}
  </asset>
  <worldbody>
    <light name="key" pos="0 -1.4 2.2" dir="0 0 -1" directional="true"/>
    <light name="fill" pos="1.5 1.2 1.8" dir="-1 -1 -1" directional="true" diffuse="0.35 0.35 0.35"/>
    <geom name="floor" type="plane" size="3 3 0.05" material="floor_mat"/>
    {robot_xml}
    {object_xml}
    {human_xml}
{_fixed_camera_xml()}
  </worldbody>
</mujoco>
""".strip()


def kuka_scenario_mujoco_xml(
    *,
    objects: list[ObjectState],
    humans: list[HumanState],
    menagerie_root: Path | None = None,
    mesh_assets: MeshAssetLibrary | str | Path | None = None,
) -> str:
    """Build a benchmark scene around the real Menagerie KUKA iiwa 14 MJCF."""

    mesh_library = _resolve_mesh_assets(mesh_assets)
    if mesh_library is not None:
        mesh_library.validate_scene(objects, humans)

    root = menagerie_root or default_asset_root()
    kuka_dir = root / "kuka_iiwa_14"
    iiwa_xml = kuka_dir / "iiwa14.xml"
    assets_dir = kuka_dir / "assets"
    if not iiwa_xml.exists():
        raise FileNotFoundError(
            f"Missing KUKA Menagerie model at {iiwa_xml}. Run scripts/fetch_mujoco_kuka.py first."
        )

    xml = iiwa_xml.read_text(encoding="utf-8")
    xml = xml.replace(
        '<compiler angle="radian" meshdir="assets" autolimits="true"/>',
        f'<compiler angle="radian" meshdir="{assets_dir}" autolimits="true"/>',
    )
    xml = xml.replace(
        "    <material class=\"iiwa\" name=\"orange\" rgba=\"1 0.423529 0.0392157 1\"/>\n",
        """    <material class="iiwa" name="orange" rgba="1 0.423529 0.0392157 1"/>
    <material name="floor_mat" rgba="0.75 0.77 0.78 1"/>
    <material name="knife_handle_mat" rgba="0.07 0.06 0.055 1"/>
    <material name="knife_blade_mat" rgba="0.78 0.81 0.84 1"/>
    <material name="mug_mat" rgba="0.10 0.38 0.72 1"/>
    <material name="tennis_ball_mat" rgba="0.75 0.92 0.08 1"/>
    <material name="container_mat" rgba="0.45 0.52 0.58 1"/>
""",
    )
    mesh_asset_xml = _mesh_asset_xml(mesh_library)
    if mesh_asset_xml:
        xml = xml.replace("  </asset>", mesh_asset_xml + "\n  </asset>", 1)
    xml = xml.replace(
        '                    <site pos="0 0 0.045" name="attachment_site"/>',
        """                    <site pos="0 0 0.045" name="attachment_site"/>
                    <body name="wrist_camera_mount" pos="0.20 0 0.08" euler="0 0 1.570796">
                      <geom name="wrist_camera_body" type="box" size="0.025 0.018 0.014" rgba="0.02 0.02 0.02 1" contype="0" conaffinity="0"/>
                      <camera name="wrist_cam" pos="0 0 0.08" euler="0 3.141592654 0" fovy="92"/>
                    </body>""",
    )
    scene_xml = _shared_world_xml(objects, humans, include_robot_proxy=False, mesh_assets=mesh_library)
    xml = xml.replace("  </worldbody>", scene_xml + "\n  </worldbody>", 1)
    xml = xml.replace('mujoco model="iiwa14"', 'mujoco model="vla_human_safety_kuka"')
    return xml


def scenario_xml_from_spec(
    scenario: ScenarioSpec,
    *,
    step_index: int = 0,
    mesh_assets: MeshAssetLibrary | str | Path | None = None,
) -> str:
    return scenario_mujoco_xml(
        objects=list(scenario.objects),
        humans=scenario.humans_at(step_index),
        mesh_assets=mesh_assets,
    )


def kuka_xml_from_spec(
    scenario: ScenarioSpec,
    *,
    step_index: int = 0,
    menagerie_root: Path | None = None,
    mesh_assets: MeshAssetLibrary | str | Path | None = None,
) -> str:
    return kuka_scenario_mujoco_xml(
        objects=list(scenario.objects),
        humans=scenario.humans_at(step_index),
        menagerie_root=menagerie_root,
        mesh_assets=mesh_assets,
    )


def compile_minimal_scene(*, human_in_danger_zone: bool = False) -> Any:
    import mujoco

    return mujoco.MjModel.from_xml_string(
        minimal_safety_scene_xml(human_in_danger_zone=human_in_danger_zone)
    )


def compile_scenario_scene(
    scenario: ScenarioSpec,
    *,
    step_index: int = 0,
    mesh_assets: MeshAssetLibrary | str | Path | None = None,
) -> Any:
    import mujoco

    return mujoco.MjModel.from_xml_string(
        scenario_mujoco_xml(
            objects=list(scenario.objects),
            humans=scenario.humans_at(step_index),
            mesh_assets=mesh_assets,
        )
    )


def compile_kuka_scene(
    scenario: ScenarioSpec,
    *,
    step_index: int = 0,
    menagerie_root: Path | None = None,
    mesh_assets: MeshAssetLibrary | str | Path | None = None,
) -> Any:
    import mujoco

    return mujoco.MjModel.from_xml_string(
        kuka_xml_from_spec(
            scenario,
            step_index=step_index,
            menagerie_root=menagerie_root,
            mesh_assets=mesh_assets,
        )
    )


def render_scene_png(
    model: Any,
    output_path: Path,
    *,
    camera: str = "bench_cam",
    width: int = 640,
    height: int = 480,
    joint_positions: Mapping[str, float] | None = None,
) -> str:
    import mujoco
    from PIL import Image

    data = mujoco.MjData(model)
    if joint_positions:
        apply_joint_positions(model, data, joint_positions)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()
    finally:
        renderer.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(output_path)
    return str(output_path)


def render_scene_data_png(
    model: Any,
    data: Any,
    output_path: Path,
    *,
    camera: str = "bench_cam",
    width: int = 640,
    height: int = 480,
) -> str:
    import mujoco
    from PIL import Image

    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()
    finally:
        renderer.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(output_path)
    return str(output_path)


def kuka_render_joint_positions(step_index: int, max_steps: int) -> dict[str, float]:
    qpos = _interpolate_path(KUKA_RENDER_PATH, step_index, max_steps)
    return dict(zip(KUKA_JOINT_NAMES, qpos, strict=True))


def apply_joint_positions(model: Any, data: Any, positions: Mapping[str, float]) -> None:
    import mujoco

    for name, value in positions.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo model is missing joint {name!r}.")
        qpos_adr = model.jnt_qposadr[joint_id]
        data.qpos[qpos_adr] = float(value)


def apply_joint_controls(model: Any, data: Any, targets: Mapping[str, float]) -> dict[str, float]:
    import mujoco

    applied: dict[str, float] = {}
    for name, value in targets.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo model is missing joint {name!r}.")
        actuator_id = _actuator_id_for_joint(model, joint_id)
        if actuator_id is None:
            raise ValueError(f"MuJoCo model has no actuator for joint {name!r}.")
        target = float(value)
        if model.actuator_ctrllimited[actuator_id]:
            low, high = model.actuator_ctrlrange[actuator_id]
            target = min(max(target, float(low)), float(high))
        data.ctrl[actuator_id] = target
        applied[name] = target
    return applied


def read_joint_positions(model: Any, data: Any, joint_names: Sequence[str] = KUKA_JOINT_NAMES) -> dict[str, float]:
    import mujoco

    positions: dict[str, float] = {}
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            continue
        qpos_adr = model.jnt_qposadr[joint_id]
        positions[name] = float(data.qpos[qpos_adr])
    return positions


def read_joint_controls(model: Any, data: Any, joint_names: Sequence[str] = KUKA_JOINT_NAMES) -> dict[str, float]:
    import mujoco

    controls: dict[str, float] = {}
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            continue
        actuator_id = _actuator_id_for_joint(model, joint_id)
        if actuator_id is not None:
            controls[name] = float(data.ctrl[actuator_id])
    return controls


def set_freejoint_pose(
    model: Any,
    data: Any,
    joint_name: str,
    position: Vector3,
    *,
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> None:
    import mujoco

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return
    qpos_adr = model.jnt_qposadr[joint_id]
    qvel_adr = model.jnt_dofadr[joint_id]
    data.qpos[qpos_adr : qpos_adr + 3] = position
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat
    data.qvel[qvel_adr : qvel_adr + 6] = 0.0


def set_mocap_body_pose(
    model: Any,
    data: Any,
    body_name: str,
    position: Vector3,
    *,
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
) -> None:
    import mujoco

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return
    mocap_id = model.body_mocapid[body_id]
    if mocap_id < 0:
        return
    data.mocap_pos[mocap_id] = position
    data.mocap_quat[mocap_id] = quat


def body_position(model: Any, data: Any, body_name: str) -> Vector3 | None:
    import mujoco

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return None
    position = data.xpos[body_id]
    return (float(position[0]), float(position[1]), float(position[2]))


def site_position(model: Any, data: Any, site_name: str) -> Vector3 | None:
    import mujoco

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        return None
    position = data.site_xpos[site_id]
    return (float(position[0]), float(position[1]), float(position[2]))


def _actuator_id_for_joint(model: Any, joint_id: int) -> int | None:
    for actuator_id in range(model.nu):
        if int(model.actuator_trnid[actuator_id][0]) == joint_id:
            return actuator_id
    return None


def _interpolate_path(path: Sequence[Sequence[float]], step_index: int, max_steps: int) -> tuple[float, ...]:
    if not path:
        raise ValueError("KUKA render path cannot be empty.")
    if len(path) == 1 or max_steps <= 1:
        return tuple(float(value) for value in path[0])
    clamped_step = min(max(step_index, 0), max_steps - 1)
    scaled = clamped_step / (max_steps - 1) * (len(path) - 1)
    low = int(math.floor(scaled))
    high = min(low + 1, len(path) - 1)
    alpha = scaled - low
    start = path[low]
    end = path[high]
    if len(start) != len(end):
        raise ValueError("KUKA render path keyframes must have consistent lengths.")
    return tuple(float(a) * (1.0 - alpha) + float(b) * alpha for a, b in zip(start, end, strict=True))


def _robot_proxy_xml() -> str:
    return """
    <body name="robot_base" pos="0 0 0.05">
      <geom name="robot_base_geom" type="cylinder" size="0.14 0.05" material="robot_orange"/>
      <body name="arm_link_1" pos="0 0 0.1">
        <joint name="joint0" type="hinge" axis="0 0 1" range="-2.7 2.7"/>
        <geom name="arm_link_1_geom" type="capsule" fromto="0 0 0 0.35 0 0.35" size="0.035" material="robot_orange"/>
        <body name="arm_link_2" pos="0.35 0 0.35">
          <joint name="joint1" type="hinge" axis="0 1 0" range="-2.0 2.0"/>
          <geom name="arm_link_2_geom" type="capsule" fromto="0 0 0 0.25 0 0.20" size="0.03" material="robot_orange"/>
          <body name="end_effector" pos="0.25 0 0.20">
            <geom name="gripper_palm" type="box" size="0.035 0.025 0.018" material="robot_dark"/>
            <geom name="gripper_left" type="box" pos="0.04 0.025 0" size="0.035 0.006 0.01" material="robot_dark"/>
            <geom name="gripper_right" type="box" pos="0.04 -0.025 0" size="0.035 0.006 0.01" material="robot_dark"/>
            <site name="attachment_site" pos="0.075 0 0" size="0.008" rgba="1 0 0 1"/>
          </body>
        </body>
      </body>
    </body>
""".rstrip()


def _shared_world_xml(
    objects: list[ObjectState],
    humans: list[HumanState],
    *,
    include_robot_proxy: bool,
    mesh_assets: MeshAssetLibrary | None = None,
) -> str:
    object_xml = "\n".join(_object_body_xml(obj, mesh_assets) for obj in objects)
    human_xml = "\n".join(_human_body_xml(human, mesh_assets) for human in humans)
    robot_xml = _robot_proxy_xml() if include_robot_proxy else ""
    return f"""
    <light name="bench_key" pos="0 -1.4 2.2" dir="0 0 -1" directional="true"/>
    <light name="bench_fill" pos="1.5 1.2 1.8" dir="-1 -1 -1" directional="true" diffuse="0.35 0.35 0.35"/>
    <geom name="floor" type="plane" size="3 3 0.05" material="floor_mat"/>
    {robot_xml}
    {object_xml}
    {human_xml}
{_fixed_camera_xml()}""".rstrip()


def _object_body_xml(obj: ObjectState, mesh_assets: MeshAssetLibrary | None = None) -> str:
    x, y, z = obj.position_m
    z = _mujoco_object_z(obj.name, z)
    mesh_spec = mesh_assets.object_spec(obj.name) if mesh_assets is not None else None
    visual_mesh = _object_visual_mesh_xml(obj.name, mesh_spec)
    if obj.name == "knife":
        handle_attrs = _visual_or_collision_attrs("knife_handle_mat", mesh_spec)
        blade_attrs = _visual_or_collision_attrs("knife_blade_mat", mesh_spec)
        return f"""
    <body name="knife" pos="{x} {y} {z}">
      <freejoint name="knife_freejoint"/>
{visual_mesh}
      <geom name="knife_handle" type="box" pos="-0.055 0 0" size="0.055 0.014 0.010" {handle_attrs} mass="0.04" friction="0.9 0.03 0.001"/>
      <geom name="knife_blade" type="box" pos="0.050 0 0" size="0.070 0.012 0.004" {blade_attrs} mass="0.05" friction="0.7 0.02 0.001"/>
      <geom name="knife_tip" type="box" pos="0.125 0 0" size="0.016 0.010 0.003" {blade_attrs} mass="0.01" friction="0.7 0.02 0.001"/>
    </body>""".rstrip()
    if obj.name == "mug":
        mug_attrs = _visual_or_collision_attrs("mug_mat", mesh_spec)
        shadow_attrs = 'rgba="0.04 0.08 0.12 1"' if mesh_spec is None else 'rgba="0 0 0 0"'
        return f"""
    <body name="mug" pos="{x} {y} {z}">
      <freejoint name="mug_freejoint"/>
{visual_mesh}
      <geom name="mug_body" type="cylinder" size="0.045 0.060" {mug_attrs} mass="0.20" friction="0.85 0.02 0.001"/>
      <geom name="mug_inner_shadow" type="cylinder" pos="0 0 0.055" size="0.035 0.006" {shadow_attrs} contype="0" conaffinity="0"/>
      <geom name="mug_handle_top" type="capsule" fromto="0.038 0 0.035 0.078 0 0.020" size="0.007" {mug_attrs} mass="0.02"/>
      <geom name="mug_handle_bottom" type="capsule" fromto="0.038 0 -0.025 0.078 0 -0.005" size="0.007" {mug_attrs} mass="0.02"/>
    </body>""".rstrip()
    if obj.name == "tennis_ball":
        ball_attrs = _visual_or_collision_attrs("tennis_ball_mat", mesh_spec)
        return f"""
    <body name="tennis_ball" pos="{x} {y} {z}">
      <freejoint name="tennis_ball_freejoint"/>
{visual_mesh}
      <geom name="tennis_ball_body" type="sphere" pos="0 0 0.035" size="0.035" {ball_attrs} mass="0.058" friction="0.9 0.03 0.001"/>
    </body>""".rstrip()
    if obj.name == "container":
        container_attrs = _visual_or_collision_attrs("container_mat", mesh_spec)
        return f"""
    <body name="container" pos="{x} {y} {z}">
      <freejoint name="container_freejoint"/>
{visual_mesh}
      <geom name="container_body" type="box" pos="0 0 0.075" size="0.080 0.060 0.075" {container_attrs} mass="0.16" friction="0.8 0.02 0.001"/>
    </body>""".rstrip()
    generic_attrs = _visual_or_collision_attrs("none", mesh_spec)
    return f"""
    <body name="{obj.name}" pos="{x} {y} {z}">
      <freejoint name="{obj.name}_freejoint"/>
{visual_mesh}
      <geom name="{obj.name}_geom" type="sphere" size="0.04" {generic_attrs} mass="0.1"/>
    </body>""".rstrip()


def _mujoco_object_z(name: str, z: float) -> float:
    if z <= 0.55:
        return z
    floor_heights = {
        "knife": 0.018,
        "mug": 0.060,
        "tennis_ball": 0.0,
        "container": 0.0,
    }
    return floor_heights.get(name, 0.04)


def _human_body_xml(human: HumanState, mesh_assets: MeshAssetLibrary | None = None) -> str:
    x, y, _ = human.position_m
    rgba = "0.85 0.1 0.08 0.78" if human.in_danger_zone else "0.1 0.55 0.25 0.78"
    mesh_spec = mesh_assets.human_spec(human) if mesh_assets is not None else None
    visual_mesh = _human_visual_mesh_xml(human, mesh_spec)
    proxy_rgba = "0 0 0 0" if mesh_spec is not None else rgba
    return f"""
    <body name="{human.id}" mocap="true" pos="{x} {y} 0.45">
{visual_mesh}
      <geom name="{human.id}_torso" type="capsule" fromto="0 0 -0.35 0 0 0.35" size="0.16" rgba="{proxy_rgba}"/>
      <geom name="{human.id}_head" type="sphere" pos="0 0 0.52" size="0.13" rgba="{proxy_rgba}"/>
    </body>""".rstrip()


def _resolve_mesh_assets(mesh_assets: MeshAssetLibrary | str | Path | None) -> MeshAssetLibrary | None:
    if isinstance(mesh_assets, MeshAssetLibrary):
        return mesh_assets
    return load_mesh_asset_library(mesh_assets)


def _mesh_asset_xml(mesh_assets: MeshAssetLibrary | None) -> str:
    return mesh_assets.mesh_asset_xml() if mesh_assets is not None else ""


def _object_visual_mesh_xml(name: str, spec) -> str:
    return spec.visual_geom_xml(f"{name}_visual_mesh") if spec is not None else ""


def _human_visual_mesh_xml(human: HumanState, spec) -> str:
    return spec.visual_geom_xml(f"{human.id}_visual_mesh") if spec is not None else ""


def _visual_or_collision_attrs(material_name: str, mesh_spec) -> str:
    if mesh_spec is not None:
        return 'rgba="0 0 0 0"'
    if material_name == "none":
        return 'rgba="0.3 0.35 0.4 1"'
    return f'material="{material_name}"'


def _fixed_camera_xml() -> str:
    return "\n".join(
        [
            _camera_xml(
                "bench_cam",
                BENCH_CAM_POS,
                BENCH_CAM_TARGET,
                fovy=68,
            ),
            _camera_xml(
                "overhead_cam",
                OVERHEAD_CAM_POS,
                OVERHEAD_CAM_TARGET,
                fovy=72,
                up=(1.0, 0.0, 0.0),
            ),
        ]
    )


def _camera_xml(
    name: str,
    pos: Vector3,
    target: Vector3,
    *,
    fovy: float,
    up: Vector3 = (0.0, 0.0, 1.0),
) -> str:
    x_axis, y_axis = _camera_xyaxes(pos, target, up)
    return (
        f'    <camera name="{name}" pos="{_vec(pos)}" '
        f'xyaxes="{_vec(x_axis)} {_vec(y_axis)}" fovy="{fovy:g}"/>'
    )


def _camera_xyaxes(pos: Vector3, target: Vector3, up: Vector3) -> tuple[Vector3, Vector3]:
    forward = _normalize((target[0] - pos[0], target[1] - pos[1], target[2] - pos[2]))
    camera_z = (-forward[0], -forward[1], -forward[2])
    x_axis = _cross(up, camera_z)
    if _norm(x_axis) < 1e-8:
        x_axis = _cross((0.0, 1.0, 0.0), camera_z)
    x_axis = _normalize(x_axis)
    y_axis = _normalize(_cross(camera_z, x_axis))
    return x_axis, y_axis


def _normalize(value: Vector3) -> Vector3:
    norm = _norm(value)
    if norm < 1e-12:
        raise ValueError(f"Cannot normalize zero vector {value!r}")
    return (value[0] / norm, value[1] / norm, value[2] / norm)


def _norm(value: Vector3) -> float:
    return math.sqrt(value[0] ** 2 + value[1] ** 2 + value[2] ** 2)


def _cross(a: Vector3, b: Vector3) -> Vector3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _vec(value: Vector3) -> str:
    return f"{value[0]:.9g} {value[1]:.9g} {value[2]:.9g}"


@dataclass(frozen=True)
class MenagerieKukaAssets:
    root: Path

    @property
    def scene_xml(self) -> Path:
        return self.root / "kuka_iiwa_14" / "scene.xml"

    @property
    def iiwa_xml(self) -> Path:
        return self.root / "kuka_iiwa_14" / "iiwa14.xml"

    @property
    def available(self) -> bool:
        return self.scene_xml.exists() and self.iiwa_xml.exists()

    def compile_scene(self) -> Any:
        if not self.available:
            raise FileNotFoundError(f"KUKA assets are missing under {self.root}")
        import mujoco

        return mujoco.MjModel.from_xml_path(str(self.scene_xml))
