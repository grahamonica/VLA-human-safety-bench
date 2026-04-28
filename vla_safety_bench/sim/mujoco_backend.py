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
KUKA_HOME_JOINT_POSITIONS = dict(
    zip(
        KUKA_JOINT_NAMES,
        (0.0, 0.62, 0.0, -1.45, 0.0, 1.12, 0.0),
        strict=True,
    )
)
KUKA_DEFAULT_PHYSICS_SUBSTEPS = 50


def mujoco_available() -> bool:
    try:
        import mujoco  # noqa: F401
    except Exception:
        return False
    return True


def kuka_scenario_mujoco_xml(
    *,
    objects: list[ObjectState],
    humans: list[HumanState],
    menagerie_root: Path | None = None,
    mesh_assets: MeshAssetLibrary | str | Path | None = None,
) -> str:
    """Build a benchmark scene around the real Menagerie KUKA iiwa 14 MJCF."""

    mesh_library = _resolve_mesh_assets(mesh_assets)
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
    scene_xml = _shared_world_xml(objects, humans, mesh_assets=mesh_library)
    xml = xml.replace("  </worldbody>", scene_xml + "\n  </worldbody>", 1)
    xml = xml.replace('mujoco model="iiwa14"', 'mujoco model="vla_human_safety_kuka"')
    return xml


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


def solve_site_position_ik(
    model: Any,
    data: Any,
    *,
    site_name: str,
    target_position: Vector3,
    joint_names: Sequence[str] = KUKA_JOINT_NAMES,
    max_iterations: int = 80,
    tolerance_m: float = 0.015,
    damping: float = 1e-3,
    step_limit_rad: float = 0.18,
) -> dict[str, float]:
    """Resolve a Cartesian site target into KUKA joint targets using MuJoCo Jacobians."""

    import mujoco
    import numpy as np

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise ValueError(f"MuJoCo model is missing site {site_name!r}.")

    joint_ids: list[int] = []
    qpos_adrs: list[int] = []
    dof_adrs: list[int] = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo model is missing joint {name!r}.")
        joint_ids.append(joint_id)
        qpos_adrs.append(int(model.jnt_qposadr[joint_id]))
        dof_adrs.append(int(model.jnt_dofadr[joint_id]))

    original_qpos = data.qpos.copy()
    original_qvel = data.qvel.copy()
    target = np.asarray(target_position, dtype=float)
    qpos = np.asarray([float(data.qpos[adr]) for adr in qpos_adrs], dtype=float)
    best_qpos = qpos.copy()
    best_error = float("inf")

    try:
        for _ in range(max_iterations):
            for value, adr in zip(qpos, qpos_adrs, strict=True):
                data.qpos[adr] = float(value)
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)

            error = target - data.site_xpos[site_id]
            error_norm = float(np.linalg.norm(error))
            if error_norm < best_error:
                best_error = error_norm
                best_qpos = qpos.copy()
            if error_norm <= tolerance_m:
                return {
                    name: float(value)
                    for name, value in zip(joint_names, qpos, strict=True)
                }

            jacp = np.zeros((3, model.nv), dtype=float)
            jacr = np.zeros((3, model.nv), dtype=float)
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
            jac = jacp[:, dof_adrs]
            system = jac @ jac.T + (damping * damping) * np.eye(3)
            dq = jac.T @ np.linalg.solve(system, error)
            step_norm = float(np.linalg.norm(dq))
            if step_norm > step_limit_rad:
                dq *= step_limit_rad / step_norm
            qpos = qpos + dq

            for index, joint_id in enumerate(joint_ids):
                if model.jnt_limited[joint_id]:
                    low, high = model.jnt_range[joint_id]
                    qpos[index] = min(max(float(qpos[index]), float(low)), float(high))
    finally:
        data.qpos[:] = original_qpos
        data.qvel[:] = original_qvel
        mujoco.mj_forward(model, data)

    if best_error <= tolerance_m * 3.0:
        return {
            name: float(value)
            for name, value in zip(joint_names, best_qpos, strict=True)
        }
    raise RuntimeError(
        f"IK could not reach {site_name!r} target {list(target_position)} within tolerance; "
        f"best error was {best_error:.4f} m."
    )


def _actuator_id_for_joint(model: Any, joint_id: int) -> int | None:
    for actuator_id in range(model.nu):
        if int(model.actuator_trnid[actuator_id][0]) == joint_id:
            return actuator_id
    return None


def _shared_world_xml(
    objects: list[ObjectState],
    humans: list[HumanState],
    *,
    mesh_assets: MeshAssetLibrary,
) -> str:
    object_xml = "\n".join(_object_body_xml(obj, mesh_assets) for obj in objects)
    human_xml = "\n".join(_human_body_xml(human, mesh_assets) for human in humans)
    return f"""
    <light name="bench_key" pos="0 -1.4 2.2" dir="0 0 -1" directional="true"/>
    <light name="bench_fill" pos="1.5 1.2 1.8" dir="-1 -1 -1" directional="true" diffuse="0.35 0.35 0.35"/>
    <geom name="floor" type="plane" size="3 3 0.05" material="floor_mat"/>
    {object_xml}
    {human_xml}
{_fixed_camera_xml()}""".rstrip()


def _object_body_xml(obj: ObjectState, mesh_assets: MeshAssetLibrary) -> str:
    x, y, z = obj.position_m
    z = _mujoco_object_z(obj.name, z)
    mesh_spec = mesh_assets.object_spec(obj.name)
    if mesh_spec is None:
        raise ValueError(f"Missing required mesh asset for object {obj.name!r}.")
    mesh = mesh_spec.geom_xml(
        f"{obj.name}_mesh",
        mass=_object_mass_kg(obj.name),
        friction="0.85 0.02 0.001",
    )
    return f"""
    <body name="{obj.name}" pos="{x} {y} {z}">
      <freejoint name="{obj.name}_freejoint"/>
{mesh}
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


def _object_mass_kg(name: str) -> float:
    masses = {
        "knife": 0.10,
        "mug": 0.24,
        "tennis_ball": 0.058,
        "container": 0.16,
    }
    return masses.get(name, 0.10)


def _human_body_xml(human: HumanState, mesh_assets: MeshAssetLibrary) -> str:
    x, y, _ = human.position_m
    mesh_spec = mesh_assets.human_spec(human)
    if mesh_spec is None:
        raise ValueError(f"Missing required mesh asset for human {human.id!r}.")
    mesh = mesh_spec.geom_xml(f"{human.id}_mesh")
    return f"""
    <body name="{human.id}" mocap="true" pos="{x} {y} 0.45">
{mesh}
    </body>""".rstrip()


def _resolve_mesh_assets(mesh_assets: MeshAssetLibrary | str | Path | None) -> MeshAssetLibrary:
    if isinstance(mesh_assets, MeshAssetLibrary):
        return mesh_assets
    return load_mesh_asset_library(mesh_assets, required=True)


def _mesh_asset_xml(mesh_assets: MeshAssetLibrary) -> str:
    return mesh_assets.mesh_asset_xml()


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
