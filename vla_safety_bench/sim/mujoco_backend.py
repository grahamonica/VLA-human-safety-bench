from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from copy import deepcopy
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
KUKA_GRIPPER_ACTUATOR = "robotiq_fingers_actuator"
KUKA_TOOL_SITE_NAME = "gripper_tcp_site"
# Home joints keep the flange above the bench with a forward/down approach so
# the mounted 2F-85 gripper and wrist camera start with workspace visibility.
KUKA_HOME_JOINT_POSITIONS = dict(
    zip(
        KUKA_JOINT_NAMES,
        (0.0, 0.30, 0.0, -1.20, 0.0, 1.30, 0.0),
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
    iiwa_assets_dir = kuka_dir / "assets"
    gripper_dir = root / "robotiq_2f85_v4"
    gripper_xml = gripper_dir / "2f85.xml"
    gripper_assets_dir = gripper_dir / "assets"
    if not iiwa_xml.exists():
        raise FileNotFoundError(
            f"Missing KUKA Menagerie model at {iiwa_xml}. Run scripts/fetch_mujoco_kuka.py first."
        )
    if not gripper_xml.exists():
        raise FileNotFoundError(
            f"Missing Robotiq gripper model at {gripper_xml}. "
            "Run scripts/fetch_mujoco_kuka.py first."
        )

    model_root = ET.fromstring(iiwa_xml.read_text(encoding="utf-8"))
    compiler = model_root.find("compiler")
    if compiler is None:
        raise ValueError("KUKA Menagerie model is missing <compiler>.")
    compiler.set("meshdir", str(iiwa_assets_dir))

    asset = model_root.find("asset")
    if asset is None:
        raise ValueError("KUKA Menagerie model is missing <asset>.")
    asset.append(_element("material", {"name": "floor_mat", "rgba": "0.75 0.77 0.78 1"}))
    _append_mesh_assets(asset, mesh_library)

    _inject_robotiq_gripper(
        model_root=model_root,
        gripper_xml_path=gripper_xml,
        gripper_assets_dir=gripper_assets_dir,
    )
    _attach_wrist_camera(model_root)
    _append_scene_worldbody(model_root, objects=objects, humans=humans, mesh_assets=mesh_library)
    model_root.set("model", "vla_human_safety_kuka")
    return ET.tostring(model_root, encoding="unicode")


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


def set_gripper_control(model: Any, data: Any, value: float) -> float:
    import mujoco

    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, KUKA_GRIPPER_ACTUATOR)
    if actuator_id < 0:
        raise ValueError(f"MuJoCo model is missing actuator {KUKA_GRIPPER_ACTUATOR!r}.")
    target = float(value)
    if model.actuator_ctrllimited[actuator_id]:
        low, high = model.actuator_ctrlrange[actuator_id]
        target = min(max(target, float(low)), float(high))
    data.ctrl[actuator_id] = target
    return target


def read_gripper_control(model: Any, data: Any) -> float | None:
    import mujoco

    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, KUKA_GRIPPER_ACTUATOR)
    if actuator_id < 0:
        return None
    return float(data.ctrl[actuator_id])


def _inject_robotiq_gripper(*, model_root: ET.Element, gripper_xml_path: Path, gripper_assets_dir: Path) -> None:
    gripper_root = ET.fromstring(gripper_xml_path.read_text(encoding="utf-8"))
    _prepare_prefixed_gripper(gripper_root, prefix="robotiq_", gripper_assets_dir=gripper_assets_dir)

    asset = model_root.find("asset")
    if asset is None:
        raise ValueError("KUKA Menagerie model is missing <asset>.")
    gripper_asset = gripper_root.find("asset")
    if gripper_asset is not None:
        for child in gripper_asset:
            asset.append(deepcopy(child))

    for gripper_default in gripper_root.findall("default"):
        model_root.append(deepcopy(gripper_default))

    contact = model_root.find("contact")
    if contact is None:
        contact = ET.SubElement(model_root, "contact")
    gripper_contact = gripper_root.find("contact")
    if gripper_contact is not None:
        for child in gripper_contact:
            contact.append(deepcopy(child))

    tendon = model_root.find("tendon")
    if tendon is None:
        tendon = ET.SubElement(model_root, "tendon")
    gripper_tendon = gripper_root.find("tendon")
    if gripper_tendon is not None:
        for child in gripper_tendon:
            tendon.append(deepcopy(child))

    equality = model_root.find("equality")
    if equality is None:
        equality = ET.SubElement(model_root, "equality")
    gripper_equality = gripper_root.find("equality")
    if gripper_equality is not None:
        for child in gripper_equality:
            equality.append(deepcopy(child))

    actuator = model_root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(model_root, "actuator")
    gripper_actuator = gripper_root.find("actuator")
    if gripper_actuator is not None:
        for child in gripper_actuator:
            actuator.append(deepcopy(child))

    worldbody = model_root.find("worldbody")
    if worldbody is None:
        raise ValueError("KUKA Menagerie model is missing <worldbody>.")
    link7 = _find_body_by_name(worldbody, "link7")
    if link7 is None:
        raise ValueError("KUKA Menagerie model is missing body 'link7'.")
    gripper_body = _find_body_by_name(gripper_root.find("worldbody"), "robotiq_base")
    if gripper_body is None:
        raise ValueError("Robotiq model is missing body 'base'.")
    gripper_mount = deepcopy(gripper_body)
    # The Menagerie gripper points in +Z of its base frame. KUKA's attachment
    # site already orients +Z toward the workspace at home, so only a 45 deg
    # roll is needed to align finger closure with world Y.
    gripper_mount.set("pos", "0 0 0.045")
    gripper_mount.set("euler", "0 0 0.7853981634")
    link7.append(gripper_mount)
    link7.append(_element("site", {"name": KUKA_TOOL_SITE_NAME, "pos": "0 0 0.215"}))


def _prepare_prefixed_gripper(gripper_root: ET.Element, *, prefix: str, gripper_assets_dir: Path) -> None:
    gripper_asset = gripper_root.find("asset")
    if gripper_asset is None:
        raise ValueError("Robotiq model is missing <asset>.")

    for mesh in gripper_asset.findall("mesh"):
        file_name = mesh.get("file")
        if file_name is None:
            continue
        mesh_name = mesh.get("name") or Path(file_name).stem
        mesh.set("name", mesh_name)
        mesh.set("file", str((gripper_assets_dir / Path(file_name).name).resolve()))

    name_attrs = {
        "name",
        "class",
        "childclass",
        "material",
        "mesh",
        "texture",
        "joint",
        "joint1",
        "joint2",
        "tendon",
        "body",
        "body1",
        "body2",
    }
    for element in gripper_root.iter():
        for attr in name_attrs:
            value = element.get(attr)
            if value is None:
                continue
            element.set(attr, f"{prefix}{value}")


def _attach_wrist_camera(model_root: ET.Element) -> None:
    worldbody = model_root.find("worldbody")
    if worldbody is None:
        raise ValueError("KUKA Menagerie model is missing <worldbody>.")
    link7 = _find_body_by_name(worldbody, "link7")
    if link7 is None:
        raise ValueError("KUKA Menagerie model is missing body 'link7'.")

    mount = _element("body", {"name": "wrist_camera_mount", "pos": "0 0 0.17"})
    # Collidable camera housing so the wrist camera cannot clip through
    # scene objects independently of the arm links.
    mount.append(
        _element(
            "geom",
            {
                "name": "wrist_camera_body",
                "type": "box",
                "size": "0.018 0.016 0.016",
                "rgba": "0.03 0.03 0.03 1",
                "mass": "0.02",
                "friction": "0.8 0.02 0.001",
            },
        )
    )
    # The camera sits behind the fingers and looks along the approach vector so
    # both pads and object contacts remain visible.
    mount.append(
        _element(
            "camera",
            {"name": "wrist_cam", "pos": "0 -0.03 0.02", "euler": "0.3 3.14159265359 0", "fovy": "110"},
        )
    )
    link7.append(mount)


def _append_mesh_assets(asset: ET.Element, mesh_assets: MeshAssetLibrary) -> None:
    xml = _mesh_asset_xml(mesh_assets).strip()
    if not xml:
        return
    container = ET.fromstring(f"<asset_fragment>{xml}</asset_fragment>")
    for child in container:
        asset.append(child)


def _append_scene_worldbody(
    model_root: ET.Element,
    *,
    objects: list[ObjectState],
    humans: list[HumanState],
    mesh_assets: MeshAssetLibrary,
) -> None:
    worldbody = model_root.find("worldbody")
    if worldbody is None:
        raise ValueError("KUKA Menagerie model is missing <worldbody>.")
    shared = _shared_world_xml(objects, humans, mesh_assets=mesh_assets).strip()
    fragment = ET.fromstring(f"<world_fragment>{shared}</world_fragment>")
    for child in fragment:
        worldbody.append(child)


def _find_body_by_name(root: ET.Element | None, body_name: str) -> ET.Element | None:
    if root is None:
        return None
    for body in root.findall(".//body"):
        if body.get("name") == body_name:
            return body
    return None


def _element(tag: str, attrib: Mapping[str, str]) -> ET.Element:
    element = ET.Element(tag)
    for key, value in attrib.items():
        element.set(key, value)
    return element


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
    visual_mesh = mesh_spec.geom_xml(
        f"{obj.name}_mesh",
        visual_only=True,
        mass=_object_mass_kg(obj.name),
        friction="0.85 0.02 0.001",
    )
    collision_geoms = "\n".join(_object_collision_geoms_xml(obj.name))
    return f"""
    <body name="{obj.name}" pos="{x} {y} {z}">
      <freejoint name="{obj.name}_freejoint"/>
{visual_mesh}
{collision_geoms}
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


def _object_collision_geoms_xml(name: str) -> list[str]:
    # Non-convex imported meshes can have weak contacts. Keep visual meshes for
    # rendering but use stable convex collision proxies so objects cannot be
    # tunneled through by links, gripper, or wrist camera housing.
    if name == "knife":
        return [
            '      <geom name="knife_collision" type="capsule" size="0.012 0.14" '
            'pos="0 0 0" euler="1.57079632679 0 0" density="0" friction="0.9 0.02 0.001"/>'
        ]
    if name == "mug":
        return [
            '      <geom name="mug_collision" type="cylinder" size="0.045 0.032" '
            'pos="0 0.045 0" euler="1.57079632679 0 0" density="0" friction="0.9 0.02 0.001"/>'
        ]
    if name == "container":
        return [
            '      <geom name="container_collision" type="box" size="0.095 0.075 0.065" '
            'pos="0 0.074 0" euler="1.57079632679 0 0" density="0" friction="0.85 0.02 0.001"/>'
        ]
    if name == "tennis_ball":
        return [
            '      <geom name="tennis_ball_collision" type="sphere" size="0.03" '
            'pos="0 0 0.032" density="0" friction="0.9 0.02 0.001"/>'
        ]
    return [
        '      <geom name="object_collision" type="sphere" size="0.04" '
        'density="0" friction="0.85 0.02 0.001"/>'
    ]


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
    def gripper_xml(self) -> Path:
        return self.root / "robotiq_2f85_v4" / "2f85.xml"

    @property
    def available(self) -> bool:
        return self.scene_xml.exists() and self.iiwa_xml.exists() and self.gripper_xml.exists()

    def compile_scene(self) -> Any:
        if not self.available:
            raise FileNotFoundError(f"KUKA assets are missing under {self.root}")
        import mujoco

        return mujoco.MjModel.from_xml_path(str(self.scene_xml))
