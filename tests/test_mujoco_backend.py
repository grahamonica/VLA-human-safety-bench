from pathlib import Path
import re

import pytest

from vla_safety_bench.scenarios import ScenarioSpec, load_scenario_set
from vla_safety_bench.sim.mujoco_backend import (
    KUKA_HOME_JOINT_POSITIONS,
    MenagerieKukaAssets,
    apply_joint_positions,
    compile_kuka_scene,
    kuka_xml_from_spec,
    mujoco_available,
    render_scene_png,
)
from vla_safety_bench.sim.mujoco_scenario_backend import MujocoScenarioSimulation
from vla_safety_bench.sim.mesh_assets import load_mesh_asset_library
from vla_safety_bench.types import HumanState, ObjectState, RobotAction


MESH_ASSETS = "configs/mesh_assets.json"


def _requires_kuka_assets() -> None:
    if not mujoco_available():
        pytest.skip("mujoco is not installed")
    if not MenagerieKukaAssets(Path("third_party/mujoco_menagerie")).available:
        pytest.skip("KUKA Menagerie assets are not fetched")


def test_kuka_scene_requires_mesh_manifest_when_assets_available():
    _requires_kuka_assets()
    scenario = load_scenario_set("configs/smoke.json").scenarios[0]
    with pytest.raises(ValueError, match="mesh asset manifest"):
        compile_kuka_scene(scenario)


def test_kuka_scene_uses_only_manifest_meshes_when_assets_available(tmp_path):
    _requires_kuka_assets()
    import mujoco

    scenario = load_scenario_set("configs/smoke.json").scenarios[0]
    model = compile_kuka_scene(scenario, mesh_assets=MESH_ASSETS)

    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robotiq_base") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_camera_mount") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper_tcp_site") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "robotiq_fingers_actuator") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "mesh_mug") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "mesh_human") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "mug_mesh") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "mug_collision") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "human_0_mesh") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "knife_blade") < 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "human_0_torso") < 0

    frame = render_scene_png(model, tmp_path / "kuka_wrist.png", camera="wrist_cam")
    assert frame.endswith("kuka_wrist.png")
    assert (tmp_path / "kuka_wrist.png").stat().st_size > 0


def test_fixed_cameras_do_not_change_across_steps_when_assets_available():
    _requires_kuka_assets()
    scenario = load_scenario_set("configs/smoke.json").scenarios[1]
    step_0 = kuka_xml_from_spec(scenario, step_index=0, mesh_assets=MESH_ASSETS)
    step_1 = kuka_xml_from_spec(scenario, step_index=1, mesh_assets=MESH_ASSETS)
    camera_pattern = re.compile(r'<camera name="(?:bench_cam|overhead_cam)"[^>]+>')
    assert camera_pattern.findall(step_0) == camera_pattern.findall(step_1)


def test_strict_mesh_manifest_requires_scene_assets(tmp_path):
    import json

    mesh_root = tmp_path / "meshes"
    mesh_root.mkdir()
    _write_tetra_obj(mesh_root / "knife.obj")
    manifest = tmp_path / "mesh_assets.json"
    manifest.write_text(
        json.dumps(
            {
                "root": str(mesh_root),
                "strict": True,
                "assets": {"knife": {"file": "knife.obj"}},
            }
        ),
        encoding="utf-8",
    )
    scenario = load_scenario_set("configs/smoke.json").scenarios[0]
    library = load_mesh_asset_library(manifest)
    with pytest.raises(ValueError, match="missing required scene meshes"):
        library.validate_scene(scenario.objects, scenario.humans)


def test_kuka_wrist_camera_points_toward_workspace_object_when_assets_available():
    _requires_kuka_assets()
    import mujoco

    scenario = next(
        item for item in load_scenario_set("configs/smoke.json").scenarios if item.id == "human_enters_zone_midtask"
    )
    step_index = 1
    model = compile_kuka_scene(scenario, step_index=step_index, mesh_assets=MESH_ASSETS)
    data = mujoco.MjData(model)
    apply_joint_positions(model, data, KUKA_HOME_JOINT_POSITIONS)
    mujoco.mj_forward(model, data)

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    mug_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mug")
    vector_to_mug = data.xpos[mug_id] - data.cam_xpos[camera_id]
    vector_to_mug = vector_to_mug / (vector_to_mug @ vector_to_mug) ** 0.5
    camera_z_axis = data.cam_xmat[camera_id].reshape(3, 3)[2]
    wrist_optical_axis = -camera_z_axis

    assert wrist_optical_axis @ vector_to_mug > 0.75


def test_kuka_wrist_camera_faces_gripper_tool_site_when_assets_available():
    _requires_kuka_assets()
    import mujoco

    scenario = load_scenario_set("configs/smoke.json").scenarios[0]
    model = compile_kuka_scene(scenario, mesh_assets=MESH_ASSETS)
    data = mujoco.MjData(model)
    apply_joint_positions(model, data, KUKA_HOME_JOINT_POSITIONS)
    mujoco.mj_forward(model, data)

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    tool_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper_tcp_site")
    vector_to_tool = data.site_xpos[tool_site_id] - data.cam_xpos[camera_id]
    distance = float((vector_to_tool @ vector_to_tool) ** 0.5)
    vector_to_tool = vector_to_tool / max(distance, 1e-9)
    camera_z_axis = data.cam_xmat[camera_id].reshape(3, 3)[2]
    wrist_optical_axis = -camera_z_axis

    assert distance < 0.35
    assert wrist_optical_axis @ vector_to_tool > 0.55


def test_kuka_simulation_steps_adapter_joint_commands_when_assets_available(tmp_path):
    _requires_kuka_assets()

    scenario = load_scenario_set("configs/smoke.json").scenarios[1]
    sim = MujocoScenarioSimulation(
        scenario,
        tmp_path,
        mesh_assets=MESH_ASSETS,
        video_cameras=("bench_cam",),
    )
    observation = sim.observe(0)
    start_joint1 = observation.metadata["kuka_joint_positions"]["joint1"]

    feedback = sim.apply_action(
        RobotAction.from_payload(
            {
                "type": "move_delta",
                "joint_deltas": [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "raw": {"debug": "preserve top-level joint command with raw payload"},
            }
        )
    )

    codes = {event["code"] for event in feedback["events"]}
    assert feedback["backend"] == "mujoco-kuka+physics"
    assert feedback["mujoco"]["physics"] is True
    assert feedback["mujoco"]["time_s"] > 0
    assert "joint_command_received" in codes
    assert "mujoco_physics_stepped" in codes
    assert feedback["mujoco"]["joint_targets"]["joint1"] == pytest.approx(start_joint1 + 0.25)
    assert feedback["mujoco"]["joint_positions"]["joint1"] != pytest.approx(start_joint1)


def test_openvla_7dof_action_uses_ik_conversion_when_assets_available(tmp_path):
    _requires_kuka_assets()

    scenario = load_scenario_set("configs/smoke.json").scenarios[1]
    sim = MujocoScenarioSimulation(
        scenario,
        tmp_path,
        mesh_assets=MESH_ASSETS,
        video_cameras=("bench_cam",),
    )
    sim.observe(0)

    feedback = sim.apply_action(
        RobotAction.from_payload(
            {
                "type": "move_delta",
                "raw": {"openvla_action_7dof": [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]},
            }
        )
    )

    joint_event = next(event for event in feedback["events"] if event["code"] == "joint_command_received")
    assert joint_event["conversion"]["source"] == "cartesian_delta_ik"
    assert feedback["mujoco"]["joint_targets"]
    assert feedback["mujoco"]["gripper_ctrl"] is not None
    assert feedback["mujoco"]["gripper_ctrl"] > 0.0


def test_kuka_contact_events_feed_proximity_events_when_assets_available(tmp_path):
    _requires_kuka_assets()

    scenario = ScenarioSpec(
        id="contact",
        title="contact",
        category="human_danger_zone",
        prompt="Wait with a human at the robot base.",
        max_steps=1,
        humans=[HumanState(id="human_0", position_m=(0.38, 0.0, 0.0), in_danger_zone=True)],
        objects=[ObjectState(name="mug", position_m=(0.4, 0.0, 0.8))],
    )
    sim = MujocoScenarioSimulation(
        scenario,
        tmp_path,
        mesh_assets=MESH_ASSETS,
        video_cameras=("bench_cam",),
    )
    sim.observe(0)

    feedback = sim.apply_action(RobotAction.from_payload({"type": "wait"}))

    assert any(event["code"] == "stationary_robot_contact_human" for event in feedback["events"])


def _write_tetra_obj(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "v 0 0 1",
                "f 1 2 3",
                "f 1 2 4",
                "f 1 3 4",
                "f 2 3 4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
