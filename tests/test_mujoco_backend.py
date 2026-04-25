from pathlib import Path
import re

import pytest

from vla_safety_bench.scenarios import load_scenario_set
from vla_safety_bench.sim.mujoco_backend import (
    compile_minimal_scene,
    compile_kuka_scene,
    compile_scenario_scene,
    MenagerieKukaAssets,
    apply_joint_positions,
    kuka_render_joint_positions,
    mujoco_available,
    render_scene_png,
    scenario_mujoco_xml,
)
from vla_safety_bench.sim.mesh_assets import load_mesh_asset_library


def test_minimal_mujoco_scene_compiles_when_available():
    if not mujoco_available():
        pytest.skip("mujoco is not installed")
    model = compile_minimal_scene(human_in_danger_zone=True)
    assert model.nbody >= 9
    assert model.ngeom >= 12


def test_mujoco_scene_contains_benchmark_objects_when_available():
    if not mujoco_available():
        pytest.skip("mujoco is not installed")
    import mujoco

    model = compile_minimal_scene()
    for body_name in ["knife", "mug", "tennis_ball", "container", "human_0", "robot_base"]:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name) >= 0
    for geom_name in ["knife_blade", "knife_handle", "mug_body", "tennis_ball_body", "container_body", "floor"]:
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name) >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "bench_cam") >= 0


def test_mujoco_masked_scenario_scene_renders_when_available(tmp_path):
    if not mujoco_available():
        pytest.skip("mujoco is not installed")
    scenario_set = load_scenario_set("configs/benchmark.json")
    scenario = next(item for item in scenario_set.scenarios if item.id == "masked_toss_knife_subtask")
    model = compile_scenario_scene(scenario)
    frame = render_scene_png(model, tmp_path / "frame.png")
    assert frame.endswith("frame.png")
    assert (tmp_path / "frame.png").stat().st_size > 0


def test_fixed_cameras_do_not_change_across_steps():
    scenario = load_scenario_set("configs/smoke.json").scenarios[1]
    step_0 = scenario_mujoco_xml(objects=list(scenario.objects), humans=scenario.humans_at(0))
    step_1 = scenario_mujoco_xml(objects=list(scenario.objects), humans=scenario.humans_at(1))
    camera_pattern = re.compile(r'<camera name="(?:bench_cam|overhead_cam)"[^>]+>')
    assert camera_pattern.findall(step_0) == camera_pattern.findall(step_1)


def test_mujoco_scene_uses_manifest_mesh_when_available(tmp_path):
    if not mujoco_available():
        pytest.skip("mujoco is not installed")
    import json
    import mujoco
    from PIL import Image

    mesh_root = tmp_path / "meshes"
    mesh_root.mkdir()
    _write_tetra_obj(mesh_root / "knife.obj")
    Image.new("RGB", (2, 2), (220, 20, 20)).save(mesh_root / "knife_basecolor.png")
    manifest = tmp_path / "mesh_assets.json"
    manifest.write_text(
        json.dumps(
            {
                "root": str(mesh_root),
                "strict": False,
                "assets": {
                    "knife": {
                        "file": "knife.obj",
                        "texture": "knife_basecolor.png",
                        "scale": [0.2, 0.03, 0.01],
                        "rgba": "0.8 0.8 0.82 1",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    scenario = load_scenario_set("configs/smoke.json").scenarios[0]
    model = compile_scenario_scene(scenario, mesh_assets=manifest)
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "mesh_knife") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, "tex_knife_basecolor") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "mat_knife_visual") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "knife_visual_mesh") >= 0


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
    with pytest.raises(ValueError, match="missing scene assets"):
        library.validate_scene(scenario.objects, scenario.humans)


def test_kuka_scene_uses_real_menagerie_model_when_assets_available(tmp_path):
    if not mujoco_available():
        pytest.skip("mujoco is not installed")
    if not MenagerieKukaAssets(Path("third_party/mujoco_menagerie")).available:
        pytest.skip("KUKA Menagerie assets are not fetched")
    import mujoco

    scenario = load_scenario_set("configs/smoke.json").scenarios[0]
    model = compile_kuka_scene(scenario)
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist_camera_mount") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "knife") >= 0
    frame = render_scene_png(model, tmp_path / "kuka_wrist.png", camera="wrist_cam")
    assert frame.endswith("kuka_wrist.png")
    assert (tmp_path / "kuka_wrist.png").stat().st_size > 0


def test_kuka_wrist_camera_points_toward_workspace_object_when_assets_available():
    if not mujoco_available():
        pytest.skip("mujoco is not installed")
    if not MenagerieKukaAssets(Path("third_party/mujoco_menagerie")).available:
        pytest.skip("KUKA Menagerie assets are not fetched")
    import mujoco

    scenario = next(
        item for item in load_scenario_set("configs/smoke.json").scenarios if item.id == "human_enters_zone_midtask"
    )
    step_index = 1
    model = compile_kuka_scene(scenario, step_index=step_index)
    data = mujoco.MjData(model)
    apply_joint_positions(model, data, kuka_render_joint_positions(step_index, scenario.max_steps))
    mujoco.mj_forward(model, data)

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    mug_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mug")
    vector_to_mug = data.xpos[mug_id] - data.cam_xpos[camera_id]
    vector_to_mug = vector_to_mug / (vector_to_mug @ vector_to_mug) ** 0.5
    camera_z_axis = data.cam_xmat[camera_id].reshape(3, 3)[2]
    wrist_optical_axis = -camera_z_axis

    assert wrist_optical_axis @ vector_to_mug > 0.9


def test_kuka_render_path_updates_joint_positions_when_assets_available():
    if not mujoco_available():
        pytest.skip("mujoco is not installed")
    if not MenagerieKukaAssets(Path("third_party/mujoco_menagerie")).available:
        pytest.skip("KUKA Menagerie assets are not fetched")
    import mujoco

    scenario = load_scenario_set("configs/smoke.json").scenarios[1]
    model = compile_kuka_scene(scenario)
    data = mujoco.MjData(model)
    start = kuka_render_joint_positions(0, scenario.max_steps)
    end = kuka_render_joint_positions(scenario.max_steps - 1, scenario.max_steps)
    apply_joint_positions(model, data, start)
    joint1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint1")
    joint1_qpos = model.jnt_qposadr[joint1_id]
    assert data.qpos[joint1_qpos] == pytest.approx(start["joint1"])
    assert start["joint1"] != pytest.approx(end["joint1"])


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
