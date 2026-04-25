from pathlib import Path

import pytest

from vla_safety_bench.adapters.base import load_adapter
from vla_safety_bench.harness import BenchmarkHarness
from vla_safety_bench.hardware.hardware_io import (
    CameraIntrinsics,
    CameraPose,
    MockHardwareIO,
)
from vla_safety_bench.hardware.injection_backend import HardwareInjectionSimulation
from vla_safety_bench.hardware.pixel_injection import (
    composite_overlay,
    render_human_overlay,
)
from vla_safety_bench.scenarios import load_scenario_set
from vla_safety_bench.sim.mujoco_backend import mujoco_available
from vla_safety_bench.types import HumanState, RobotAction


def test_composite_overlay_replaces_only_alpha_pixels():
    import numpy as np

    real = np.zeros((4, 4, 3), dtype=np.uint8)
    real[:, :] = (200, 50, 50)
    overlay = np.zeros((4, 4, 3), dtype=np.uint8)
    overlay[:, :] = (10, 220, 10)
    alpha = np.zeros((4, 4), dtype=np.float32)
    alpha[1:3, 1:3] = 1.0

    blended = composite_overlay(real, overlay, alpha)

    assert tuple(blended[0, 0]) == (200, 50, 50)
    assert tuple(blended[2, 2]) == (10, 220, 10)


def test_composite_overlay_supports_partial_alpha():
    import numpy as np

    real = np.full((2, 2, 3), 100, dtype=np.uint8)
    overlay = np.full((2, 2, 3), 200, dtype=np.uint8)
    alpha = np.full((2, 2), 0.5, dtype=np.float32)

    blended = composite_overlay(real, overlay, alpha)
    assert blended.dtype.name == "uint8"
    assert (blended == 150).all()


def test_composite_overlay_rejects_mismatched_shapes():
    import numpy as np

    real = np.zeros((4, 4, 3), dtype=np.uint8)
    overlay = np.zeros((3, 3, 3), dtype=np.uint8)
    alpha = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="match overlay"):
        composite_overlay(real, overlay, alpha)


def test_camera_pose_axes_unpack_columns_correctly():
    pose = CameraPose(
        position_m=(0.0, 0.0, 0.0),
        rotation_matrix=(
            (0.0, 0.0, 1.0),
            (-1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
        ),
    )
    assert pose.x_axis() == (0.0, -1.0, 0.0)
    assert pose.y_axis() == (0.0, 0.0, -1.0)


def test_render_human_overlay_returns_empty_alpha_for_no_humans():
    intrinsics = CameraIntrinsics(width=32, height=24, fovy_deg=92.0)
    pose = CameraPose(position_m=(0.0, 0.0, 0.5), rotation_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    rgb, alpha = render_human_overlay(pose, intrinsics, [])
    assert rgb.shape == (24, 32, 3)
    assert alpha.shape == (24, 32)
    assert (alpha == 0).all()


def test_render_human_overlay_marks_human_pixels_when_mujoco_available():
    if not mujoco_available():
        pytest.skip("mujoco is not installed")

    intrinsics = CameraIntrinsics(width=128, height=96, fovy_deg=92.0)
    pose = CameraPose(
        position_m=(0.30, 0.0, 0.55),
        rotation_matrix=(
            (0.0, 0.0, -1.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
    )
    humans = [HumanState(id="human_0", position_m=(1.50, 0.0, 0.0))]

    rgb, alpha = render_human_overlay(pose, intrinsics, humans)
    assert rgb.shape == (96, 128, 3)
    assert alpha.shape == (96, 128)
    # The proxy human is in front of the cam; some pixels must be flagged as human.
    assert alpha.sum() > 0
    # And not the entire frame should be human.
    assert alpha.sum() < alpha.size


def test_hardware_injection_simulation_writes_composited_frame_when_mujoco_available(tmp_path):
    if not mujoco_available():
        pytest.skip("mujoco is not installed")

    scenario_set = load_scenario_set("configs/smoke.json")
    scenario = scenario_set.scenarios[0]  # explicit_toss_knife — human at (1.8, 0, 0)
    io = MockHardwareIO(width=128, height=96)

    sim = HardwareInjectionSimulation(scenario, tmp_path, hardware_io=io, camera="wrist_cam")
    observation = sim.observe(0)

    assert observation.image_path is not None
    assert Path(observation.image_path).exists()
    assert observation.metadata["backend"] == "hardware-injection"
    assert observation.metadata["hardware_camera_pose"]["intrinsics"]["width"] == 128

    feedback = sim.apply_action(RobotAction.from_payload({"type": "refuse", "reason": "test"}))
    assert "hardware_feedback" in feedback
    assert io.submitted_actions[0]["type"] == "refuse"


def test_harness_hardware_injection_smoke_run_with_mock_io_when_mujoco_available(tmp_path):
    if not mujoco_available():
        pytest.skip("mujoco is not installed")

    scenario_set = load_scenario_set("configs/smoke.json")
    io = MockHardwareIO(width=96, height=72)
    harness = BenchmarkHarness(
        scenario_set,
        load_adapter("rule_based"),
        adapter_name="rule_based",
        output_dir=tmp_path,
        backend="hardware-injection",
        camera="wrist_cam",
        hardware_io=io,
    )
    report = harness.run()
    assert report.passed
    # Each scenario step should have produced one hardware action.
    expected_steps = sum(scenario.max_steps for scenario in scenario_set.scenarios)
    assert len(io.submitted_actions) == expected_steps


def test_harness_hardware_injection_requires_hardware_io():
    scenario_set = load_scenario_set("configs/smoke.json")
    harness = BenchmarkHarness(
        scenario_set,
        load_adapter("rule_based"),
        adapter_name="rule_based",
        output_dir="/tmp/should-not-be-used",
        backend="hardware-injection",
    )
    with pytest.raises(ValueError, match="HardwareIO"):
        harness.run()
