from vla_safety_bench.adapters.base import load_adapter
from vla_safety_bench.adapters.model_registry import MODEL_REGISTRY
from vla_safety_bench.adapters.openvla import OpenVLAAdapter, normalize_openvla_action, openvla_status
from vla_safety_bench.adapters.vla_models import (
    RegistryVLAAdapter,
    RuntimeUnavailable,
    model_status,
    openpi_observation,
)
import pytest


def test_openvla_status_is_available_without_network():
    status = openvla_status(network=False).to_dict()
    assert status["model_id"] == "openvla/openvla-7b"
    assert "torch" in status["required_modules"]
    assert status["source"]["github_main_commit_checked"]


def test_openvla_adapter_does_not_load_by_default():
    adapter = OpenVLAAdapter(load_model=False)
    with pytest.raises(RuntimeError, match="runtime is not loaded"):
        adapter.act({"prompt": "Move the mug.", "objects": [{"name": "mug"}]})


def test_load_adapter_openvla_alias():
    adapter = load_adapter("openvla")
    assert isinstance(adapter, OpenVLAAdapter)


def test_all_registry_adapters_load_in_dry_mode():
    for model_id in MODEL_REGISTRY:
        if model_id == "openvla":
            continue
        adapter = load_adapter(model_id)
        assert isinstance(adapter, RegistryVLAAdapter)
        with pytest.raises(RuntimeUnavailable, match="runtime is not loaded"):
            adapter.act({"prompt": "Move the mug.", "objects": [{"name": "mug"}]})


def test_model_status_covers_all_requested_models():
    expected = {"openvla", "pi0", "octo", "smolvla", "tinyvla", "nora", "nora15", "bitvla"}
    assert expected.issubset(MODEL_REGISTRY)
    for model_id in expected:
        status = model_status(model_id).to_dict()
        assert status["id"] == model_id
        assert status["source_repo"] or status["hf_model"]


def test_normalize_openvla_action_keeps_raw_7dof():
    action = normalize_openvla_action(
        [0.1, 0.2, 0.2, 0.0, 0.0, 0.0, 1.0],
        {"prompt": "Move the mug.", "objects": [{"name": "mug"}]},
        "openvla/openvla-7b",
        "bridge_orig",
    )
    assert action["type"] == "move_delta"
    assert action["target_object"] == "mug"
    assert action["raw"]["openvla_action_7dof"] == [0.1, 0.2, 0.2, 0.0, 0.0, 0.0, 1.0]


def test_normalize_openvla_action_flattens_batched_7dof():
    action = normalize_openvla_action(
        [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]],
        {"prompt": "Move the mug.", "objects": [{"name": "mug"}]},
        "model",
        "key",
    )
    assert action["raw"]["openvla_action_7dof"] == [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]


def test_normalize_openvla_action_rejects_invalid_decoder_output():
    with pytest.raises(RuntimeError, match="non-sequence action"):
        normalize_openvla_action("not an action vector", {"prompt": "Move the mug."}, "model", "key")


def test_openpi_observation_uses_kuka_joint_and_gripper_metadata(tmp_path):
    from PIL import Image

    image_path = tmp_path / "obs.png"
    Image.new("RGB", (8, 8), (20, 30, 40)).save(image_path)
    payload = openpi_observation(
        {
            "prompt": "Move the mug.",
            "image_path": str(image_path),
            "metadata": {
                "kuka_joint_positions": {
                    "joint1": 0.1,
                    "joint2": 0.2,
                    "joint3": 0.3,
                    "joint4": 0.4,
                    "joint5": 0.5,
                    "joint6": 0.6,
                    "joint7": 0.7,
                },
                "kuka_gripper_open_fraction": 0.25,
            },
        }
    )
    assert payload["observation/joint_position"].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    assert payload["observation/gripper_position"].tolist() == pytest.approx([0.25])
