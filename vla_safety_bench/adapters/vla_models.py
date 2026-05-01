from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from vla_safety_bench.adapters.model_registry import ModelRuntimeSpec, get_model_spec
from vla_safety_bench.adapters.openvla import normalize_openvla_action
from vla_safety_bench.types import JsonDict


class RuntimeUnavailable(RuntimeError):
    pass


class RegistryVLAAdapter:
    """Adapter facade for model runtimes that are installed only on PACE/profile envs."""

    def __init__(
        self,
        model_id: str,
        *,
        load_model: bool | None = None,
        checkpoint: str | None = None,
        device: str | None = None,
    ) -> None:
        self.spec = get_model_spec(model_id)
        self.checkpoint = checkpoint or self.spec.default_checkpoint
        self.device = device or os.environ.get("VLA_SAFETY_DEVICE", "cuda")
        self.model: Any | None = None
        should_load = load_model
        if should_load is None:
            should_load = os.environ.get(f"VLA_SAFETY_{self.spec.id.upper()}_LOAD") == "1"
        if should_load:
            self._load_model()

    def act(self, observation: JsonDict) -> JsonDict:
        if self.model is None:
            raise RuntimeUnavailable(
                f"{self.spec.display_name} runtime is not loaded. Run model-check or set the "
                "adapter-specific LOAD env var on PACE."
            )
        return self._predict(observation)

    def preflight(self) -> None:
        if self.model is None:
            raise RuntimeUnavailable(
                f"{self.spec.display_name} preflight failed: runtime is not loaded. "
                "This is an infrastructure error, not a benchmark result."
            )

    def _load_model(self) -> None:
        repo_dir = os.environ.get(f"VLA_SAFETY_{self.spec.id.upper()}_REPO")
        if repo_dir:
            path = Path(repo_dir).expanduser().resolve()
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
            if self.spec.runtime == "bitvla_repo":
                for extra_path in (
                    path / "openvla-oft",
                    path / "openvla-oft" / "bitvla",
                    path / "openvla-oft" / "bitvla" / "model",
                ):
                    if extra_path.exists() and str(extra_path) not in sys.path:
                        sys.path.insert(0, str(extra_path))

        missing = [
            module
            for module, state in module_status(self.spec.required_modules).items()
            if state == "missing"
        ]
        if missing:
            raise RuntimeUnavailable(
                f"Missing {self.spec.display_name} runtime modules: {', '.join(missing)}"
            )

        if self.spec.runtime == "openpi":
            self.model = load_openpi_policy(self.checkpoint)
        elif self.spec.runtime == "octo_jax":
            self.model = load_octo_policy(self.checkpoint)
        elif self.spec.runtime == "lerobot":
            self.model = load_lerobot_policy(self.checkpoint, self.device)
        elif self.spec.runtime == "nora_repo":
            self.model = load_nora_policy(self.device)
        elif self.spec.runtime == "nora15_repo":
            self.model = load_nora15_policy(self.checkpoint, self.device)
        elif self.spec.runtime == "bitvla_repo":
            self.model = load_hf_vision2seq(self.checkpoint, self.device)
        elif self.spec.runtime == "repo_subprocess":
            command = os.environ.get("VLA_SAFETY_TINYVLA_COMMAND")
            if not command:
                raise RuntimeUnavailable(
                    "TinyVLA requires VLA_SAFETY_TINYVLA_COMMAND pointing to a repo-local inference command."
                )
            self.model = SubprocessModel(command)
        else:
            raise RuntimeUnavailable(f"Unsupported runtime {self.spec.runtime}")

    def _predict(self, observation: JsonDict) -> JsonDict:
        if self.spec.runtime == "openpi":
            action = self.model.infer(openpi_observation(observation))["actions"]
            return normalize_openvla_action(action, observation, self.spec.display_name, "openpi")
        if self.spec.runtime == "octo_jax":
            action = sample_octo_action(self.model, observation)
            return normalize_openvla_action(action, observation, self.spec.display_name, "octo")
        if self.spec.runtime == "lerobot":
            action = sample_lerobot_action(self.model, observation)
            return normalize_openvla_action(action, observation, self.spec.display_name, "lerobot")
        if self.spec.runtime == "nora_repo":
            action = self.model.inference(
                image=load_pil_image(observation),
                instruction=str(observation.get("prompt", "")),
                unnorm_key=os.environ.get("VLA_SAFETY_UNNORM_KEY", "bridge_orig"),
            )
            return normalize_openvla_action(action, observation, self.spec.display_name, "nora")
        if self.spec.runtime == "nora15_repo":
            action = self.model.sample_actions(
                load_pil_image(observation),
                str(observation.get("prompt", "")),
                num_steps=int(os.environ.get("VLA_SAFETY_ACTION_STEPS", "10")),
            )
            return normalize_openvla_action(action, observation, self.spec.display_name, "nora15")
        if self.spec.runtime == "bitvla_repo":
            return predict_hf_vision2seq(self.model, observation, self.spec.display_name)
        if isinstance(self.model, SubprocessModel):
            return self.model.predict(observation)
        raise RuntimeUnavailable(f"No prediction path for {self.spec.runtime}")


class ModelStatus:
    def __init__(self, spec: ModelRuntimeSpec, modules: dict[str, str]) -> None:
        self.spec = spec
        self.modules = modules

    @property
    def can_import_runtime(self) -> bool:
        return all(state == "available" for state in self.modules.values())

    def to_dict(self) -> JsonDict:
        return {
            **self.spec.to_dict(),
            "module_status": self.modules,
            "can_import_runtime": self.can_import_runtime,
        }


class SubprocessModel:
    def __init__(self, command: str) -> None:
        from vla_safety_bench.adapters.base import ExternalProcessAdapter

        self.adapter = ExternalProcessAdapter(command)

    def predict(self, observation: JsonDict) -> JsonDict:
        return self.adapter.act(observation)


def model_status(spec: ModelRuntimeSpec | str) -> ModelStatus:
    resolved = get_model_spec(spec) if isinstance(spec, str) else spec
    return ModelStatus(resolved, module_status(resolved.required_modules))


def module_status(names: tuple[str, ...]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for name in names:
        import_name = {"PIL": "PIL", "openpi": "openpi", "octo": "octo"}.get(name, name)
        normalized[name] = "available" if importlib.util.find_spec(import_name) else "missing"
    return normalized


def load_openpi_policy(checkpoint: str | None) -> Any:
    from openpi.policies import policy_config
    from openpi.shared import download
    from openpi.training import config as openpi_config

    config_name = os.environ.get("VLA_SAFETY_OPENPI_CONFIG", "pi05_droid")
    config = openpi_config.get_config(config_name)
    checkpoint_dir = download.maybe_download(checkpoint or "gs://openpi-assets/checkpoints/pi05_droid")
    return policy_config.create_trained_policy(config, checkpoint_dir)


def load_octo_policy(checkpoint: str | None) -> Any:
    from octo.model.octo_model import OctoModel

    return OctoModel.load_pretrained(checkpoint or "hf://rail-berkeley/octo-small-1.5")


def load_lerobot_policy(checkpoint: str | None, device: str) -> Any:
    try:
        # New lerobot API: use the policy class directly
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        return SmolVLAPolicy.from_pretrained(checkpoint or "lerobot/smolvla_base").to(device)
    except ImportError:
        pass
    try:
        from lerobot.common.policies.factory import make_policy
    except ImportError as exc:
        raise RuntimeUnavailable(
            "Installed LeRobot does not expose the expected make_policy API."
        ) from exc

    # Old API path (only reached if SmolVLAPolicy import failed but make_policy exists)
    return make_policy(policy_path=checkpoint or "lerobot/smolvla_base", device=device)


def load_nora_policy(device: str) -> Any:
    from inference.nora import Nora

    return Nora(device=device)


def load_nora15_policy(checkpoint: str | None, device: str) -> Any:
    from inference.modelling_expert import VLAWithExpert

    model = VLAWithExpert.from_pretrained(checkpoint or "declare-lab/nora-1.5")
    if hasattr(model, "to"):
        model = model.to(device)
    return model


def load_hf_vision2seq(checkpoint: str | None, device: str) -> dict[str, Any]:
    import torch
    try:
        from transformers import AutoModelForVision2Seq, AutoProcessor
    except ImportError:
        from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
        from transformers import AutoProcessor

    model_id = checkpoint or "lxsy/bitvla-bf16"
    if "bitvla" in model_id.lower():
        model_id = _prepare_bitvla_checkpoint(model_id)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    if "bitvla" in str(model_id).lower():
        _configure_bitvla_constants(model)
    return {"processor": processor, "model": model, "device": device, "dtype": dtype, "model_id": model_id}


def _prepare_bitvla_checkpoint(model_id: str) -> str:
    checkpoint_path = Path(model_id).expanduser()
    if not checkpoint_path.exists():
        try:
            from huggingface_hub import snapshot_download
        except Exception:
            return model_id
        checkpoint_path = Path(snapshot_download(repo_id=model_id)).resolve()
    if not checkpoint_path.is_dir():
        return model_id

    repo_dir = os.environ.get("VLA_SAFETY_BITVLA_REPO")
    if not repo_dir:
        return str(checkpoint_path)
    prepared_root = Path(os.environ.get("VLA_BENCH_CACHE_DIR", str(checkpoint_path.parent))).expanduser()
    prepared_path = prepared_root / "bitvla_prepared" / checkpoint_path.name
    shutil.copytree(checkpoint_path, prepared_path, symlinks=True, dirs_exist_ok=True)
    checkpoint_path = prepared_path.resolve()

    repo_path = Path(repo_dir).expanduser().resolve()
    source_root = repo_path / "openvla-oft" / "bitvla"
    config_src = source_root / "configuration_bit_vla.py"
    model_src = source_root / "model" / "bitvla_for_action_prediction.py"
    if not config_src.exists() or not model_src.exists():
        return str(checkpoint_path)

    _copy_if_needed(config_src, checkpoint_path / "configuration_bit_vla.py")
    _copy_if_needed(config_src, checkpoint_path / "configuration_bitvla.py")
    _copy_if_needed(model_src, checkpoint_path / "bitvla_for_action_prediction.py")
    _copy_if_needed(model_src, checkpoint_path / "bitvla.py")
    _update_bitvla_auto_map(checkpoint_path / "config.json")
    return str(checkpoint_path)


def _copy_if_needed(source: Path, destination: Path) -> None:
    if destination.exists() and destination.read_bytes() == source.read_bytes():
        return
    shutil.copy2(source, destination)


def _update_bitvla_auto_map(config_path: Path) -> None:
    if not config_path.exists():
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        return
    config["auto_map"] = {
        "AutoConfig": "configuration_bit_vla.Bitvla_Config",
        "AutoModelForVision2Seq": "bitvla_for_action_prediction.BitVLAForActionPrediction",
    }
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _configure_bitvla_constants(model: Any) -> None:
    if not hasattr(model, "set_constant"):
        return
    try:
        from bitvla.constants import (
            BITNET_ACTION_TOKEN_BEGIN_IDX,
            BITNET_DEFAULT_IMAGE_TOKEN_IDX,
            BITNET_IGNORE_INDEX,
            BITNET_PROPRIO_PAD_IDX,
            BITNET_STOP_INDEX,
        )
    except Exception:
        return
    model.set_constant(
        image_token_idx=BITNET_DEFAULT_IMAGE_TOKEN_IDX,
        proprio_pad_idx=BITNET_PROPRIO_PAD_IDX,
        ignore_idx=BITNET_IGNORE_INDEX,
        action_token_begin_idx=BITNET_ACTION_TOKEN_BEGIN_IDX,
        stop_index=BITNET_STOP_INDEX,
    )


def predict_hf_vision2seq(runtime: dict[str, Any], observation: JsonDict, family: str) -> JsonDict:
    image = load_pil_image(observation)
    prompt = f"In: What action should the robot take to {observation.get('prompt', '')}?\nOut:"
    inputs = runtime["processor"](prompt, image).to(runtime["device"], dtype=runtime["dtype"])
    model = runtime["model"]
    if hasattr(model, "predict_action"):
        action = model.predict_action(**inputs, do_sample=False)
        return normalize_openvla_action(action, observation, family, runtime["model_id"])
    output = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    raw_tokens = output.tolist() if hasattr(output, "tolist") else repr(output)
    raise RuntimeUnavailable(
        f"{family} generated text rather than normalized action; add model-specific decoder. "
        f"Raw tokens: {raw_tokens}"
    )


def sample_octo_action(model: Any, observation: JsonDict) -> Any:
    import jax
    import numpy as np

    image = np.asarray(load_pil_image(observation).resize((256, 256)))
    obs = {
        "image_primary": image[None, None],
        "timestep_pad_mask": np.array([[True]]),
        "pad_mask_dict": {"image_primary": np.array([[True]])},
    }
    task = model.create_tasks(texts=[str(observation.get("prompt", ""))])
    return model.sample_actions(obs, task, rng=jax.random.PRNGKey(0))[0]


def sample_lerobot_action(model: Any, observation: JsonDict) -> Any:
    import numpy as np

    image = np.asarray(load_pil_image(observation), dtype=np.uint8)
    obs = {
        "observation.images.front": image,
        "observation.images.camera1": image,
        "observation.images.camera2": image,
        "observation.images.camera3": image,
        "task": str(observation.get("prompt", "")),
    }
    if hasattr(model, "select_action"):
        return model.select_action(obs)
    if hasattr(model, "predict_action"):
        return model.predict_action(obs)
    raise RuntimeUnavailable("LeRobot policy did not expose select_action or predict_action.")


def openpi_observation(observation: JsonDict) -> JsonDict:
    import numpy as np
    image = load_pil_image(observation)
    metadata = observation.get("metadata") if isinstance(observation.get("metadata"), dict) else {}
    raw_joint_positions = metadata.get("kuka_joint_positions", {})
    joint_positions = np.zeros(7, dtype=np.float32)
    if isinstance(raw_joint_positions, dict):
        for index, name in enumerate(("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7")):
            value = raw_joint_positions.get(name)
            if value is not None:
                joint_positions[index] = float(value)
    gripper_open = metadata.get("kuka_gripper_open_fraction")
    if gripper_open is None:
        ctrl = metadata.get("kuka_gripper_ctrl")
        if isinstance(ctrl, (int, float)):
            gripper_open = 1.0 - (min(max(float(ctrl), 0.0), 255.0) / 255.0)
    gripper_position = np.array([float(gripper_open or 0.0)], dtype=np.float32)
    # pi05_droid expects state and image keys
    return {
        "observation/exterior_image_1_left": image,
        "observation/exterior_image_2_left": image,
        "observation/wrist_image_left": image,
        "observation/joint_position": joint_positions,
        "observation/gripper_position": gripper_position,
        "prompt": str(observation.get("prompt", "")),
    }


def load_pil_image(observation: JsonDict) -> Any:
    from PIL import Image

    image_path = observation.get("image_path")
    if image_path:
        return Image.open(str(image_path)).convert("RGB")
    raise RuntimeUnavailable("VLA adapter requires observation.image_path; visual feed is missing.")


def dump_status_json(model_ids: list[str] | None = None) -> str:
    from vla_safety_bench.adapters.model_registry import all_model_specs

    specs = [get_model_spec(model_id) for model_id in model_ids] if model_ids else all_model_specs()
    return json.dumps([model_status(spec).to_dict() for spec in specs], indent=2, sort_keys=True)
