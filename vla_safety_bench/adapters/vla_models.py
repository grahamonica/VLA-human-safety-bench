from __future__ import annotations

import importlib.util
import json
import os
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
        from lerobot.common.policies.factory import make_policy
    except Exception as exc:
        raise RuntimeUnavailable(
            "Installed LeRobot does not expose the expected make_policy API."
        ) from exc

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
    from transformers import AutoModelForVision2Seq, AutoProcessor

    model_id = checkpoint or "lxsy/bitvla-bf16"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    return {"processor": processor, "model": model, "device": device, "dtype": dtype, "model_id": model_id}


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
    obs = {
        "observation.images.front": load_pil_image(observation),
        "task": str(observation.get("prompt", "")),
    }
    if hasattr(model, "select_action"):
        return model.select_action(obs)
    if hasattr(model, "predict_action"):
        return model.predict_action(obs)
    raise RuntimeUnavailable("LeRobot policy did not expose select_action or predict_action.")


def openpi_observation(observation: JsonDict) -> JsonDict:
    image = load_pil_image(observation)
    return {
        "observation/exterior_image_1_left": image,
        "observation/wrist_image_left": image,
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
