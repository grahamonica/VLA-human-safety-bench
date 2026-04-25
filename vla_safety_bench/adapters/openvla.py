from __future__ import annotations

import importlib.util
import json
import math
import os
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from vla_safety_bench.types import (
    JsonDict,
    any_human_in_danger_zone,
    prompt_contains_dangerous_fragment,
)

OPENVLA_MODEL_ID = "openvla/openvla-7b"
OPENVLA_GITHUB_REPO = "https://github.com/openvla/openvla"
OPENVLA_HF_MODEL_URL = f"https://huggingface.co/{OPENVLA_MODEL_ID}"
OPENVLA_HF_API_URL = f"https://huggingface.co/api/models/{OPENVLA_MODEL_ID}"
OPENVLA_MAIN_COMMIT = "c8f03f48af692657d3060c19588038c7220e9af9"

REQUIRED_MODULES = ("torch", "transformers", "PIL")
RECOMMENDED_MODULES = ("timm", "tokenizers")


@dataclass(frozen=True)
class OpenVLAStatus:
    model_id: str
    required_modules: dict[str, str]
    recommended_modules: dict[str, str]
    can_import_runtime: bool
    can_load_model: bool
    device_hint: str
    source: JsonDict
    network: JsonDict | None = None

    def to_dict(self) -> JsonDict:
        return {
            "model_id": self.model_id,
            "required_modules": self.required_modules,
            "recommended_modules": self.recommended_modules,
            "can_import_runtime": self.can_import_runtime,
            "can_load_model": self.can_load_model,
            "device_hint": self.device_hint,
            "source": self.source,
            "network": self.network,
        }


class OpenVLAAdapter:
    """Optional OpenVLA bridge.

    The adapter intentionally does not install dependencies or download weights
    unless `load_model=True` or `VLA_SAFETY_OPENVLA_LOAD=1` is set.
    """

    def __init__(
        self,
        *,
        model_id: str = OPENVLA_MODEL_ID,
        device: str | None = None,
        unnorm_key: str = "bridge_orig",
        load_model: bool | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.unnorm_key = unnorm_key
        self.trust_remote_code = trust_remote_code
        self.processor: Any | None = None
        self.model: Any | None = None
        should_load = load_model
        if should_load is None:
            should_load = os.environ.get("VLA_SAFETY_OPENVLA_LOAD") == "1"
        if should_load:
            self._load_model()

    def act(self, observation: JsonDict) -> JsonDict:
        if self.model is None or self.processor is None:
            raise RuntimeError(
                "OpenVLA runtime is not loaded. Set VLA_SAFETY_OPENVLA_LOAD=1 after installing "
                "the required dependencies and model access."
            )
        return self._predict(observation)

    def preflight(self) -> None:
        if self.model is None or self.processor is None:
            raise RuntimeError(
                "OpenVLA preflight failed: runtime is not loaded. This is an infrastructure error, "
                "not a benchmark result."
            )

    def _load_model(self) -> None:
        missing = [name for name, state in module_status(REQUIRED_MODULES).items() if state == "missing"]
        if missing:
            raise RuntimeError(f"Missing OpenVLA dependencies: {', '.join(missing)}")

        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        resolved_device = self.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        if resolved_device == "cpu" and os.environ.get("VLA_SAFETY_OPENVLA_ALLOW_CPU") != "1":
            raise RuntimeError(
                "OpenVLA is an 8B model; refusing CPU load by default. "
                "Use a CUDA environment or set VLA_SAFETY_OPENVLA_ALLOW_CPU=1 for an explicit experiment."
            )

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
        )
        model_kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16 if resolved_device.startswith("cuda") else torch.float32,
            "low_cpu_mem_usage": True,
            "trust_remote_code": self.trust_remote_code,
        }
        if importlib.util.find_spec("flash_attn"):
            model_kwargs["attn_implementation"] = "flash_attention_2"
        self.model = AutoModelForVision2Seq.from_pretrained(self.model_id, **model_kwargs).to(resolved_device)
        self.device = resolved_device

    def _predict(self, observation: JsonDict) -> JsonDict:
        import torch
        from PIL import Image

        image_path = observation.get("image_path")
        if not image_path:
            raise RuntimeError("OpenVLA adapter requires observation.image_path; visual feed is missing.")
        image = Image.open(str(image_path)).convert("RGB")
        instruction = str(observation.get("prompt", "")).strip()
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        device = self.device or "cuda:0"
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        inputs = self.processor(prompt, image).to(device, dtype=dtype)
        raw_action = self.model.predict_action(
            **inputs,
            unnorm_key=self.unnorm_key,
            do_sample=False,
        )
        return normalize_openvla_action(raw_action, observation, self.model_id, self.unnorm_key)


class GuardedOpenVLAAdapter(OpenVLAAdapter):
    """OpenVLA behind a transparent benchmark safety envelope.

    This is useful for sim-to-real demos where an independent safety layer must
    block known-dangerous requests. It should not be reported as raw OpenVLA
    performance.
    """

    def act(self, observation: JsonDict) -> JsonDict:
        if any_human_in_danger_zone(observation):
            return {"type": "stop", "reason": "guarded_openvla safety envelope: human in danger zone"}
        if prompt_contains_dangerous_fragment(observation.get("prompt", "")):
            return {"type": "refuse", "reason": "guarded_openvla safety envelope: dangerous instruction"}
        return super().act(observation)


def normalize_openvla_action(
    raw_action: Any,
    observation: JsonDict,
    model_id: str,
    unnorm_key: str,
) -> JsonDict:
    values = _to_float_list(raw_action)
    target_object = _guess_target_object(observation)
    translation_norm = math.sqrt(sum(value * value for value in values[:3])) if len(values) >= 3 else None
    action_type = "move_delta"
    if target_object and "knife" in target_object:
        action_type = "move_delta"
    return {
        "type": action_type,
        "target_object": target_object,
        "speed_mps": translation_norm,
        "raw": {
            "model_id": model_id,
            "unnorm_key": unnorm_key,
            "openvla_action_7dof": values,
            "note": "OpenVLA emits normalized/un-normalized 7-DoF end-effector deltas, not semantic refusals.",
        },
    }


def openvla_status(*, network: bool = False) -> OpenVLAStatus:
    required = module_status(REQUIRED_MODULES)
    recommended = module_status(RECOMMENDED_MODULES)
    can_import_runtime = all(state == "available" for state in required.values())
    device_hint = _device_hint() if required.get("torch") == "available" else "unavailable: torch missing"
    can_load_model = can_import_runtime and device_hint.startswith("cuda")
    network_payload = check_openvla_network() if network else None
    return OpenVLAStatus(
        model_id=OPENVLA_MODEL_ID,
        required_modules=required,
        recommended_modules=recommended,
        can_import_runtime=can_import_runtime,
        can_load_model=can_load_model,
        device_hint=device_hint,
        source={
            "github_repo": OPENVLA_GITHUB_REPO,
            "github_main_commit_checked": OPENVLA_MAIN_COMMIT,
            "huggingface_model": OPENVLA_HF_MODEL_URL,
            "license_note": "HF model card says MIT; GitHub README notes pretrained models may inherit Llama-2 license restrictions.",
        },
        network=network_payload,
    )


def check_openvla_network() -> JsonDict:
    try:
        request = urllib.request.Request(OPENVLA_HF_API_URL, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=20, context=_ssl_context()) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"reachable": False, "error": str(exc)}
    return {
        "reachable": True,
        "model_id": payload.get("id"),
        "pipeline_tag": payload.get("pipeline_tag"),
        "tags": payload.get("tags", [])[:20],
    }


def module_status(names: tuple[str, ...]) -> dict[str, str]:
    return {name: "available" if importlib.util.find_spec(name) else "missing" for name in names}


def _device_hint() -> str:
    try:
        import torch
    except Exception:
        return "unavailable: torch import failed"
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
    except Exception:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def _to_float_list(raw_action: Any) -> list[float]:
    if hasattr(raw_action, "tolist"):
        raw_action = raw_action.tolist()
    if not isinstance(raw_action, (list, tuple)):
        raise RuntimeError(f"OpenVLA action decoder returned non-sequence action: {raw_action!r}.")
    values: list[float] = []
    for value in raw_action:
        try:
            values.append(float(value))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"OpenVLA action decoder returned non-numeric action value: {value!r}.") from exc
    if not values:
        raise RuntimeError("OpenVLA action decoder returned an empty action vector.")
    return values


def _guess_target_object(observation: JsonDict) -> str | None:
    prompt = str(observation.get("prompt", "")).lower()
    objects = observation.get("objects", [])
    names = [str(obj.get("name")) for obj in objects if isinstance(obj, dict) and obj.get("name")]
    for name in names:
        if name.lower() in prompt:
            return name
    return names[0] if names else None
