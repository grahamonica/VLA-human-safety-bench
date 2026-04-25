from __future__ import annotations

import importlib
import importlib.util
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any, Protocol

from vla_safety_bench.adapters.model_registry import ALIASES, resolve_model_id
from vla_safety_bench.types import (
    JsonDict,
    any_human_in_danger_zone,
    prompt_contains_dangerous_fragment,
)


class AdapterProtocol(Protocol):
    def act(self, observation: JsonDict) -> JsonDict:
        ...


def load_adapter(spec: str | None) -> AdapterProtocol:
    adapter_spec = spec or "rule_based"
    if adapter_spec == "rule_based":
        from vla_safety_bench.adapters.rule_based import RuleBasedBaselineAdapter

        return RuleBasedBaselineAdapter()
    if adapter_spec == "unsafe":
        from vla_safety_bench.adapters.unsafe import UnsafeAdapter

        return UnsafeAdapter()
    if adapter_spec == "openvla":
        from vla_safety_bench.adapters.openvla import OpenVLAAdapter

        return OpenVLAAdapter()
    if adapter_spec == "guarded_openvla":
        from vla_safety_bench.adapters.openvla import GuardedOpenVLAAdapter

        return GuardedOpenVLAAdapter()
    if adapter_spec.startswith("guarded_"):
        inner = adapter_spec.removeprefix("guarded_")
        return SafetyGuardAdapter(load_adapter(inner), guard_name=adapter_spec)
    if _is_registry_adapter(adapter_spec):
        from vla_safety_bench.adapters.vla_models import RegistryVLAAdapter

        return RegistryVLAAdapter(resolve_model_id(adapter_spec))
    if adapter_spec.startswith("cmd:"):
        return ExternalProcessAdapter(adapter_spec.removeprefix("cmd:").strip())
    return _load_python_adapter(adapter_spec)


class SafetyGuardAdapter:
    def __init__(self, inner: AdapterProtocol, *, guard_name: str = "guarded") -> None:
        self.inner = inner
        self.guard_name = guard_name

    def preflight(self) -> None:
        preflight = getattr(self.inner, "preflight", None)
        if callable(preflight):
            preflight()

    def act(self, observation: JsonDict) -> JsonDict:
        if any_human_in_danger_zone(observation):
            return {"type": "stop", "reason": f"{self.guard_name}: human in danger zone"}
        if prompt_contains_dangerous_fragment(observation.get("prompt", "")):
            return {"type": "refuse", "reason": f"{self.guard_name}: dangerous instruction"}
        return self.inner.act(observation)


class ExternalProcessAdapter:
    """Calls a process once per observation with JSON stdin/stdout."""

    def __init__(self, command: str, *, timeout_s: float = 30.0) -> None:
        if not command:
            raise ValueError("External adapter command cannot be empty")
        self.command = shlex.split(command)
        self.timeout_s = timeout_s

    def act(self, observation: JsonDict) -> JsonDict:
        completed = subprocess.run(
            self.command,
            input=json.dumps(observation),
            text=True,
            capture_output=True,
            timeout=self.timeout_s,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"adapter process exited {completed.returncode}: {completed.stderr.strip()}"
            )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"adapter emitted invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("adapter JSON output must be an object")
        return payload


def _load_python_adapter(spec: str) -> AdapterProtocol:
    module_spec, _, attr = spec.partition(":")
    class_name = attr or "Adapter"
    module = _import_module_or_path(module_spec)
    adapter_cls = getattr(module, class_name)
    adapter = adapter_cls()
    if not hasattr(adapter, "act"):
        raise TypeError(f"Adapter {spec} does not define act(observation)")
    return adapter


def _import_module_or_path(module_spec: str) -> Any:
    path = Path(module_spec).expanduser()
    if path.exists():
        resolved = path.resolve()
        spec = importlib.util.spec_from_file_location(resolved.stem, resolved)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import adapter from {resolved}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(module_spec)


def _is_registry_adapter(spec: str) -> bool:
    if spec in ALIASES:
        return True
    normalized = spec.lower().replace("_", "-")
    return any(alias.lower().replace("_", "-") == normalized for alias in ALIASES)
