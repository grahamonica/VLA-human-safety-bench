from __future__ import annotations

from dataclasses import dataclass, field

from vla_safety_bench.types import JsonDict


@dataclass(frozen=True)
class ModelRuntimeSpec:
    id: str
    adapter_aliases: tuple[str, ...]
    display_name: str
    runtime: str
    source_repo: str | None = None
    checked_commit: str | None = None
    hf_model: str | None = None
    required_modules: tuple[str, ...] = ()
    recommended_modules: tuple[str, ...] = ()
    default_checkpoint: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> JsonDict:
        return {
            "id": self.id,
            "adapter_aliases": list(self.adapter_aliases),
            "display_name": self.display_name,
            "runtime": self.runtime,
            "source_repo": self.source_repo,
            "checked_commit": self.checked_commit,
            "hf_model": self.hf_model,
            "required_modules": list(self.required_modules),
            "recommended_modules": list(self.recommended_modules),
            "default_checkpoint": self.default_checkpoint,
            "notes": list(self.notes),
        }


MODEL_REGISTRY: dict[str, ModelRuntimeSpec] = {
    "openvla": ModelRuntimeSpec(
        id="openvla",
        adapter_aliases=("openvla",),
        display_name="OpenVLA",
        runtime="hf_vision2seq",
        source_repo="https://github.com/openvla/openvla",
        checked_commit="c8f03f48af692657d3060c19588038c7220e9af9",
        hf_model="openvla/openvla-7b",
        required_modules=("torch", "transformers", "PIL"),
        recommended_modules=("timm", "tokenizers", "flash_attn"),
        default_checkpoint="openvla/openvla-7b",
        notes=("Emits 7-DoF end-effector deltas; not semantic refusals.",),
    ),
    "pi0": ModelRuntimeSpec(
        id="pi0",
        adapter_aliases=("pi0", "pi_zero", "pi-zero"),
        display_name="pi-zero / openpi",
        runtime="openpi",
        source_repo="https://github.com/Physical-Intelligence/openpi",
        checked_commit="650c5b0283a49c42784fb5055a0507da2c6d347d",
        required_modules=("openpi", "numpy", "PIL"),
        recommended_modules=("torch", "jax"),
        default_checkpoint="gs://openpi-assets/checkpoints/pi05_droid",
        notes=("openpi recommends NVIDIA GPU inference and caches checkpoints under OPENPI_DATA_HOME.",),
    ),
    "octo": ModelRuntimeSpec(
        id="octo",
        adapter_aliases=("octo",),
        display_name="Octo",
        runtime="octo_jax",
        source_repo="https://github.com/octo-models/octo",
        checked_commit="241fb3514b7c40957a86d869fecb7c7fc353f540",
        hf_model="rail-berkeley/octo-small-1.5",
        required_modules=("octo", "jax", "numpy", "PIL"),
        recommended_modules=("tensorflow",),
        default_checkpoint="hf://rail-berkeley/octo-small-1.5",
        notes=("Octo expects observation dictionaries with image/state history and samples action chunks.",),
    ),
    "smolvla": ModelRuntimeSpec(
        id="smolvla",
        adapter_aliases=("smolvla", "smol_vla"),
        display_name="SmolVLA",
        runtime="lerobot",
        source_repo="https://github.com/huggingface/lerobot",
        checked_commit="05a5223885bcd36064fc1a967620329696595a76",
        hf_model="lerobot/smolvla_base",
        required_modules=("lerobot", "torch", "numpy", "PIL"),
        recommended_modules=("transformers",),
        default_checkpoint="lerobot/smolvla_base",
        notes=("Base model is intended for fine-tuning; raw base behavior should be interpreted cautiously.",),
    ),
    "tinyvla": ModelRuntimeSpec(
        id="tinyvla",
        adapter_aliases=("tinyvla", "tiny_vla"),
        display_name="TinyVLA",
        runtime="repo_subprocess",
        source_repo="https://github.com/liyaxuanliyaxuan/TinyVLA",
        checked_commit="94f441827b45e4f76316ef6a0ae443736dc93a5d",
        hf_model="lesjie/Llava-Pythia-400M",
        required_modules=("torch", "PIL", "h5py"),
        recommended_modules=("transformers",),
        default_checkpoint="lesjie/Llava-Pythia-400M",
        notes=("Official README points to repo-local llava-pythia and policy_heads packages.",),
    ),
    "nora": ModelRuntimeSpec(
        id="nora",
        adapter_aliases=("nora",),
        display_name="NORA",
        runtime="nora_repo",
        source_repo="https://github.com/declare-lab/nora",
        checked_commit="6b18c23d7875052e03fba4f8c2f32bd6a8a5c4a9",
        required_modules=("torch", "transformers", "PIL"),
        recommended_modules=("accelerate",),
        default_checkpoint="declare-lab/nora",
        notes=("Repository exposes inference.nora.Nora and returns 7-DoF actions.",),
    ),
    "nora15": ModelRuntimeSpec(
        id="nora15",
        adapter_aliases=("nora15", "nora_1_5", "nora-1.5"),
        display_name="NORA-1.5",
        runtime="nora15_repo",
        source_repo="https://github.com/declare-lab/nora-1.5",
        checked_commit="d1cdce29e9a9ce9f0e05d3f4b3d1c6eed592a9a9",
        hf_model="declare-lab/nora-1.5",
        required_modules=("torch", "transformers", "PIL"),
        recommended_modules=("accelerate",),
        default_checkpoint="declare-lab/nora-1.5",
        notes=("Repository exposes inference.modelling_expert.VLAWithExpert.from_pretrained.",),
    ),
    "bitvla": ModelRuntimeSpec(
        id="bitvla",
        adapter_aliases=("bitvla", "bit_vla"),
        display_name="BitVLA",
        runtime="bitvla_repo",
        source_repo="https://github.com/ustcwhy/BitVLA",
        checked_commit="8afac0260b3748b14657a69ec58e3d9f0d6da3a7",
        hf_model="lxsy/bitvla-bf16",
        required_modules=("torch", "transformers", "PIL"),
        recommended_modules=("accelerate",),
        default_checkpoint="lxsy/bitvla-bf16",
        notes=("Official repo recommends custom Transformers/OpenVLA-OFT integration for robotics eval.",),
    ),
}


ALIASES: dict[str, str] = {
    alias: model_id for model_id, spec in MODEL_REGISTRY.items() for alias in spec.adapter_aliases
}


def resolve_model_id(alias: str) -> str:
    normalized = alias.lower().strip().replace("_", "-")
    if alias in MODEL_REGISTRY:
        return alias
    for candidate, model_id in ALIASES.items():
        if candidate.lower().replace("_", "-") == normalized:
            return model_id
    raise KeyError(f"Unknown VLA model adapter: {alias}")


def get_model_spec(alias: str) -> ModelRuntimeSpec:
    return MODEL_REGISTRY[resolve_model_id(alias)]


def all_model_specs() -> list[ModelRuntimeSpec]:
    return list(MODEL_REGISTRY.values())
