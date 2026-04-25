# VLA Integration Notes

The initial target list from the project document is captured in `configs/vlas.json`:

- OpenVLA
- pi-zero
- Octo
- SmolVLA
- TinyVLA
- NORA / NORA-1.5
- BitVLA

Each model should be integrated as an adapter that conforms to `vla_safety_bench.adapters.base.AdapterProtocol`.

## Recommended Adapter Pattern

1. Convert the harness observation into the model's expected prompt/image/action input.
2. Call the model in a sandboxed local environment or through an already-running process.
3. Normalize the model output into a benchmark action dictionary.
4. Preserve raw model output in `raw` for debugging.

Example action:

```json
{
  "type": "handover",
  "target_object": "knife",
  "orientation": "handle_toward_human",
  "speed_mps": 0.12,
  "raw": {"model_action": "..."}
}
```

## Integration Risk Checklist

- Keep model repos pinned to a commit or release.
- Read install scripts before running them.
- Prefer adapters that call already-installed local environments.
- Do not give model code direct write access to benchmark results beyond its own run directory.
- Log prompts, selected frames, normalized actions, and raw actions for auditability.

## Adapter Registry

Run:

```bash
python -m vla_safety_bench models
python -m vla_safety_bench model-check
```

The adapter registry covers:

| Model | Adapter | Runtime strategy |
| --- | --- | --- |
| OpenVLA | `openvla` | Hugging Face `AutoProcessor` / `AutoModelForVision2Seq` |
| pi-zero | `pi0` | Physical Intelligence `openpi` policy runtime |
| Octo | `octo` | `octo.model.octo_model.OctoModel` JAX runtime |
| SmolVLA | `smolvla` | Hugging Face LeRobot policy runtime |
| TinyVLA | `tinyvla` | Repo-local subprocess hook for processed task checkpoint |
| NORA | `nora` | `inference.nora.Nora` |
| NORA-1.5 | `nora15` | `inference.modelling_expert.VLAWithExpert` |
| BitVLA | `bitvla` | Hugging Face/custom Transformers-style VLA runtime |

Raw adapters fail preflight until their runtime is installed and the relevant `VLA_SAFETY_<MODEL>_LOAD=1` environment variable is set. The PACE model Slurm scripts handle this setup in isolated environments. Runtime/load failures are infrastructure errors, not benchmark results.

Adapter failures are intentionally fail-fast: non-object JSON, missing action `type`, bad `speed_mps`, missing visual input, invalid model action vectors, missing referenced objects, and external adapter subprocess failures all terminate the run with a nonzero exit code. `--allow-failures` only changes the exit code for valid benchmark traces that score as unsafe.

## OpenVLA

This repo includes an optional `openvla` adapter and a `guarded_openvla` adapter. The checked public sources are:

- GitHub: `https://github.com/openvla/openvla`
- Checked `main` commit: `c8f03f48af692657d3060c19588038c7220e9af9`
- Hugging Face model: `https://huggingface.co/openvla/openvla-7b`

Run:

```bash
python -m vla_safety_bench openvla-check
```

The adapter intentionally does not install dependencies or download the 7B model automatically. Raw OpenVLA returns 7-DoF end-effector deltas rather than explicit semantic refusals, so refusal-oriented scenarios should be interpreted carefully. `guarded_openvla` adds a transparent safety envelope for demonstrations where independent safety controls are required.
