# VLA Human Safety Bench

This repository scaffolds a safety benchmark for vision-language-action (VLA) models that interact near humans. It is built from the project document in `Copy of VLA safety human interaction bench-2.pdf` and focuses on two initial risk families:

- Endangering humans: explicit dangerous requests, unsafe knife handovers, and dangerous subtasks hidden inside ordinary instructions.
- Human danger zone: refusal before starting when a human is in the robot work envelope, plus slow/stop behavior if a human enters during execution.

The default harness runs locally with the Python standard library. Optional MuJoCo and image packages are detected at runtime; nothing is installed automatically. Frame-enabled runs fail hard if a requested renderer or visual dependency is missing.

## Quick Start

```bash
python -m vla_safety_bench doctor
python -m vla_safety_bench list --scenario-set configs/benchmark.json
python -m vla_safety_bench run --adapter rule_based --scenario-set configs/benchmark.json --backend kinematic --out runs/rule_based
python -m pytest
```

The built-in `rule_based` adapter is a sanity baseline that should pass the current benchmark. The `unsafe` adapter intentionally ignores the safety rules and is useful for checking that failures are caught.

## PACE

PACE-ready Slurm scripts are included under `slurm/`. To create a clean upload bundle:

```bash
bash scripts/pace/make_upload_bundle.sh
```

On PACE, unpack the bundle and submit:

```bash
sbatch slurm/pace_smoke.sbatch
sbatch slurm/pace_full_benchmark.sbatch
sbatch slurm/pace_openvla_a100.sbatch
sbatch slurm/pace_all_models.sbatch
```

See [docs/pace.md](docs/pace.md) for job details and optional overrides.

## Simulation Backends

The default `kinematic` backend steps a deterministic robot/floor/human world state and records safety events in `trace.jsonl`.

```bash
python -m vla_safety_bench run --adapter unsafe --scenario-set configs/smoke.json --backend kinematic --out runs/unsafe_smoke --no-frames
python -m vla_safety_bench run --adapter rule_based --scenario-set configs/smoke.json --backend mujoco-kuka --camera wrist_cam --out runs/kuka_wrist_smoke
```

`mujoco-minimal` uses a primitive robot proxy for fast smoke testing. `mujoco-kuka` uses the real KUKA iiwa 14 MJCF from MuJoCo Menagerie, plus a floor, knife, mug, tennis ball, container, human capsule, fixed scene cameras, and a `wrist_cam` attached to KUKA `link7`. Both MuJoCo modes use deterministic kinematic stepping for trace consistency while rendering the current world state into the visual feed.

The current textured glTF/GLB downloads can be converted into MuJoCo-ready OBJ meshes and texture references with:

```bash
python scripts/import_gltf_assets.py
python -m vla_safety_bench run --adapter rule_based --scenario-set configs/smoke.json --backend mujoco-kuka --camera wrist_cam --mesh-assets configs/mesh_assets.json --out runs/kuka_mesh_smoke
```

The manifest supports OBJ/STL/MSH meshes. For converted glTF/GLB assets, the base-color texture is used as the MuJoCo material texture, JPEG textures are converted to PNG, and normal/roughness maps are copied alongside the asset for future renderers. The mesh is used for camera visuals, and transparent primitive collision geoms remain in the body for stable MuJoCo physics.

## Adapting a VLA

Adapters receive a JSON-serializable observation and return a JSON-serializable action. Start with `vla_safety_bench/adapters/base.py` and implement:

```python
class Adapter:
    def act(self, observation):
        return {"type": "refuse", "reason": "unsafe request"}
```

Then run it with:

```bash
python -m vla_safety_bench run --adapter /path/to/adapter.py:Adapter --scenario-set configs/benchmark.json --out runs/my_vla
```

External process adapters are also supported:

```bash
python -m vla_safety_bench run --adapter 'cmd:python /path/to/policy_server.py' --out runs/my_process
```

The harness sends one observation JSON object on stdin and expects one action JSON object on stdout for each step.

Adapter failures are fail-fast by design. Non-object JSON, missing action `type`, invalid `speed_mps`, missing visual input, missing referenced objects, invalid model action vectors, and subprocess failures terminate the run with a nonzero exit code. `--allow-failures` only affects valid traces that score as unsafe.

## OpenVLA Bridge

The OpenVLA bridge is optional and does not clone repos, install packages, or download model weights by default.

```bash
python -m vla_safety_bench openvla-check
```

To run the actual model, prepare a separate environment with the OpenVLA runtime dependencies and set:

```bash
VLA_SAFETY_OPENVLA_LOAD=1 python -m vla_safety_bench run --adapter openvla --scenario-set configs/smoke.json --out runs/openvla
```

Raw OpenVLA returns 7-DoF end-effector actions rather than semantic refusals. For sim-to-real demonstrations that require an independent safety layer, `--adapter guarded_openvla` adds a transparent prompt and danger-zone envelope around the same bridge.

## VLA Adapter Registry

Supported adapter aliases:

```bash
python -m vla_safety_bench models
python -m vla_safety_bench model-check
```

The registry currently covers `openvla`, `pi0`, `octo`, `smolvla`, `tinyvla`, `nora`, `nora15`, and `bitvla`. Raw model adapters run only when their runtime dependencies are installed. On PACE, `slurm/pace_all_models.sbatch` installs each runtime into an isolated profile before running the smoke benchmark.

## Optional MuJoCo Assets

The project document asks for the KUKA iiwa 14 MuJoCo model from Google DeepMind's MuJoCo Menagerie. The repo and model path were verified against `google-deepmind/mujoco_menagerie` at pinned commit `affef0836947b64cc06c4ab1cbf0152835693374`, and the KUKA model is BSD-3-Clause licensed.

Fetch the selected KUKA files explicitly:

```bash
python scripts/fetch_mujoco_kuka.py --dest third_party/mujoco_menagerie
python -m vla_safety_bench doctor
```

Human 3D assets are intentionally not fetched. The document names RenderPeople free 3D people as the intended source; that requires a web flow/license acceptance, so this repo only provides placeholders and 2D synthetic overlays until those assets are added manually.

Without `--mesh-assets`, the knife, mug, tennis ball, container, and human proxy visuals are procedural MuJoCo primitives created in code. They are not downloaded meshes.

## Repository Layout

- `configs/benchmark.json`: seven initial benchmark scenarios.
- `configs/vlas.json`: target VLA integration inventory.
- `vla_safety_bench/`: harness, simulation state, adapters, scoring, overlays, and optional MuJoCo utilities.
- `scripts/fetch_mujoco_kuka.py`: pinned, hash-validated downloader for the KUKA Menagerie assets.
- `docs/`: architecture, asset policy, and integration notes.
- `tests/`: harness and scoring tests.

## Safety Notes

This harness evaluates model behavior in simulation and synthetic camera frames. It is not a replacement for real robot safety controls, risk assessment, or hardware interlocks. Physical robot integration should keep an independent safety layer in charge of emergency stop, speed limits, collision constraints, and workspace exclusion.
