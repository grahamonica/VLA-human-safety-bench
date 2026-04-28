# VLA Human Safety Bench

This repository scaffolds a safety benchmark for vision-language-action (VLA) models that interact near humans. It is built from the project document in `Copy of VLA safety human interaction bench-2.pdf` and focuses on two initial risk families:

- Endangering humans: explicit dangerous requests, unsafe knife handovers, and dangerous subtasks hidden inside ordinary instructions.
- Human danger zone: refusal before starting when a human is in the robot work envelope, plus slow/stop behavior if a human enters during execution.

The harness requires MuJoCo, Pillow, NumPy, KUKA Menagerie assets, and a mesh manifest. There is no synthetic or no-frame simulation fallback; missing render dependencies fail before scoring.

## Quick Start

```bash
python -m vla_safety_bench doctor
python -m vla_safety_bench list --scenario-set configs/benchmark.json
python scripts/fetch_mujoco_kuka.py
python -m vla_safety_bench run --adapter rule_based --scenario-set configs/benchmark.json --backend mujoco-kuka --mesh-assets configs/mesh_assets.json --out runs/rule_based
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

The default `mujoco-kuka` backend uses the real KUKA iiwa 14 MJCF from MuJoCo Menagerie, third-party object/human meshes from `configs/mesh_assets.json`, fixed scene cameras, and a `wrist_cam` attached to KUKA `link7`.

```bash
python -m vla_safety_bench run --adapter unsafe --scenario-set configs/smoke.json --backend mujoco-kuka --mesh-assets configs/mesh_assets.json --out runs/unsafe_smoke --allow-failures
python -m vla_safety_bench run --adapter rule_based --scenario-set configs/smoke.json --backend mujoco-kuka --camera wrist_cam --mesh-assets configs/mesh_assets.json --out runs/kuka_wrist_smoke
```

The harness writes rendered PNG frames for every step and, unless `--no-video` is set, animated GIF slideshow artifacts under `videos/` listed in `summary.json`. MuJoCo KUKA runs render `bench_cam`, `overhead_cam`, and `wrist_cam` into the review slideshow by default while leaving the model's observation camera controlled by `--camera`. Use `--video-cameras bench_cam,wrist_cam` to choose a subset.

`mujoco-kuka` keeps a persistent `MjModel`/`MjData`, writes KUKA joint targets to the iiwa actuators, calls `mj_step`, renders all observations from MuJoCo, and records MuJoCo contact events in the trace. Adapter joint targets are used directly. OpenVLA-style 7-DoF Cartesian deltas and simple task targets are converted to KUKA joint targets with MuJoCo IK; impossible conversions fail the run instead of falling back to canned motion.

`hardware-injection` is the real-robot backend. At each step it reads a live wrist-cam frame and the wrist-cam pose from a `HardwareIO` driver you provide, renders the scenario's human mesh from that exact pose using the same MuJoCo manifest used in sim, alpha-composites the rendered human onto the real frame, and feeds the composited frame to the VLA. The VLA's action is forwarded to the driver. No real human is ever in the workspace; the perceived human is injected pixels. See [Hardware-Injection Backend](#hardware-injection-backend).

The current textured glTF/GLB downloads can be converted into MuJoCo-ready OBJ meshes and texture references with:

```bash
python scripts/import_gltf_assets.py
python -m vla_safety_bench run --adapter rule_based --scenario-set configs/smoke.json --backend mujoco-kuka --camera wrist_cam --mesh-assets configs/mesh_assets.json --out runs/kuka_mesh_smoke
```

The manifest supports OBJ/STL/MSH meshes. For converted glTF/GLB assets, the base-color texture is used as the MuJoCo material texture, JPEG textures are converted to PNG, and normal/roughness maps are copied alongside the asset for future renderers. Object and human bodies use manifest meshes directly; procedural object/person geometry has been removed.

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

For `--backend mujoco-kuka`, adapters can drive real MuJoCo stepping by including absolute joint targets or deltas at the top level or under `raw`. Lists are ordered as `joint1` through `joint7`; mappings may use names such as `joint1` or `j1`.

```json
{
  "type": "move_delta",
  "raw": {"joint_deltas": [0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
}
```

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

Raw OpenVLA returns 7-DoF end-effector actions rather than semantic refusals; the benchmark scores those raw outputs.

## Hardware-Injection Backend

When the benchmark runs on a real robot, the camera feed is real but the "human" is still purely synthetic — a 3D-scanned model rendered from the wrist cam's pose at the synthetic human's world position, then composited into the real frame before the VLA sees it. The benchmark therefore measures the same property on hardware as it does in sim (does the VLA refuse / de-escalate when it perceives a human?) without ever exposing a person to the arm.

To run:

```bash
python -m vla_safety_bench run \
  --backend hardware-injection \
  --camera wrist_cam \
  --hardware-io path/to/your_driver.py:YourHardwareIO \
  --adapter openvla \
  --scenario-set configs/benchmark.json \
  --out runs/hw_openvla
```

`--hardware-io` accepts:

- `mock` — uses [`MockHardwareIO`](vla_safety_bench/hardware/hardware_io.py) (constant background frame, fixed cam pose). Useful for end-to-end smoke tests with no robot attached.
- `module:Class` or `/path/to/file.py:Class` — your implementation of the `HardwareIO` Protocol, instantiated with no arguments.

A real driver implements four reads and one write:

```python
class YourHardwareIO:
    def read_camera_frame(self, camera_name: str): ...      # uint8 HxWx3
    def read_camera_intrinsics(self, camera_name: str): ... # CameraIntrinsics
    def read_camera_pose(self, camera_name: str): ...       # CameraPose (world frame)
    def read_robot_state(self) -> dict: ...                 # logged in trace
    def submit_action(self, action: dict) -> dict: ...      # forwarded to controller
```

`CameraPose.rotation_matrix` is a 3×3 row-major matrix whose columns are the camera-frame axes expressed in world coordinates: column 0 = image right, column 1 = image up, column 2 = away from the scene (the optical axis is `-z`). For a wrist cam attached to KUKA `link7`, you typically compute this by chaining the arm's forward kinematics with the calibrated cam mount transform.

Mesh quality for the injected human follows the same `--mesh-assets` path as the MuJoCo backends — point at the manifest with your RenderPeople OBJ + texture and the wrist-cam composite uses the same model.

## VLA Adapter Registry

Supported adapter aliases:

```bash
python -m vla_safety_bench models
python -m vla_safety_bench model-check
```

The registry currently covers `openvla`, `pi0`, `octo`, `smolvla`, `tinyvla`, `nora`, `nora15`, and `bitvla`. Raw model adapters run only when their runtime dependencies are installed. On PACE, `slurm/pace_all_models.sbatch` installs each runtime into an isolated profile before running the smoke benchmark.

## Required MuJoCo Assets

The project document asks for the KUKA iiwa 14 MuJoCo model from Google DeepMind's MuJoCo Menagerie. The repo and model path were verified against `google-deepmind/mujoco_menagerie` at pinned commit `affef0836947b64cc06c4ab1cbf0152835693374`, and the KUKA model is BSD-3-Clause licensed.

Fetch the selected KUKA files explicitly:

```bash
python scripts/fetch_mujoco_kuka.py --dest third_party/mujoco_menagerie
python -m vla_safety_bench doctor
```

Human 3D assets are intentionally not fetched. The document names RenderPeople free 3D people as the intended source; that requires a web flow/license acceptance, so this repo expects those meshes to be present under `third_party/object_meshes` and referenced by `configs/mesh_assets.json`.

## Repository Layout

- `configs/benchmark.json`: seven initial benchmark scenarios.
- `configs/vlas.json`: target VLA integration inventory.
- `vla_safety_bench/`: harness, MuJoCo simulation, hardware injection, adapters, scoring, and mesh utilities.
- `scripts/fetch_mujoco_kuka.py`: pinned, hash-validated downloader for the KUKA Menagerie assets.
- `docs/`: architecture, asset policy, and integration notes.
- `tests/`: harness and scoring tests.

## Safety Notes

By design this benchmark never exposes a real human to the robot. "Humans" are 2D pixel overlays composited into the camera feed (rendered from a 3D scan), so dangerous actions the VLA might emit cannot harm anyone — the worst case is the arm moving in an empty workspace. The benchmark measures whether the VLA refuses or de-escalates when it *perceives* a human in the frame; it is not a hardware safety controller and does not replace standard robot interlocks (e-stop, joint limits, collision envelopes) that protect against equipment damage.
