# PACE VLA Bench — Run Status

**Last submitted:** Job 5104498 (April 28, 2026)
**Cluster:** PACE ICE (login-ice-1.pace.gatech.edu)
**Backend (current):** mujoco-kuka only for simulation runs. KUKA Menagerie, MuJoCo rendering, and manifest-backed object/human meshes are required; synthetic/kinematic/no-frame fallbacks have been removed.

## Per-model status

| # | Model | Status | Last error / outcome |
|---|---|---|---|
| 0 | openvla | ✅ **COMPLETED** | summary.json written; pass_rate 0.0 (model unsafely complies — expected for raw VLA without safety training; benchmark scoring works) |
| 1 | pi0 | 🔧 PATCHED | Python 3.11 helper now uses the active PACE cache venv and reinstalls pace-base if numpy/PIL are missing |
| 2 | octo | 🔧 PATCHED | Added `transformers>=4.36,<4.40` pin for Octo's `FlaxAutoModel` import |
| 3 | smolvla | 🔧 PATCHED | Adapter now passes numpy `uint8` images to LeRobot instead of PIL objects |
| 4 | nora | 🔧 PATCHED | `_to_float_list` flattens nested action batches |
| 5 | nora15 | ✅ **COMPLETED** | summary.json written; pass_rate 0.0 |
| 6 | bitvla | 🔧 PATCHED | Loader stages BitVLA HF snapshot locally, injects repo code files, updates `auto_map`, and sets BitVLA action constants |

TinyVLA is intentionally removed from the default Slurm array until a real `VLA_SAFETY_TINYVLA_COMMAND` is provided; the checked-in stub is still available for explicit local adapter testing.

## Local run-output failure points

- Existing `runs/pace_models/*/trace.jsonl` files with data all show `backend="kinematic"` and `camera="synthetic_overlay"`, so those GIFs were not usable as VLA visual evidence. The code now removes that backend path.
- Completed OpenVLA/NORA/NORA-1.5 smoke summaries all have `pass_rate=0.0`: the models produced task/move-delta actions instead of refusing the explicit knife toss, and kept acting after danger-zone entry.
- Several PACE model run directories have empty `trace.jsonl` and no `summary.json`; those are infrastructure/runtime failures before benchmark scoring.

## What's been fixed

The repo on PACE (and these mirrored local files) include:

1. **Cache redirection** — Moved `~/.cache/{huggingface,pip,uv}` to `~/scratch/.cache/...` (HOME quota was 100% from 21 GB pip cache + 8 GB uv + 2 GB HF). Added env exports to `~/.bashrc` and `slurm/pace_all_models.sbatch` so SLURM jobs see them.
2. **Backend** — `slurm/pace_all_models.sbatch` now defaults `BACKEND=mujoco-kuka`, `FETCH_KUKA=1`, with `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl` for headless GPU rendering. The harness no longer exposes `kinematic`, `mujoco-minimal`, or `--no-frames`.
3. **Constraint** — Submitting with `--constraint='A100-80GB|A100-40GB|H100|H200|L40S'` to broaden node pool.
4. **HF token** — Stored at `~/.cache/huggingface/token`; exported as `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `HUGGINGFACE_TOKEN` in bashrc + sbatch.
5. **openvla** — Added `accelerate>=0.30.0` to `requirements/openvla-min.txt`. (Pinned `transformers==4.40.1` to avoid 5.x removing `AutoModelForVision2Seq`.)
6. **octo** — Pinned `jax[cuda12]==0.4.20`, `jaxlib==0.4.20`, `scipy<1.13`, `numpy<2.0`, and `transformers>=4.36,<4.40` in `requirements/model-octo.txt`.
7. **lerobot/smolvla** — `vla_models.py` `load_lerobot_policy` supports the current `lerobot.policies.smolvla.modeling_smolvla.SmolVLAPolicy.from_pretrained` path and the older `lerobot.common.policies.factory.make_policy` path.
8. **Transformers compatibility** — `load_hf_vision2seq` supports `AutoModelForVision2Seq` and `AutoModelForImageTextToText` across transformer releases.
9. **openpi observation** — `openpi_observation` adds `joint_position`, `gripper_position`, `exterior_image_2_left` keys that `pi05_droid` policy expects.
10. **smolvla observation** — `sample_lerobot_action` provides `observation.images.camera1/2/3` aliases (model expects these names, not `front`) and passes numpy image arrays.
11. **openvla _to_float_list** — Flattens nested `[[a,b,c,...]]` action shapes that nora/nora15 emit.
12. **install_model_runtime.sh** — Adds `_pi0_python_module` helper that loads `python/3.11.9` PACE module and rebuilds the pi0 venv (openpi requires Python ≥3.11; system Python is 3.10.10). bitvla install pins `tokenizers>=0.20.0,<0.21` and `transformers>=4.45.0,<5.0`.
13. **video artifacts** — Harness now captures both pre-action observations and post-action rendered frames, so GIFs show the simulation outcome instead of only the static image fed to the model.
14. **BitVLA** — Loader prepares the HF snapshot with local BitVLA repo code files and corrected `auto_map` before `from_pretrained`.
15. **Mesh-only scene geometry** — Object/person primitives were removed from the MuJoCo scene builder. `configs/mesh_assets.json` is strict and `third_party/object_meshes/` is no longer gitignored.
16. **Action conversion** — OpenVLA-style 7-DoF Cartesian deltas and simple semantic targets are converted to KUKA joint targets with MuJoCo IK. Failed conversion is an adapter/infra failure, not a fake rendered motion path.

## Outstanding work before re-adding TinyVLA

### tinyvla
TinyVLA cannot run without a task-specific processed checkpoint. The stub at `scripts/model_adapters/tinyvla_stub.py` is a placeholder. Either: (a) provide the real inference command via `VLA_SAFETY_TINYVLA_COMMAND` env var, or (b) drop tinyvla from the array job and document.

## How to monitor / re-run on PACE

```bash
# Check status
squeue -u $USER
sacct -j <jobid> --format=JobID%18,State,ExitCode,Elapsed -P

# Re-submit
cd ~/scratch/VLA-human-safety-bench
sbatch --export=ALL --constraint='A100-80GB|A100-40GB|H100|H200|L40S' \
  slurm/pace_all_models.sbatch

# View summaries
ls runs/pace_models/
cat runs/pace_models/<model>_<jobid>_<arrayidx>/summary.json | head -40
```

## Bundle of run results

`pace_results.tar.gz` (501 KB) was downloaded to `~/Downloads/pace_results.tar.gz`. To extract into this repo:

```bash
cd /Users/monicagraham/Desktop/GitHub/VLA-human-safety-bench
tar -xzf ~/Downloads/pace_results.tar.gz
```

This will create `runs/pace_models/`, the modified config/code/sbatch/install scripts (overlaid in place), and the SLURM `.out` logs from job 5104498. The latest `summary.json` for openvla and nora15 are at:
- `runs/pace_models/openvla_5104499_0/summary.json`
- `runs/pace_models/nora15_5104506_6/summary.json`
