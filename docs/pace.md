# Georgia Tech PACE Runbook

This repo is set up so the uploaded directory can be submitted directly with Slurm.

## What Was Assumed

PACE uses Slurm. Public PACE training material shows GPU jobs requested with directives such as:

```bash
#SBATCH -N1 --gres=gpu:1
```

The same material mentions GPU constraints such as `A100-40GB` for Phoenix GPU nodes. The OpenVLA scripts request an A100-class node because OpenVLA is an 8B BF16 model and is not a good fit for a small CPU-only or low-memory GPU job.

## Upload

Create a clean bundle from your laptop:

```bash
bash scripts/pace/make_upload_bundle.sh
```

Upload the printed `.tar.gz` to PACE, unpack it, and submit from the repo root:

```bash
sbatch slurm/pace_smoke.sbatch
sbatch slurm/pace_full_benchmark.sbatch
sbatch slurm/pace_openvla_a100.sbatch
sbatch slurm/pace_all_models.sbatch
```

The Slurm scripts create a virtualenv under `$SCRATCH/vla-human-safety-bench` when `$SCRATCH` exists, install explicit requirements, fetch pinned KUKA MuJoCo assets with hash verification, and write outputs under `runs/pace/`.

## Jobs

- `pace_smoke.sbatch`: installs the base harness, runs tests, compiles the minimal MuJoCo scene, and runs smoke scenarios.
- `pace_full_benchmark.sbatch`: runs the full rule-based benchmark and the unsafe negative-control smoke run.
- `pace_openvla_a100.sbatch`: installs OpenVLA dependencies, downloads/caches the public `openvla/openvla-7b` model through Hugging Face, and records raw OpenVLA safety results.
- `pace_guarded_openvla_a100.sbatch`: same model path, but with a transparent guard layer for demo-oriented safety-envelope tests.
- `pace_all_models.sbatch`: Slurm array for `openvla`, `pi0`, `octo`, `smolvla`, `tinyvla`, `nora`, `nora15`, and `bitvla`. Each array task creates its own profile under `$SCRATCH` and installs only that model runtime.

`pace_all_models.sbatch` intentionally uses `--allow-failures`; unsafe behavior is a benchmark result written to `summary.json`, while infrastructure failures still stop that array task.

The PACE smoke/model scripts now default to `BACKEND=mujoco-kuka` and `CAMERA=bench_cam`. To run the arm-mounted visual feed:

```bash
sbatch --export=ALL,CAMERA=wrist_cam slurm/pace_smoke.sbatch
```

If `configs/mesh_assets.json` exists in the uploaded bundle, the PACE scripts pass it to the harness automatically. You can also point to a different manifest:

```bash
sbatch --export=ALL,MESH_ASSETS=configs/mesh_assets.json,CAMERA=wrist_cam slurm/pace_full_benchmark.sbatch
```

The upload bundle does not exclude `third_party/object_meshes/`, so locally added STL/OBJ/MSH files are included in the tarball. KUKA Menagerie assets are still excluded and re-fetched on PACE from the pinned, hash-verified source.

## Optional Overrides

You should not need these for the default path, but they are supported:

```bash
sbatch --constraint=A100-80GB slurm/pace_openvla_a100.sbatch
sbatch --export=ALL,PACE_PYTHON_MODULE=python/3.11 slurm/pace_smoke.sbatch
sbatch --export=ALL,VLA_BENCH_CACHE_DIR=/path/to/cache slurm/pace_openvla_a100.sbatch
```

If your PACE allocation requires an account, submit with:

```bash
sbatch -A <account> slurm/pace_smoke.sbatch
```

## Outputs

Each run writes:

- `summary.json`: aggregate pass rate and per-scenario findings.
- `trace.jsonl`: per-step observation, action, simulation feedback, and safety events.
- `frames/`: synthetic camera frames when rendering is enabled.
