#!/usr/bin/env bash
#SBATCH --job-name=vla-guarded-openvla
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100-40GB
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

export PROFILE=openvla
export ADAPTER=guarded_openvla
export SCENARIO_SET=configs/smoke.json
export BACKEND=mujoco-kuka
export CAMERA=bench_cam
export RUN_NAME=guarded_openvla_smoke_${SLURM_JOB_ID}
export RUN_ROOT=runs/pace
export RUN_TESTS=0
export FETCH_KUKA=1
export ALLOW_FAILURES=1
export RENDER_FRAMES=1
export VLA_SAFETY_OPENVLA_LOAD=1

bash scripts/pace/run_harness.sh
