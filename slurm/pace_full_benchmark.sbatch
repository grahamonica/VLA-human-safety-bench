#!/usr/bin/env bash
#SBATCH --job-name=vla-safety-full
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

export PROFILE=base
export ADAPTER=rule_based
export SCENARIO_SET=configs/benchmark.json
export BACKEND=mujoco-kuka
export CAMERA=bench_cam
export RUN_NAME=full_rule_based_${SLURM_JOB_ID}
export RUN_ROOT=runs/pace
export RUN_TESTS=1
export FETCH_KUKA=1

bash scripts/pace/run_harness.sh

export ADAPTER=unsafe
export SCENARIO_SET=configs/smoke.json
export RUN_NAME=negative_control_unsafe_${SLURM_JOB_ID}
export ALLOW_FAILURES=1
export RENDER_FRAMES=0
bash scripts/pace/run_harness.sh
