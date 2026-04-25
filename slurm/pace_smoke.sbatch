#!/usr/bin/env bash
#SBATCH --job-name=vla-safety-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

export PROFILE=base
export ADAPTER=rule_based
export SCENARIO_SET=configs/smoke.json
export BACKEND=mujoco-kuka
export CAMERA=bench_cam
export RUN_NAME=smoke_${SLURM_JOB_ID}
export RUN_ROOT=runs/pace
export RUN_TESTS=1
export FETCH_KUKA=1

bash scripts/pace/run_harness.sh
