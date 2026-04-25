#!/usr/bin/env bash
#SBATCH --job-name=vla-all-models
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100-40GB
#SBATCH --time=08:00:00
#SBATCH --array=0-7
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

MODELS=(openvla pi0 octo smolvla tinyvla nora nora15 bitvla)
export MODEL_ID="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
export SCENARIO_SET="${SCENARIO_SET:-configs/smoke.json}"
export BACKEND="${BACKEND:-mujoco-kuka}"
export CAMERA="${CAMERA:-bench_cam}"
export RUN_ROOT="${RUN_ROOT:-runs/pace_models}"
export RUN_NAME="${MODEL_ID}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export ALLOW_FAILURES=1
export RENDER_FRAMES=1
export FETCH_KUKA=1

bash scripts/pace/run_model_adapter.sh "${MODEL_ID}"
