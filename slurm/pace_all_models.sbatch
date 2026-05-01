#!/usr/bin/env bash
#SBATCH --job-name=vla-all-models
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100-40GB
#SBATCH --time=08:00:00
#SBATCH --array=0-6
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

# --- Cache redirection (added by setup) ---
export VLA_BENCH_CACHE_DIR="$HOME/scratch/VLA-human-safety-bench/.pace_cache/vla-human-safety-bench"
export HF_HOME="$HOME/scratch/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export PIP_CACHE_DIR="$HOME/scratch/.cache/pip"
export UV_CACHE_DIR="$HOME/scratch/.cache/uv"
export TORCH_HOME="$VLA_BENCH_CACHE_DIR/torch"
export XDG_CACHE_HOME="$HOME/scratch/.cache"
export TMPDIR="$HOME/scratch/tmp"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_TOKEN="$(cat ~/.cache/huggingface/token 2>/dev/null || true)"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export HUGGINGFACE_TOKEN="$HF_TOKEN"
mkdir -p "$TMPDIR" "$HF_HOME/hub" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$TORCH_HOME" 2>/dev/null
if command -v module >/dev/null 2>&1; then module load python/3.10.10 2>/dev/null || true; fi

MODELS=(openvla pi0 octo smolvla nora nora15 bitvla)
export MODEL_ID="${MODELS[${SLURM_ARRAY_TASK_ID}]}"
export SCENARIO_SET="${SCENARIO_SET:-configs/smoke.json}"
export BACKEND="${BACKEND:-mujoco-kuka}"
export CAMERA="${CAMERA:-bench_cam}"
export RUN_ROOT="${RUN_ROOT:-runs/pace_models}"
export RUN_NAME="${MODEL_ID}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export ALLOW_FAILURES=1
export FETCH_KUKA=1

bash scripts/pace/run_model_adapter.sh "${MODEL_ID}"
