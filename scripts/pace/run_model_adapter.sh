#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-${1:-openvla}}"
PROFILE="model-${MODEL_ID}"
ADAPTER="${ADAPTER:-${MODEL_ID}}"
SCENARIO_SET="${SCENARIO_SET:-configs/smoke.json}"
BACKEND="${BACKEND:-mujoco-kuka}"
CAMERA="${CAMERA:-bench_cam}"
RUN_NAME="${RUN_NAME:-${MODEL_ID}_${SLURM_JOB_ID:-local}}"
RUN_ROOT="${RUN_ROOT:-runs/pace_models}"
ALLOW_FAILURES="${ALLOW_FAILURES:-1}"
FETCH_KUKA="${FETCH_KUKA:-1}"
MESH_ASSETS="${MESH_ASSETS:-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

"${SCRIPT_DIR}/bootstrap_env.sh" "${PROFILE}"
# shellcheck disable=SC1091
source "${VLA_BENCH_VENV:-${VLA_BENCH_CACHE_DIR:-${SCRATCH:-${REPO_ROOT}/.pace_cache}/vla-human-safety-bench}/venvs/${PROFILE}}/bin/activate"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/install_model_runtime.sh" "${MODEL_ID}"

cd "${REPO_ROOT}"

if [[ -z "${MESH_ASSETS}" && -f configs/mesh_assets.json ]]; then
  MESH_ASSETS=configs/mesh_assets.json
fi
if [[ -z "${MESH_ASSETS}" || ! -f "${MESH_ASSETS}" ]]; then
  echo "MESH_ASSETS must point to a mesh manifest; no fallback renderer is available." >&2
  exit 2
fi

if [[ "${FETCH_KUKA}" == "1" ]]; then
  python scripts/fetch_mujoco_kuka.py
fi

cmd=(python -m vla_safety_bench run
  --adapter "${ADAPTER}"
  --scenario-set "${SCENARIO_SET}"
  --backend "${BACKEND}"
  --camera "${CAMERA}"
  --out "${RUN_ROOT}/${RUN_NAME}")

cmd+=(--mesh-assets "${MESH_ASSETS}")

if [[ "${ALLOW_FAILURES}" == "1" ]]; then cmd+=(--allow-failures); fi

printf 'Running model adapter:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
