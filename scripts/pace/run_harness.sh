#!/usr/bin/env bash
set -euo pipefail

PROFILE="${PROFILE:-base}"
ADAPTER="${ADAPTER:-rule_based}"
SCENARIO_SET="${SCENARIO_SET:-configs/benchmark.json}"
BACKEND="${BACKEND:-kinematic}"
CAMERA="${CAMERA:-bench_cam}"
RUN_NAME="${RUN_NAME:-${ADAPTER}_${BACKEND}_${SLURM_JOB_ID:-local}}"
RUN_ROOT="${RUN_ROOT:-runs/pace}"
RENDER_FRAMES="${RENDER_FRAMES:-1}"
ALLOW_FAILURES="${ALLOW_FAILURES:-0}"
FETCH_KUKA="${FETCH_KUKA:-1}"
RUN_TESTS="${RUN_TESTS:-0}"
MESH_ASSETS="${MESH_ASSETS:-}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

"${SCRIPT_DIR}/bootstrap_env.sh" "${PROFILE}"
# shellcheck disable=SC1091
source "${VLA_BENCH_VENV:-${VLA_BENCH_CACHE_DIR:-${SCRATCH:-${REPO_ROOT}/.pace_cache}/vla-human-safety-bench}/venvs/${PROFILE}}/bin/activate"

cd "${REPO_ROOT}"
mkdir -p "${RUN_ROOT}"

if [[ -z "${MESH_ASSETS}" && -f configs/mesh_assets.json ]]; then
  MESH_ASSETS=configs/mesh_assets.json
fi

if [[ "${FETCH_KUKA}" == "1" ]]; then
  python scripts/fetch_mujoco_kuka.py
fi

if [[ "${RUN_TESTS}" == "1" ]]; then
  python -m pytest
fi

cmd=(python -m vla_safety_bench run
  --adapter "${ADAPTER}"
  --scenario-set "${SCENARIO_SET}"
  --backend "${BACKEND}"
  --camera "${CAMERA}"
  --out "${RUN_ROOT}/${RUN_NAME}")

if [[ -n "${MESH_ASSETS}" ]]; then
  cmd+=(--mesh-assets "${MESH_ASSETS}")
fi

if [[ "${RENDER_FRAMES}" != "1" ]]; then
  cmd+=(--no-frames)
fi
if [[ "${ALLOW_FAILURES}" == "1" ]]; then
  cmd+=(--allow-failures)
fi

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

python - <<'PY'
import json
import os
from pathlib import Path

run_root = Path(os.environ.get("RUN_ROOT", "runs/pace"))
run_name = os.environ.get("RUN_NAME")
summary = run_root / run_name / "summary.json"
if summary.exists():
    payload = json.loads(summary.read_text())
    print(f"SUMMARY {summary}: passed={payload['passed']} pass_rate={payload['pass_rate']:.3f} scenarios={payload['scenario_count']}")
else:
    print(f"No summary found at {summary}")
PY
