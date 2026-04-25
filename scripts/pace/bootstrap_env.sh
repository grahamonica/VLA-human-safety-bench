#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-base}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if [[ "${VLA_BENCH_LOAD_MODULES:-1}" == "1" ]] && command -v module >/dev/null 2>&1; then
  if [[ -n "${PACE_PYTHON_MODULE:-}" ]]; then
    module load "${PACE_PYTHON_MODULE}"
  fi
  if [[ -n "${PACE_CUDA_MODULE:-}" ]]; then
    module load "${PACE_CUDA_MODULE}"
  fi
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Could not find ${PYTHON_BIN}. Set PYTHON_BIN or PACE_PYTHON_MODULE." >&2
  exit 2
fi

CACHE_ROOT="${VLA_BENCH_CACHE_DIR:-${SCRATCH:-${REPO_ROOT}/.pace_cache}/vla-human-safety-bench}"
VENV_DIR="${VLA_BENCH_VENV:-${CACHE_ROOT}/venvs/${PROFILE}}"
mkdir -p "${CACHE_ROOT}/"{pip,huggingface,torch,logs}

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CACHE_ROOT}/pip}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${REPO_ROOT}/requirements/pace-base.txt"
if [[ "${PROFILE}" == "openvla" ]]; then
  python -m pip install -r "${REPO_ROOT}/requirements/openvla-min.txt"
fi
python -m pip install -e "${REPO_ROOT}"

python -m vla_safety_bench doctor
echo "VLA_BENCH_REPO_ROOT=${REPO_ROOT}"
echo "VLA_BENCH_VENV=${VENV_DIR}"
echo "VLA_BENCH_CACHE_DIR=${CACHE_ROOT}"

