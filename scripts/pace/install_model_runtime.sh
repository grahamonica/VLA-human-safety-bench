#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${1:?usage: source scripts/pace/install_model_runtime.sh <model-id>}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
CACHE_ROOT="${VLA_BENCH_CACHE_DIR:-${SCRATCH:-${REPO_ROOT}/.pace_cache}/vla-human-safety-bench}"
REPO_CACHE="${CACHE_ROOT}/repos"
mkdir -p "${REPO_CACHE}"

clone_checked_repo() {
  local name="$1"
  local url="$2"
  local commit="$3"
  local dest="${REPO_CACHE}/${name}"
  if [[ ! -d "${dest}/.git" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone --recursive "${url}" "${dest}" >&2
  fi
  git -C "${dest}" fetch --all --tags >&2
  git -C "${dest}" checkout --detach "${commit}" >&2
  git -C "${dest}" submodule update --init --recursive >&2
  local actual
  actual="$(git -C "${dest}" rev-parse HEAD)"
  if [[ "${actual}" != "${commit}" ]]; then
    echo "Refusing ${name}: expected ${commit}, got ${actual}" >&2
    return 3
  fi
  echo "${dest}"
}

install_repo_requirements_if_present() {
  local repo_dir="$1"
  local req="$2"
  if [[ -f "${repo_dir}/${req}" ]]; then
    python -m pip install -r "${repo_dir}/${req}"
  fi
}


# Per-model env tweaks (added by setup)
_pi0_python_module() {
    if command -v module >/dev/null 2>&1; then
        module load python/3.11.9 2>/dev/null || true
    fi
    PY311=$(command -v python3.11 || true)
    if [[ -z "$PY311" ]]; then
        echo "pi0 requires Python >=3.11, but python3.11 was not found after loading python/3.11.9." >&2
        return 2
    fi
    "$PY311" -m ensurepip --upgrade 2>/dev/null || "$PY311" -m ensurepip 2>/dev/null || true
    # Recreate the model-pi0 venv with python 3.11 if it currently uses 3.10 or is missing.
    local cache_root="${VLA_BENCH_CACHE_DIR:-${SCRATCH:-${REPO_ROOT}/.pace_cache}/vla-human-safety-bench}"
    local venv_dir="${VLA_BENCH_VENV:-${cache_root}/venvs/model-pi0}"
    if [[ ! -x "$venv_dir/bin/python" ]] || ! "$venv_dir/bin/python" -c 'import sys; assert sys.version_info[:2]>=(3,11)' 2>/dev/null; then
        rm -rf "$venv_dir"
        "$PY311" -m venv "$venv_dir"
    fi
    source "$venv_dir/bin/activate"
    python -m pip install --upgrade pip setuptools wheel
    if ! python -c 'import numpy, PIL' 2>/dev/null; then
        python -m pip install -r "$REPO_ROOT/requirements/pace-base.txt"
    fi
    python -m pip install -e "$REPO_ROOT"
}

case "${MODEL_ID}" in
  openvla)
    python -m pip install -r "${REPO_ROOT}/requirements/openvla-min.txt"
    export VLA_SAFETY_OPENVLA_LOAD=1
    ;;
  pi0|pi_zero|pi-zero)
    _pi0_python_module
    repo_dir="$(clone_checked_repo openpi https://github.com/Physical-Intelligence/openpi.git 650c5b0283a49c42784fb5055a0507da2c6d347d)"
    python -m pip install uv
    (cd "${repo_dir}" && GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .)
    export VLA_SAFETY_PI0_REPO="${repo_dir}/src"
    export VLA_SAFETY_PI0_LOAD=1
    export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${CACHE_ROOT}/openpi}"
    ;;
  octo)
    repo_dir="$(clone_checked_repo octo https://github.com/octo-models/octo.git 241fb3514b7c40957a86d869fecb7c7fc353f540)"
    install_repo_requirements_if_present "${repo_dir}" requirements.txt
    python -m pip install -r "${REPO_ROOT}/requirements/model-octo.txt"
    python -m pip install -e "${repo_dir}"
    python -m pip install -r "${REPO_ROOT}/requirements/model-octo.txt"
    export VLA_SAFETY_OCTO_REPO="${repo_dir}"
    export VLA_SAFETY_OCTO_LOAD=1
    ;;
  smolvla|smol_vla)
    repo_dir="$(clone_checked_repo lerobot https://github.com/huggingface/lerobot.git 05a5223885bcd36064fc1a967620329696595a76)"
    python -m pip install -r "${REPO_ROOT}/requirements/model-smolvla.txt" || python -m pip install -e "${repo_dir}[smolvla]"
    export VLA_SAFETY_SMOLVLA_REPO="${repo_dir}"
    export VLA_SAFETY_SMOLVLA_LOAD=1
    ;;
  tinyvla|tiny_vla)
    repo_dir="$(clone_checked_repo tinyvla https://github.com/liyaxuanliyaxuan/TinyVLA.git 94f441827b45e4f76316ef6a0ae443736dc93a5d)"
    install_repo_requirements_if_present "${repo_dir}" requirements.txt
    if [[ -d "${repo_dir}/policy_heads" ]]; then python -m pip install -e "${repo_dir}/policy_heads"; fi
    if [[ -d "${repo_dir}/llava-pythia" ]]; then python -m pip install -e "${repo_dir}/llava-pythia"; fi
    export VLA_SAFETY_TINYVLA_REPO="${repo_dir}"
    export VLA_SAFETY_TINYVLA_LOAD=1
    export VLA_SAFETY_TINYVLA_COMMAND="${VLA_SAFETY_TINYVLA_COMMAND:-python ${REPO_ROOT}/scripts/model_adapters/tinyvla_stub.py}"
    ;;
  nora)
    repo_dir="$(clone_checked_repo nora https://github.com/declare-lab/nora.git 6b18c23d7875052e03fba4f8c2f32bd6a8a5c4a9)"
    python -m pip install -r "${REPO_ROOT}/requirements/model-nora-common.txt"
    install_repo_requirements_if_present "${repo_dir}/inference" requirements.txt
    export VLA_SAFETY_NORA_REPO="${repo_dir}"
    export VLA_SAFETY_NORA_LOAD=1
    ;;
  nora15|nora_1_5|nora-1.5)
    repo_dir="$(clone_checked_repo nora-1.5 https://github.com/declare-lab/nora-1.5.git d1cdce29e9a9ce9f0e05d3f4b3d1c6eed592a9a9)"
    python -m pip install -r "${REPO_ROOT}/requirements/model-nora-common.txt"
    install_repo_requirements_if_present "${repo_dir}" requirements.txt
    export VLA_SAFETY_NORA15_REPO="${repo_dir}"
    export VLA_SAFETY_NORA15_LOAD=1
    ;;
  bitvla|bit_vla)
    repo_dir="$(clone_checked_repo bitvla https://github.com/ustcwhy/BitVLA.git 8afac0260b3748b14657a69ec58e3d9f0d6da3a7)"
    python -m pip install -r "${REPO_ROOT}/requirements/openvla-min.txt"
    python -m pip install -U "tokenizers>=0.20.0,<0.21" "transformers>=4.45.0,<5.0" accelerate
    python -m pip install -e "${repo_dir}/openvla-oft/bitvla"
    export VLA_SAFETY_BITVLA_REPO="${repo_dir}"
    export VLA_SAFETY_BITVLA_LOAD=1
    ;;
  *)
    echo "Unknown model id: ${MODEL_ID}" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

python -m vla_safety_bench model-check --model "${MODEL_ID}" || true
