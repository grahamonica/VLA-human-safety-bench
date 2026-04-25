#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/dist}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ARCHIVE="${OUT_DIR}/vla-human-safety-bench-pace-${STAMP}.tar.gz"

mkdir -p "${OUT_DIR}"
tar \
  --exclude='.git' \
  --exclude='.pytest_cache' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='runs' \
  --exclude='dist' \
  --exclude='.pace_cache' \
  --exclude='third_party/mujoco_menagerie' \
  -czf "${ARCHIVE}" \
  -C "${REPO_ROOT}/.." \
  "$(basename "${REPO_ROOT}")"

echo "${ARCHIVE}"

