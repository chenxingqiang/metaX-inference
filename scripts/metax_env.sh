#!/usr/bin/env bash
# Shared MetaX server environment — source from remote scripts
# shellcheck disable=SC2034

source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate base 2>/dev/null || true

export MACA_PATH="${MACA_PATH:-/opt/maca}"
export LD_LIBRARY_PATH="/opt/maca/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if command -v python3 &>/dev/null; then
  PYTHON="${PYTHON:-python3}"
elif command -v python &>/dev/null; then
  PYTHON="${PYTHON:-python}"
else
  echo "ERROR: python3/python not found" >&2
  return 1 2>/dev/null || exit 1
fi
export PYTHON
