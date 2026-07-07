#!/usr/bin/env bash
# Run operator micro-benchmark on MACA GPU
set -euo pipefail
cd "$(dirname "$0")/.."
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate base 2>/dev/null || true
export MACA_PATH="${MACA_PATH:-/opt/maca}"
export LD_LIBRARY_PATH="/opt/maca/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}"

python -m metax_kernels.bench.op_bench "$@"
