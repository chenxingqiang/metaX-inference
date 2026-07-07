#!/usr/bin/env bash
# Quick smoke test on MetaX server (~5 min, no full benchmark sweep)
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate base 2>/dev/null || true
export MACA_PATH="${MACA_PATH:-/opt/maca}"
export LD_LIBRARY_PATH="/opt/maca/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

REPO_DIR="${REPO_DIR:-/data/metaX-inference}"
MODEL="${MODEL:-/data/models/Qwen3.6-27B-AWQ}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
LOG="${LOG:-/data/metax-test-logs/quick-smoke.log}"

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "=== metaX-inference quick smoke ==="
echo "time: $(date -Iseconds)"
echo "gpu: $(mx-smi 2>/dev/null | grep -m1 MetaX || echo unknown)"

if [[ ! -d "$REPO_DIR/scripts" ]]; then
  echo "Repo missing — run sync_from_github.sh first"
  curl -fsSL "https://raw.githubusercontent.com/chenxingqiang/metaX-inference/main/scripts/sync_from_github.sh" | bash
fi

cd "$REPO_DIR"
bash scripts/test-env-check.sh || true

echo "--- MTP head check ---"
python scripts/check_mtp_head.py "$MODEL" --json || true

echo "--- Operator micro-bench (S=256) ---"
PYTHONPATH=. bash scripts/run_op_bench.sh --seq-len 256 --json || true

echo "--- mcoplib stub path (METAX_MCOP_STUB) ---"
METAX_MCOP_STUB=1 PYTHONPATH=. python3 -c "
from metax_kernels.mcoplib_bridge import bootstrap_mcoplib
wired = bootstrap_mcoplib()
print('stub wired:', wired)
" || true

if curl -sf "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "--- vLLM already up: single request bench ---"
  python scripts/bench_qwen36.py --url "http://${HOST}:${PORT}" --max-tokens 32 --json
else
  echo "--- vLLM not running (skip E2E; start with serve_qwen36_metax.sh) ---"
fi

echo "--- Acceptance preview ---"
python scripts/bench_acceptance.py /data/metax-test-logs 2>/dev/null || \
  python scripts/bench_acceptance.py . --markdown

echo "=== quick smoke done === log: $LOG"
