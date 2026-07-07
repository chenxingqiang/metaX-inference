#!/usr/bin/env bash
# Quick smoke test on MetaX server (~5 min, no full benchmark sweep)
set -euo pipefail

REPO_DIR="${REPO_DIR:-/data/metaX-inference}"
MODEL="${MODEL:-/data/models/Qwen3.6-27B-AWQ}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
LOG="${LOG:-/data/metax-test-logs/quick-smoke.log}"
GITHUB_RAW="${GITHUB_RAW:-https://raw.githubusercontent.com/chenxingqiang/metaX-inference}"

if [[ -f "${REPO_DIR}/scripts/metax_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${REPO_DIR}/scripts/metax_env.sh"
else
  curl -fsSL "$GITHUB_RAW/main/scripts/metax_env.sh" -o /tmp/metax_env.sh
  # shellcheck source=/dev/null
  source /tmp/metax_env.sh
fi

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
"$PYTHON" scripts/check_mtp_head.py "$MODEL" --json || true

echo "--- Operator micro-bench (S=256) ---"
PYTHONPATH=. bash scripts/run_op_bench.sh --seq-len 256 --json || true

echo "--- mcoplib stub path (METAX_MCOP_STUB) ---"
METAX_MCOP_STUB=1 PYTHONPATH=. "$PYTHON" -c "
from metax_kernels.mcoplib_bridge import bootstrap_mcoplib
wired = bootstrap_mcoplib()
print('stub wired:', wired)
" || true

if curl -sf "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "--- vLLM already up: single request bench ---"
  "$PYTHON" scripts/bench_qwen36.py --url "http://${HOST}:${PORT}" --max-tokens 32 --json
else
  echo "--- vLLM not running (skip E2E; start with serve_qwen36_metax.sh) ---"
fi

echo "--- Acceptance preview ---"
"$PYTHON" scripts/bench_acceptance.py /data/metax-test-logs 2>/dev/null || \
  "$PYTHON" scripts/bench_acceptance.py . --markdown

echo "=== quick smoke done === log: $LOG"
