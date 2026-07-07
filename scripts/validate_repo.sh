#!/usr/bin/env bash
# Local validation before remote deploy (no GPU required for most checks)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== metaX-inference local validation ==="

echo "[1/5] Python syntax..."
python3 -m py_compile \
  scripts/bench_qwen36.py \
  scripts/check_mtp_head.py \
  scripts/parse_bench_results.py \
  scripts/profile_decode.py \
  metax_kernels/mcoplib_bridge.py \
  metax_kernels/qwen36/fused_mlp.py \
  metax_kernels/qwen36/fused_rope_rms.py \
  engine/vllm_metax_plugin/register.py \
  engine/vllm_metax_plugin/loader.py

echo "[2/5] Shell script syntax..."
bash -n scripts/serve_qwen36_metax.sh
bash -n scripts/remote_run_all_benches.sh
bash -n scripts/run_phase3_mtp_bench.sh

echo "[3/5] Unit tests..."
if python3 -c "import torch" 2>/dev/null; then
  PYTHONPATH=. python3 -m unittest discover -s tests -v
else
  echo "  SKIP: torch not installed, installing CPU torch..."
  pip install torch --index-url https://download.pytorch.org/whl/cpu -q
  PYTHONPATH=. python3 -m unittest discover -s tests -v
fi

echo "[4/5] check_mtp_head fixture..."
PYTHONPATH=. python3 scripts/check_mtp_head.py tests/fixtures/model_with_mtp --json | grep -q '"has_mtp_head": true'
PYTHONPATH=. python3 scripts/check_mtp_head.py tests/fixtures/model_no_mtp --json | grep -q '"has_mtp_head": false'

echo "[5/5] mcoplib bridge smoke..."
PYTHONPATH=. python3 -c "
from metax_kernels.mcoplib_bridge import bootstrap_mcoplib, mcoplib_available
assert bootstrap_mcoplib() == [] or mcoplib_available()
print('  mcoplib_available:', mcoplib_available())
"

echo "=== ALL PASS ==="
