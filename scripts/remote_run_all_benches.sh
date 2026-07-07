#!/usr/bin/env bash
# One-shot: run Phase 1 concurrent + Phase 2 op bench + Phase 3 MTP on MetaX server
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate base 2>/dev/null || true
export MACA_PATH="${MACA_PATH:-/opt/maca}"
export LD_LIBRARY_PATH="/opt/maca/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

REPO_DIR="${REPO_DIR:-/data/metaX-inference}"
LOG_ROOT="${LOG_ROOT:-/data/metax-test-logs}"
SUMMARY="$LOG_ROOT/ALL_BENCH_SUMMARY.md"

mkdir -p "$LOG_ROOT"

{
  echo "# metaX-inference Full Benchmark — $(date -Iseconds)"
  echo ""
  echo "- Repo: $REPO_DIR"
  echo "- GPU: $(mx-smi 2>/dev/null | grep -m1 MetaX || echo unknown)"
  echo ""
} > "$SUMMARY"

run_section() {
  local title="$1"
  shift
  echo "## $title" | tee -a "$SUMMARY"
  echo '```' | tee -a "$SUMMARY"
  "$@" 2>&1 | tee -a "$SUMMARY" || echo "WARN: $title failed (exit $?)" | tee -a "$SUMMARY"
  echo '```' | tee -a "$SUMMARY"
  echo "" | tee -a "$SUMMARY"
}

cd "$REPO_DIR"

run_section "Pre-check — MTP head in checkpoint" \
  python scripts/check_mtp_head.py "${MODEL:-/data/models/Qwen3.6-27B-AWQ}" --json

run_section "Phase 2 — Operator micro-benchmark" \
  bash -c "bash scripts/run_op_bench.sh --seq-len 256 --json | tee '$LOG_ROOT/phase2_op_bench.json'"

run_section "Phase 2 — Decode profiler" \
  env PYTHONPATH=. python scripts/profile_decode.py --seq-len 256 --json

run_section "Phase 1 — Concurrent batch (1/4/8 req)" \
  bash scripts/run_phase1_concurrent_bench.sh

run_section "Phase 3 — MTP speculative" \
  bash scripts/run_phase3_mtp_bench.sh

run_section "Parse results" \
  python scripts/parse_bench_results.py "$LOG_ROOT/ALL_BENCH_SUMMARY.md" -o "$LOG_ROOT/PARSED_RESULTS.md"

run_section "Acceptance check" \
  bash -c "python scripts/bench_acceptance.py '$LOG_ROOT' --json -o '$LOG_ROOT/ACCEPTANCE.json' && python scripts/bench_acceptance.py '$LOG_ROOT' --markdown -o '$LOG_ROOT/ACCEPTANCE.md'"

python scripts/bench_acceptance.py "$LOG_ROOT" | tee -a "$SUMMARY"

echo "All benchmarks complete. Summary: $SUMMARY"
echo "Detailed logs:"
echo "  $LOG_ROOT/phase1/PHASE1_CONCURRENT_BENCH.md"
echo "  $LOG_ROOT/phase3/PHASE3_MTP_BENCH.md"
echo "  $LOG_ROOT/PARSED_RESULTS.md"

if [[ -x "$REPO_DIR/scripts/export_bench_bundle.sh" ]]; then
  bash "$REPO_DIR/scripts/export_bench_bundle.sh"
fi
