#!/usr/bin/env bash
# Bundle benchmark logs for download / upload to issue tracker
set -euo pipefail

LOG_ROOT="${LOG_ROOT:-/data/metax-test-logs}"
OUT="${OUT:-/tmp/metax-bench-bundle.tar.gz}"

echo "Bundling $LOG_ROOT -> $OUT"
tar -czf "$OUT" -C "$(dirname "$LOG_ROOT")" "$(basename "$LOG_ROOT")"
echo "Created: $OUT"
echo "Size: $(du -h "$OUT" | cut -f1)"
echo ""
echo "Download from server:"
echo "  scp -P 32222 root+vm-Fcar5bmuV0AjJMTG@140.207.205.81:$OUT ."
