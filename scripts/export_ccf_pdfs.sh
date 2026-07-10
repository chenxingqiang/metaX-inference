#!/usr/bin/env bash
# Export CCF submission PDFs from docs/CCF_*.md
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FONT="${CCF_PDF_FONT:-PingFang SC}"
for md in "$ROOT"/docs/CCF_*.md; do
  pdf="${md%.md}.pdf"
  pandoc "$md" -o "$pdf" \
    --pdf-engine=xelatex \
    -V "CJKmainfont=$FONT" \
    -V geometry:margin=2.5cm \
    -V fontsize=11pt \
    --toc --toc-depth=2
  echo "OK $pdf"
done
