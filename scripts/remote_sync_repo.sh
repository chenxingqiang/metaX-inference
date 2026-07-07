#!/usr/bin/env bash
# Sync metaX-inference repo to MetaX server via tarball (no git required on remote)
set -euo pipefail

REMOTE="${REMOTE:-root+vm-Fcar5bmuV0AjJMTG@140.207.205.81}"
PORT="${PORT:-32222}"
DEST="${DEST:-/data/metaX-inference}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARBALL="/tmp/metaX-inference-sync.tar.gz"

echo "Packaging $REPO_ROOT ..."
tar -czf "$TARBALL" \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  -C "$REPO_ROOT" .

echo "Uploading to ${REMOTE}:${DEST} ..."
ssh -p "$PORT" "$REMOTE" "mkdir -p '$DEST'"
scp -P "$PORT" "$TARBALL" "${REMOTE}:/tmp/metaX-inference-sync.tar.gz"
ssh -p "$PORT" "$REMOTE" "tar -xzf /tmp/metaX-inference-sync.tar.gz -C '$DEST' && rm -f /tmp/metaX-inference-sync.tar.gz"

echo "Sync complete: $DEST"
echo "Run on remote:"
echo "  cd $DEST && bash scripts/run_phase1_concurrent_bench.sh"
