#!/usr/bin/env bash
# Sync metaX-inference from GitHub (no scp/SSH from laptop required)
set -euo pipefail

REPO_DIR="${REPO_DIR:-/data/metaX-inference}"
BRANCH="${BRANCH:-main}"
TMP="/tmp/metaX-inference-archive"

echo "Syncing chenxingqiang/metaX-inference@${BRANCH} -> $REPO_DIR"

rm -rf "$TMP"
mkdir -p "$TMP"
curl -fsSL "https://github.com/chenxingqiang/metaX-inference/archive/refs/heads/${BRANCH}.tar.gz" \
  | tar -xz -C "$TMP" --strip-components=1

mkdir -p "$REPO_DIR"
rsync -a --delete "$TMP/" "$REPO_DIR/"
rm -rf "$TMP"

echo "Sync complete: $REPO_DIR ($(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo no-git))"
echo "Validate: cd $REPO_DIR && bash scripts/validate_repo.sh"
