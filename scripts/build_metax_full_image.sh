#!/usr/bin/env bash
# Build MetaX full stack Docker images:
#   metax-openclaw:local  (TrustClaw + TRA UI)
#   metax-vllm:local      (vLLM wrapper, uses host MACA/conda mounts)
#   metax-trustclaw:local (Gateway entrypoint + vLLM config)
#
# Usage:
#   ./scripts/build_metax_full_image.sh
#   ./scripts/build_metax_full_image.sh --save   # export offline bundle
#   TRUSTCLAW_DIR=/path/to/TrustClaw ./scripts/build_metax_full_image.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRUSTCLAW_DIR="${TRUSTCLAW_DIR:-$REPO_ROOT/../TrustClaw}"
SAVE_BUNDLE=0
DOCKER_USER="${DOCKER_USER:-chenxingqiang}"

for arg in "$@"; do
  case "$arg" in
    --save) SAVE_BUNDLE=1 ;;
    -h|--help)
      echo "Usage: $0 [--save]"
      exit 0
      ;;
  esac
done

if ! command -v docker &>/dev/null; then
  echo "ERROR: docker not found"
  exit 1
fi

if [[ ! -f "$TRUSTCLAW_DIR/Dockerfile" ]]; then
  echo "ERROR: TrustClaw not found at $TRUSTCLAW_DIR"
  echo "Clone: git clone https://github.com/chenxingqiang/TrustClaw.git $TRUSTCLAW_DIR"
  exit 1
fi

echo "=== [1/3] Build metax-openclaw:local from TrustClaw ==="
if [[ -f "$TRUSTCLAW_DIR/pnpm-workspace.yaml" ]]; then
  sed -i 's/^minimumReleaseAge: 2880/minimumReleaseAge: 0/' "$TRUSTCLAW_DIR/pnpm-workspace.yaml" 2>/dev/null || \
    sed -i '' 's/^minimumReleaseAge: 2880/minimumReleaseAge: 0/' "$TRUSTCLAW_DIR/pnpm-workspace.yaml" 2>/dev/null || true
fi

docker build \
  -t metax-openclaw:local \
  --build-arg OPENCLAW_TRUSTCLAW_UI=1 \
  -f "$TRUSTCLAW_DIR/Dockerfile" \
  "$TRUSTCLAW_DIR"

echo "=== [2/3] Build metax-vllm:local ==="
docker build \
  -t metax-vllm:local \
  -f "$REPO_ROOT/docker/metax-full/Dockerfile.vllm" \
  "$REPO_ROOT"

echo "=== [3/3] Build metax-trustclaw:local ==="
docker build \
  -t metax-trustclaw:local \
  --build-arg OPENCLAW_IMAGE=metax-openclaw:local \
  -f "$REPO_ROOT/docker/metax-full/Dockerfile.trustclaw" \
  "$REPO_ROOT"

echo ""
echo "=== Build complete ==="
docker images | grep -E 'metax-(openclaw|vllm|trustclaw)' || true
echo ""
echo "Run:"
echo "  cd $REPO_ROOT/docker/metax-full"
echo "  cp app.env.example app.env   # edit tokens"
echo "  docker compose up -d"
echo ""
echo "Access:"
echo "  TrustClaw: http://127.0.0.1:19001/?token=<OPENCLAW_GATEWAY_TOKEN>"
echo "  vLLM API:  http://127.0.0.1:8000/v1"

if [[ "$SAVE_BUNDLE" -eq 1 ]]; then
  OUT="$REPO_ROOT/docker/metax-full/dist/metax-full-images.tar"
  mkdir -p "$(dirname "$OUT")"
  echo "Saving bundle to $OUT ..."
  docker save -o "$OUT" \
    metax-openclaw:local \
    metax-vllm:local \
    metax-trustclaw:local
  ls -lh "$OUT"
  echo "Offline load: docker load -i metax-full-images.tar"
fi
