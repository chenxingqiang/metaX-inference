#!/bin/bash
# Scheme A: Unsloth GGUF + llama.cpp (Vulkan) on MetaX — AGENT.md §4
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/data/huggingface_home}"
export HF_HUB_CACHE="$HF_HOME/hub"
export PATH="/opt/conda/bin:$PATH"

LOG_DIR=/data/metax-test-logs
RESULT_FILE="$LOG_DIR/SCHEME_A_RESULTS.md"
WORK_DIR=/data/scheme-a
LLAMA_DIR="$WORK_DIR/llama.cpp"
GGUF_DIR="$WORK_DIR/models"
GGUF_FILE="${GGUF_FILE:-$GGUF_DIR/qwen3.6-27b-mtp-q4_k_m.gguf}"
GGUF_REPO="${GGUF_REPO:-unsloth/Qwen3.6-27B-MTP-GGUF}"
GGUF_FILENAME="${GGUF_FILENAME:-Qwen3.6-27B-Q4_K_M.gguf}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"

mkdir -p "$LOG_DIR" "$GGUF_DIR" "$WORK_DIR"

{
  echo "# Scheme A Test Results — Unsloth GGUF + llama.cpp (Vulkan)"
  echo ""
  echo "- Date: $(date -Iseconds)"
  echo "- GPU: MetaX C500"
  echo "- Model: $GGUF_REPO / $GGUF_FILENAME"
  echo ""
} > "$RESULT_FILE"

log() { echo "$1" | tee -a "$RESULT_FILE"; }

# --- A01: Vulkan ---
log "=== A01 Vulkan check ==="
if ! command -v vulkaninfo &>/dev/null; then
  log "Installing vulkan-sdk..."
  apt-get update -qq >> "$LOG_DIR/scheme-a-apt.log" 2>&1 || true
  apt-get install -y -qq vulkan-tools libvulkan-dev vulkan-validationlayers-dev spirv-tools glslang-tools cmake build-essential git wget curl >> "$LOG_DIR/scheme-a-apt.log" 2>&1 || {
    # LunarG SDK fallback (Ubuntu 22.04 jammy repo on 20.04 may need focal)
    wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | apt-key add - >> "$LOG_DIR/scheme-a-apt.log" 2>&1 || true
    wget -qO /etc/apt/sources.list.d/lunarg-vulkan-jammy.list https://packages.lunarg.com/vulkan/1.3.280/lunarg-vulkan-1.3.280-jammy.list 2>>"$LOG_DIR/scheme-a-apt.log" || true
    apt-get update -qq >> "$LOG_DIR/scheme-a-apt.log" 2>&1 || true
    apt-get install -y -qq vulkan-sdk >> "$LOG_DIR/scheme-a-apt.log" 2>&1 || true
  }
fi

VULKAN_OK=0
if command -v vulkaninfo &>/dev/null; then
  vulkaninfo --summary 2>"$LOG_DIR/vulkaninfo.log" | head -40 | tee -a "$RESULT_FILE" || true
  if grep -qiE 'GPU|device|MetaX|muxi|llvmpipe' "$LOG_DIR/vulkaninfo.log" 2>/dev/null; then
    VULKAN_OK=1
    log "PASS A01: vulkaninfo available"
  else
    log "WARN A01: vulkaninfo ran but GPU device unclear — see $LOG_DIR/vulkaninfo.log"
    VULKAN_OK=1
  fi
else
  log "FAIL A01: vulkaninfo not found after install attempt"
  exit 1
fi

# --- Build llama.cpp ---
log "=== Build llama.cpp (Vulkan) ==="
if [[ ! -x "$LLAMA_DIR/build/bin/llama-server" ]]; then
  rm -rf "$LLAMA_DIR"
  mkdir -p "$WORK_DIR"
  clone_ok=0
  for url in \
    "https://ghproxy.net/https://github.com/ggerganov/llama.cpp/archive/refs/heads/master.tar.gz" \
    "https://mirror.ghproxy.com/https://github.com/ggerganov/llama.cpp/archive/refs/heads/master.tar.gz" \
    "https://github.com/ggerganov/llama.cpp/archive/refs/heads/master.tar.gz"; do
    log "Downloading llama.cpp tarball from $url ..."
    if wget -q -O "$WORK_DIR/llama.cpp.tar.gz" "$url" >> "$LOG_DIR/scheme-a-build.log" 2>&1; then
      tar -xzf "$WORK_DIR/llama.cpp.tar.gz" -C "$WORK_DIR" >> "$LOG_DIR/scheme-a-build.log" 2>&1
      mv "$WORK_DIR"/llama.cpp-* "$LLAMA_DIR" 2>/dev/null || mv "$WORK_DIR/llama.cpp-master" "$LLAMA_DIR" 2>/dev/null || true
      if [[ -f "$LLAMA_DIR/CMakeLists.txt" ]]; then clone_ok=1; break; fi
    fi
  done
  if [[ "$clone_ok" -ne 1 ]]; then
    for repo in \
      "https://gitclone.com/github.com/ggerganov/llama.cpp" \
      "https://github.com/ggerganov/llama.cpp"; do
      log "Fallback git clone from $repo ..."
      if git clone --depth 1 "$repo" "$LLAMA_DIR" >> "$LOG_DIR/scheme-a-build.log" 2>&1; then
        clone_ok=1; break
      fi
    done
  fi
  if [[ "$clone_ok" -ne 1 ]]; then
    log "FAIL: cannot fetch llama.cpp from any mirror"
    tail -20 "$LOG_DIR/scheme-a-build.log" | tee -a "$RESULT_FILE" || true
    exit 1
  fi
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release >> "$LOG_DIR/scheme-a-build.log" 2>&1
  cmake --build "$LLAMA_DIR/build" --config Release -j"$(nproc)" --target llama-server llama-cli >> "$LOG_DIR/scheme-a-build.log" 2>&1
fi
LLAMA_SERVER="$LLAMA_DIR/build/bin/llama-server"
if [[ ! -x "$LLAMA_SERVER" ]]; then
  LLAMA_SERVER=$(find "$LLAMA_DIR/build" -name 'llama-server' -type f 2>/dev/null | head -1)
fi
if [[ -z "$LLAMA_SERVER" || ! -x "$LLAMA_SERVER" ]]; then
  log "FAIL A01: llama-server build failed"
  tail -30 "$LOG_DIR/scheme-a-build.log" | tee -a "$RESULT_FILE" || true
  exit 1
fi
log "PASS A01: llama-server built at $LLAMA_SERVER"

# --- Download GGUF ---
log "=== Download GGUF ==="
if [[ ! -f "$GGUF_FILE" ]]; then
  if command -v hf &>/dev/null; then
    hf download "$GGUF_REPO" "$GGUF_FILENAME" --local-dir "$GGUF_DIR" \
      >> "$LOG_DIR/scheme-a-download.log" 2>&1
  else
    huggingface-cli download "$GGUF_REPO" "$GGUF_FILENAME" \
      --local-dir "$GGUF_DIR" --local-dir-use-symlinks False \
      >> "$LOG_DIR/scheme-a-download.log" 2>&1
  fi
  # huggingface-cli may place file directly in GGUF_DIR
  if [[ ! -f "$GGUF_FILE" ]] && [[ -f "$GGUF_DIR/$GGUF_FILENAME" ]]; then
    GGUF_FILE="$GGUF_DIR/$GGUF_FILENAME"
  fi
fi
if [[ ! -f "$GGUF_FILE" ]]; then
  log "FAIL: GGUF not found at $GGUF_FILE"
  ls -la "$GGUF_DIR" | tee -a "$RESULT_FILE" || true
  exit 1
fi
ls -lh "$GGUF_FILE" | tee -a "$RESULT_FILE"
log "PASS B01-equiv: GGUF ready"

# --- Start llama-server ---
log "=== Start llama-server ==="
pkill -f "llama-server" 2>/dev/null || true
sleep 2

nohup "$LLAMA_SERVER" \
  -m "$GGUF_FILE" \
  -ngl 99 \
  -c 8192 \
  --host "$HOST" \
  --port "$PORT" \
  --spec-type mtp --spec-draft-n-max 3 \
  > "$LOG_DIR/llama-server.log" 2>&1 &
LLAMA_PID=$!
log "llama-server PID=$LLAMA_PID"

READY=0
for i in $(seq 1 120); do
  if curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1 || \
     curl -sf "http://${HOST}:${PORT}/" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
    log "FAIL A02: llama-server exited early"
    tail -50 "$LOG_DIR/llama-server.log" | tee -a "$RESULT_FILE" || true
    exit 1
  fi
  sleep 3
done

if [[ "$READY" -ne 1 ]]; then
  log "FAIL A02: server not ready"
  tail -50 "$LOG_DIR/llama-server.log" | tee -a "$RESULT_FILE" || true
  exit 1
fi

if grep -qiE 'vulkan|gpu|offload' "$LOG_DIR/llama-server.log" 2>/dev/null; then
  grep -iE 'vulkan|gpu|offload|layer' "$LOG_DIR/llama-server.log" 2>/dev/null | head -15 | tee -a "$RESULT_FILE" || true
fi
log "PASS A02: llama-server running"

# --- Inference ---
log "=== Inference test (A03) ==="
RESP=$(curl -sf "http://${HOST}:${PORT}/completion" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"你好，我是","n_predict":64,"temperature":0.7}' || true)

echo "$RESP" | tee -a "$RESULT_FILE"

if echo "$RESP" | grep -qE 'content|generation|text'; then
  log "PASS A03: completion endpoint OK"
else
  log "FAIL A03: unexpected response"
  exit 1
fi

mx-smi 2>/dev/null | head -25 | tee -a "$RESULT_FILE" || true
log "Scheme A test completed."

kill "$LLAMA_PID" 2>/dev/null || true
