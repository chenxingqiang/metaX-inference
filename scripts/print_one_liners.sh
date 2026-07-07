#!/usr/bin/env bash
# Print copy-paste one-liners for MetaX server (no execution)
set -euo pipefail

cat <<'EOF'
# metaX-inference — MetaX C500 实机 One-Liners (main)

# 同步代码
curl -fsSL https://raw.githubusercontent.com/chenxingqiang/metaX-inference/main/scripts/sync_from_github.sh | bash

# 快速冒烟 ~5min
curl -fsSL https://raw.githubusercontent.com/chenxingqiang/metaX-inference/main/scripts/quick_smoke_metax.sh | bash

# 全套 benchmark
curl -fsSL https://raw.githubusercontent.com/chenxingqiang/metaX-inference/main/scripts/metax_paste_and_run.sh | bash

# 仅冒烟（FAST 模式）
FAST=1 curl -fsSL https://raw.githubusercontent.com/chenxingqiang/metaX-inference/main/scripts/metax_paste_and_run.sh | bash

# 生产 vLLM
/data/metaX-inference/scripts/serve_qwen36_metax.sh

# 下载结果
scp -P 32222 root+vm-Fcar5bmuV0AjJMTG@140.207.205.81:/tmp/metax-bench-bundle.tar.gz .

# 本地验收预览（有 baseline 即可）
python3 scripts/bench_acceptance.py . --markdown
EOF
