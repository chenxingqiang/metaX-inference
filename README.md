# metaX-inference

在沐曦显卡上使用 Unsloth / vLLM 运行 Qwen3.6 推理的测试、内核与基准仓库。

## 文档

- **[AGENT.md](./AGENT.md)** — 测试方案 + **§12 MACA 最佳推理架构设计**
- **[TEST_RESULTS.md](./TEST_RESULTS.md)** — 实机 A/B 测试结果

## 快速开始

```bash
# 本地验证（部署前）
./scripts/validate_repo.sh

# 生产 vLLM 启动
./scripts/serve_qwen36_metax.sh
METAX_KERNELS=1 ENABLE_MTP=1 ./scripts/serve_qwen36_metax.sh

# 环境检查（沐曦实机）
./scripts/test-env-check.sh

# 方案 B：vLLM 冒烟
export VLLM_MODEL=/data/models/Qwen3.6-27B-AWQ
./scripts/test-scheme-b.sh

# Phase 1：vLLM 参数扫描 + tok/s
./scripts/run_phase1_bench.sh

# Phase 1：并发 batch（1/4/8 req，目标 >40 tok/s @ 8 req）
./scripts/run_phase1_concurrent_bench.sh

# Phase 3：MTP speculative（目标 >20 tok/s 等效）
./scripts/run_phase3_mtp_bench.sh

# 实机一键全套基准
./scripts/remote_run_all_benches.sh

# SSH 登录沐曦后直接粘贴（curl | bash）
curl -fsSL "https://raw.githubusercontent.com/chenxingqiang/metaX-inference/main/scripts/metax_paste_and_run.sh" | bash

# Phase 2：算子 micro-benchmark（需 MACA PyTorch）
PYTHONPATH=. ./scripts/run_op_bench.sh --seq-len 512 --json

# Decode 热点 profiling
PYTHONPATH=. python scripts/profile_decode.py --seq-len 256 --json

# 单元测试 + 验收预览（含 baseline）
python -m unittest discover -s tests -v
python scripts/bench_acceptance.py . --markdown

# 可编辑安装
pip install -e .

# vLLM with metax_kernels auto-load
METAX_KERNELS=1 METAX_KERNEL_IMPL=fused vllm serve /data/models/Qwen3.6-27B-AWQ ...
```

## 包结构

```text
metax_kernels/
  qwen36/               # fused RoPE, GQA, AWQ GEMM, SwiGLU MLP
  mcoplib_bridge.py     # runtime mcoplib wiring
engine/vllm_metax_plugin/  # vLLM CustomOp + METAX_KERNELS=1 loader
configs/
scripts/bench_qwen36.py
tests/
```

详细步骤、模型选型与故障排查见 [AGENT.md](./AGENT.md)。
