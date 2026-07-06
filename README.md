# metaX-inference

在沐曦显卡上使用 Unsloth 运行 Qwen3.6 推理的测试与脚本仓库。

## 文档

- **[AGENT.md](./AGENT.md)** — 完整测试方案（方案 A：GGUF + llama.cpp/Vulkan；方案 B：Unsloth 量化 + MacaRT-vLLM）、验收标准与结果记录模板

## 快速开始

```bash
# 环境检查（需在沐曦实机 + MXMACA 环境）
./scripts/test-env-check.sh

# 方案 A：设置模型与 llama-server 路径后冒烟测试
export GGUF_MODEL=/path/to/qwen3.6-27b-mtp-q4_k_m.gguf
export LLAMA_SERVER=/path/to/llama-server
./scripts/test-scheme-a.sh

# 方案 B：量化导出后启动 vLLM 冒烟测试
python scripts/quantize-qwen36.py --output ./qwen3.6-27b-4bit
export VLLM_MODEL=./qwen3.6-27b-4bit
./scripts/test-scheme-b.sh
```

详细步骤、模型选型与故障排查见 [AGENT.md](./AGENT.md)。
