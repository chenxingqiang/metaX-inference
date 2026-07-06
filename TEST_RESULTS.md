# MetaX 实机测试记录

## 测试环境

| 项目 | 值 |
|------|-----|
| 日期 | 2026-07-06 |
| 服务器 | 140.207.205.81:32222 |
| GPU | MetaX C500（sGPU 配额 **32GB** VRAM，算力 50%） |
| MACA | 3.5.3.20 |
| 系统 | Ubuntu 20.04 / Linux 5.15 |
| Python | 3.10.10 (conda base) |
| vLLM | 0.17.0 + vllm_metax 0.17.0 |
| PyTorch | 2.8.0+metax3.5.3.9 |

## 环境检查 (E01–E05)

| 编号 | 结果 | 备注 |
|------|------|------|
| E01 | PASS | metax/maca 3.5.3 软件栈已安装 |
| E02 | **SKIP** | 未安装 `vulkan-sdk` / `vulkaninfo`（方案 A 前置缺失） |
| E03 | PASS | Python 3.10.10 |
| E04 | **SKIP** | 未安装 Unsloth（方案 B 使用预量化 AWQ，无需 Unsloth 在线量化） |
| E05 | PASS | mx-smi 可见 MetaX C500，32GB sGPU 显存配额 |

## 方案 A：Unsloth GGUF + llama.cpp (Vulkan)

| 编号 | 结果 | 备注 |
|------|------|------|
| A01–A05 | **未测** | 服务器无 Vulkan SDK、无 llama.cpp；需额外安装后补测 |

## 方案 B：量化模型 + MacaRT-vLLM

| 编号 | 结果 | 备注 |
|------|------|------|
| B01 | PASS | 使用 `QuantTrio/Qwen3.6-27B-AWQ`（21GB，8 shards） |
| B02 | PASS | `vllm serve` 启动成功，vllm_metax 插件激活 |
| B03 | **PASS** | `/v1/completions` 返回 200，生成正常 |
| B04 | 观察 | 平均生成吞吐约 **9.5 tokens/s**（单请求 smoke test） |
| B05 | PASS | 推理时显存 **28604/32000 MiB**，GPU 利用率 70% |

### 关键修复

Qwen3.6 的 `config.json` 中 `model_type` 为 `qwen3_5`，系统自带 `transformers 4.57.6` 无法识别，需升级：

```bash
pip install "git+https://github.com/huggingface/transformers.git"
# 实测版本: transformers 5.14.0.dev0
```

### 启动命令（实测可用）

```bash
source /opt/conda/etc/profile.d/conda.sh && conda activate base
export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib:$LD_LIBRARY_PATH
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download QuantTrio/Qwen3.6-27B-AWQ \
  --local-dir /data/models/Qwen3.6-27B-AWQ --local-dir-use-symlinks False

vllm serve /data/models/Qwen3.6-27B-AWQ \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --dtype auto \
  --trust-remote-code
```

### 推理样例

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"你好，我是","max_tokens":64,"temperature":0.7}'
```

返回 `finish_reason: length`，`completion_tokens: 64`，HTTP 200。

## 32GB 显存选型建议（本机）

| 模型 | 量化 | 结论 |
|------|------|------|
| Qwen3.6-27B-AWQ | INT4 AWQ | **推荐**，占用 ~28GB，可稳定运行 |
| Qwen3.6-35B-A3B | MoE | 可尝试，需实测显存 |
| Qwen3.6-27B BF16 | 全精度 | 不推荐，超出 32GB 配额 |

## 结论

- [x] **方案 B** 在 MetaX C500（32GB）上可用于 Qwen3.6-27B-AWQ 生产推理
- [ ] **方案 A** 待安装 Vulkan SDK + 编译 llama.cpp 后补测
- **阻塞项**：Qwen3.6 需 transformers ≥5.x dev；方案 A 缺 Vulkan 环境
