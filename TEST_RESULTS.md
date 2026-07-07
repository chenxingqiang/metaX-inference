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
| E02 | **PARTIAL** | 已装 `vulkan-tools`，但 `/usr/share/vulkan/icd.d/` 无沐曦 ICD，仅 Mesa/Intel |
| E03 | PASS | Python 3.10.10 |
| E04 | **SKIP** | 未安装 Unsloth（方案 B 使用预量化 AWQ，无需 Unsloth 在线量化） |
| E05 | PASS | mx-smi 可见 MetaX C500，32GB sGPU 显存配额 |

## 方案 A：Unsloth GGUF + llama.cpp (Vulkan)

| 编号 | 结果 | 备注 |
|------|------|------|
| A01 | **PARTIAL** | llama.cpp 编译成功（**CPU 回退**）；`-DGGML_VULKAN=ON` 因缺 `SPIRV-Headers` 失败 |
| A02 | **PASS** | `llama-server` 启动，模型加载 ~84s |
| A03 | **PASS** | `/completion` 返回中文续写正常 |
| A04 | **FAIL** | 未启用 Vulkan/GPU；`-ngl` 被忽略，**GPU 利用率 0%** |
| A05 | PASS | `-c 2048` 上下文可用 |

### 模型与命令（实测）

```bash
# 下载 GGUF（注意文件名大小写）
hf download unsloth/Qwen3.6-27B-MTP-GGUF Qwen3.6-27B-Q4_K_M.gguf \
  --local-dir /data/scheme-a/models

# llama-server（当前为 CPU 编译版）
/data/scheme-a/llama.cpp/build/bin/llama-server \
  -m /data/scheme-a/models/Qwen3.6-27B-Q4_K_M.gguf \
  -ngl 0 -c 2048 -t 32 --host 127.0.0.1 --port 8080
```

### 性能（CPU 回退）

| 指标 | 值 |
|------|-----|
| 模型 | `Qwen3.6-27B-Q4_K_M.gguf`（16GB） |
| 加载时间 | ~84s |
| 生成吞吐 | **~0.56 tokens/s**（CPU，32 线程） |
| 显存 | GPU **未使用**（826 MiB 空闲） |

### 方案 A 阻塞项（沐曦 GPU 加速）

1. **无沐曦 Vulkan ICD**：`/opt/maca` 下未发现 `*vulkan*`，系统 icd 仅 Intel/LVP/Radeon
2. **SPIRV-Headers 缺失**：Ubuntu apt 包不足以编译 `GGML_VULKAN`
3. 需沐曦官方 Vulkan 驱动 + LunarG SDK 完整安装后，方可 `-ngl 99` 走 GPU

### 推理样例响应

Prompt: `你好，我是` → 返回 `小雅。...你好，小雅！很高兴认识你。`（32 tokens，HTTP 200）

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

- [x] **方案 A** GGUF + llama.cpp **功能验证 PASS**（CPU 回退；沐曦 Vulkan GPU 加速未打通）
- [x] **方案 B** 在 MetaX C500（32GB）上可用于 Qwen3.6-27B-AWQ 生产推理
- **阻塞项**：方案 A 沐曦 GPU 需 Vulkan ICD + SPIRV 编译链；方案 B 需 transformers ≥5.x dev

## Phase 2 算子 Baseline（MetaX C500，2026-07-07）

`PYTHONPATH=. python -m metax_kernels.bench.op_bench --seq-len 256 --json`

| Kernel | avg ms | 说明 |
|--------|--------|------|
| qwen36.fused_rope_rms (eager) | **0.94** | RoPE+RMSNorm+QKV baseline，待 mcoplib 融合 |
| qwen36.gqa_attention:sdpa | **0.12** | PyTorch SDPA，当前 GQA 最快 |
| qwen36.gqa_attention:fused | 0.16 | flash_attn 路径（MACA 版） |
| qwen36.gqa_attention:eager | 0.25 | 纯 matmul softmax |

原始 JSON：`metax_kernels/bench/results_op_bench_c500.json`

## Phase 1 vLLM 参数扫描（MetaX C500，2026-07-07）

| 配置 | tokens/s | elapsed (128 tok) |
|------|----------|-------------------|
| Baseline（默认） | **7.44** | 17.20s |
| Phase 1 tuned（chunked prefill + prefix cache + gpu-mem 0.92） | 7.35 | 17.42s |

单请求场景下 Phase 1 参数无明显提升；并发 batch 测试待补。Qwen3.6 默认输出含 thinking 块，影响 tok/s 对比。
