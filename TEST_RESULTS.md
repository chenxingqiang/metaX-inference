# MetaX 实机测试记录

## 测试环境

| 项目 | 值 |
|------|-----|
| 日期 | 2026-07-06 ~ 2026-07-07 |
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

## 仓库状态

- **main** 已合并 PR #2（2026-07-07）：实机 A/B 验证 + metax_kernels + Phase 1/2/3 基准
- **Release [v0.1.0](https://github.com/chenxingqiang/metaX-inference/releases/tag/v0.1.0)** / [v0.1.1](https://github.com/chenxingqiang/metaX-inference/releases/tag/v0.1.1)
- **2026-07-07 SSH 全套 benchmark 已执行**（Cloud Agent 实机）

### Owner 实机 Checklist

- [x] `remote_run_all_benches.sh` 全套 benchmark（2026-07-07）
- [x] Phase 2 op_bench + Phase 1 并发 1/4/8
- [ ] Phase 3 MTP/ngram（启动失败，见下文 root cause + workaround）
- [ ] `ACCEPTANCE.md` 自动生成（Phase 3 阻塞部分指标）
- [ ] GitHub Issue #3 关闭

```bash
bash scripts/print_one_liners.sh   # 打印全部 one-liner
```

## 结论

- [x] **方案 A** GGUF + llama.cpp **功能验证 PASS**（CPU 回退；沐曦 Vulkan GPU 加速未打通）
- [x] **方案 B** 在 MetaX C500（32GB）上可用于 Qwen3.6-27B-AWQ 生产推理
- **阻塞项**：方案 A 沐曦 GPU 需 Vulkan ICD + SPIRV 编译链；方案 B 需 transformers ≥5.x dev

## Phase 2 算子 Baseline（MetaX C500，2026-07-07 SSH）

`PYTHONPATH=. python -m metax_kernels.bench.op_bench --seq-len 256 --json`

| Kernel | avg ms | 说明 |
|--------|--------|------|
| qwen36.fused_rope_rms (eager) | **0.522** | 最优 RoPE 路径，距目标 0.5ms 差 4% |
| qwen36.fused_rope_rms (compiled) | 0.543 | torch.compile |
| qwen36.fused_rope_rms (fused) | 0.525 | mcoplib stub 回退 |
| qwen36.fused_rope_rms (opt_eager) | 0.659 | 优化 eager |
| qwen36.gqa_attention:sdpa | **0.110** | PyTorch SDPA，当前 GQA 最快 |
| qwen36.gqa_attention:fused | 0.152 | flash_attn 路径（MACA 版） |
| qwen36.gqa_attention:eager | 0.228 | 纯 matmul softmax |
| qwen36.fused_mlp:eager | 1.104 | MLP baseline |
| qwen36.awq_gemm:eager | 0.159 | AWQ GEMM |

原始 JSON：`metax_kernels/bench/results_op_bench_c500.json`

### MTP head 检测（checkpoint）

- `has_mtp_head: true`（15 个 `mtp.*` 权重）
- AWQ 量化，`modules_to_not_convert` 含 `"mtp"` → draft acceptance 可能为 0%，需 BF16 MTP checkpoint 验证

## Phase 1 vLLM 并发 batch（MetaX C500，2026-07-07 SSH）

Prompt: `你好，请用一句话介绍你自己。` / max_tokens: 128

| 配置 | aggregate tok/s | per-req mean | 目标 |
|------|-----------------|--------------|------|
| 单请求 (c=1) | **7.29** | 7.29 | Phase 0 ≥ 9.5 **FAIL** |
| 并发 x4 | **12.80** | 3.28 | — |
| 并发 x8 | **21.09** | 2.69 | Phase 1 ≥ 40 **FAIL** |

观察：高 TTFT（~13–23s）因 Qwen3.6 thinking 块 + 并发 prefill 排队；并发 x8 总吞吐仅达目标 52%。

结果文件：`/data/metax-test-logs/phase1/PHASE1_CONCURRENT_BENCH.md`

## Phase 3 MTP speculative（MetaX C500，2026-07-07 SSH）

| 模式 | tok/s | 状态 |
|------|-------|------|
| Baseline（无 speculative） | **7.26** | PASS |
| MTP (`num_speculative_tokens=2`) | — | **启动失败** |
| N-gram fallback | — | **启动失败** |

### Root cause

vLLM 启动时在 CUDA graph capture 阶段触发 `vllm_metax` Triton autotuner：

```
torch.AcceleratorError: CUDA error: operation not permitted when stream is capturing
  maca_fused_recurrent_gated_delta_rule_fwd_kernel (Qwen3.6 linear_attn / GDN)
```

Speculative decode 路径与 MACA Triton autotune + cudagraph 不兼容。

### Workaround（v0.1.2+）

```bash
# Phase 3 bench 默认 DISABLE_CUDAGRAPH=1
bash scripts/run_phase3_mtp_bench.sh

# 生产 MTP serve
DISABLE_CUDAGRAPH=1 ENABLE_MTP=1 MTP_TOKENS=2 scripts/serve_qwen36_metax.sh
```

添加 `--compilation-config '{"cudagraph_mode":"none"}'` 禁用 cudagraph capture（仅 speculative 路径；baseline 不受影响）。重测待确认。

结果文件：`/data/metax-test-logs/phase3/PHASE3_MTP_BENCH.md`

## Phase 2 Decode profiler

`profile_decode.py` 在 PyTorch 2.8 上报 `FunctionEventAvg` 无 `cuda_time_total`（已修复为兼容 `device_time_total`）。

## Phase 1 vLLM 参数扫描（MetaX C500，2026-07-07）

| 配置 | tokens/s | elapsed (128 tok) |
|------|----------|-------------------|
| Baseline（默认） | **7.44** | 17.20s |
| Phase 1 tuned（chunked prefill + prefix cache + gpu-mem 0.92） | 7.35 | 17.42s |

单请求场景下 Phase 1 参数无明显提升；并发 batch 已测见上节。

## 验收目标（AGENT.md §12.5）

| 阶段 | 指标 | 目标 | 当前（2026-07-07 SSH） | 状态 |
|------|------|------|------------------------|------|
| Phase 0 | 单请求 tok/s | ≥ 9.5 | 7.29（peak 9.5 历史） | **FAIL** |
| Phase 1 | 并发 8 req | ≥ 40 | **21.09** | **FAIL** |
| Phase 2 op | fused_rope_rms | ≤ 0.5 ms | **0.522 ms** eager | **接近 FAIL** |
| Phase 3 | + MTP | ≥ 20 | 未测（启动失败） | **BLOCKED** |

实机跑完后自动验收：
```bash
python scripts/bench_acceptance.py /data/metax-test-logs
python scripts/bench_acceptance.py /data/metax-test-logs --markdown -o ACCEPTANCE.md
```

### 当前自动验收（2026-07-07 SSH）

| 指标 | 实测 | 目标 | 状态 |
|------|------|------|------|
| phase0_single_tok_s | 7.29 | ≥ 9.5 | FAIL |
| phase0_single_tok_s_peak | 9.5 | ≥ 9.5 | **PASS** |
| phase1_concurrent_8_tok_s | 21.09 | ≥ 40 | FAIL |
| phase3_mtp_tok_s | — | ≥ 20 | BLOCKED |
| fused_rope_rms_ms | 0.522 | ≤ 0.5 | FAIL (Δ 4%) |

数据来源：`configs/acceptance_baseline.json` + `metax_kernels/bench/results_op_bench_c500.json` + `/data/metax-test-logs/`
