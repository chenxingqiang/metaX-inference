# AGENT.md — 沐曦显卡 Unsloth + Qwen3.6 推理测试

> 本文档记录 `metaX-inference` 仓库中，在沐曦显卡上使用 Unsloth 运行 Qwen3.6 推理的测试方案、执行步骤与验收标准。

## 1. 测试目标

验证在沐曦 GPU 上通过以下两条路径完成 Qwen3.6 推理：

| 方案 | 路径 | 适用场景 |
|------|------|----------|
| **A** | Unsloth GGUF + llama.cpp (Vulkan) | 快速验证、本地调试 |
| **B** | Unsloth 量化 + 沐曦 MacaRT-vLLM | 生产部署、高吞吐 |

## 2. 硬件与软件前提

### 2.1 硬件

- **显卡**：曦云 C 系列 / 曦思系列
- **显存建议**：
  - Qwen3.6-7B / 14B：< 24GB
  - Qwen3.6-27B：≥ 18GB（推荐 ≥ 24GB）
  - Qwen3.6-35B-A3B (MoE)：≥ 22GB（推荐 ≥ 32GB）

### 2.2 软件栈

- **MXMACA**：v3.3.0+（PyTorch 兼容 + Vulkan 驱动）
- **系统**：Linux（Ubuntu 20.04 / 22.04），Windows 需 WSL2
- **Python**：3.10（推荐）
- **Vulkan SDK**：llama.cpp Vulkan 后端必需

### 2.3 环境安装

```bash
# 沐曦驱动与基础库
sudo apt install metax-maca-driver metax-maca-runtime metax-maca-dev

# Vulkan SDK
wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo apt-key add -
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.3.280-jammy.list \
  https://packages.lunarg.com/vulkan/1.3.280/lunarg-vulkan-1.3.280-jammy.list
sudo apt update && sudo apt install vulkan-sdk

# Python 环境
conda create -n unsloth-meta python=3.10 -y
conda activate unsloth-meta
pip install unsloth torch==2.2.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate sentencepiece
```

## 3. 测试前检查

在沐曦机器上运行仓库内环境检查脚本：

```bash
./scripts/test-env-check.sh
```

### 3.1 检查项与通过标准

| 编号 | 检查项 | 命令 / 方法 | 通过标准 |
|------|--------|-------------|----------|
| E01 | MXMACA 驱动 | `dpkg -l \| grep metax-maca` | 已安装 driver / runtime / dev |
| E02 | Vulkan 设备 | `vulkaninfo \| grep -i device` | 能识别沐曦 GPU |
| E03 | Python 环境 | `python --version` | 3.10.x |
| E04 | Unsloth 可用 | `python -c "import unsloth"` | 无 ImportError |
| E05 | 显存容量 | 系统工具 / `mx-smi` | 满足所选模型下限 |

## 4. 方案 A 测试：Unsloth GGUF + llama.cpp (Vulkan)

### 4.1 原理

Unsloth 提供 Qwen3.6 的 **MTP 优化 GGUF 量化模型**；llama.cpp 通过 Vulkan 后端调用沐曦 GPU 加速，无需额外适配层。

### 4.2 测试步骤

```bash
# 1. 下载模型（Q4_K_M 量化，速度与精度平衡）
git clone https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF
cd Qwen3.6-27B-MTP-GGUF
wget https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/resolve/main/qwen3.6-27b-mtp-q4_k_m.gguf

# 2. 编译 llama.cpp（启用 Vulkan）
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && mkdir build && cd build
cmake -DGGML_VULKAN=ON ..
make -j$(nproc)

# 3. 启动推理服务
./llama-server \
  -m ../../Qwen3.6-27B-MTP-GGUF/qwen3.6-27b-mtp-q4_k_m.gguf \
  -ngl 99 \
  -c 8192 \
  --spec-type mtp --spec-draft-n-max 3

# 4. 功能验证
curl http://localhost:8080/completion \
  -d '{"prompt":"你好，我是","n_predict":128}'
```

也可使用仓库脚本（需先设置 `GGUF_MODEL` 与 `LLAMA_SERVER` 环境变量）：

```bash
export GGUF_MODEL=/path/to/qwen3.6-27b-mtp-q4_k_m.gguf
export LLAMA_SERVER=/path/to/llama.cpp/build/bin/llama-server
./scripts/test-scheme-a.sh
```

### 4.3 验收标准

| 编号 | 测试项 | 预期结果 |
|------|--------|----------|
| A01 | 服务启动 | `llama-server` 无 Vulkan 初始化错误 |
| A02 | GPU 卸载 | 日志显示 GPU 层已加载（`-ngl 99`） |
| A03 | 补全接口 | `curl` 返回合法 JSON，`content` 含中文续写 |
| A04 | MTP 加速 | 启用 `--spec-type mtp` 后 tokens/s 高于未启用（约 1.5–2×） |
| A05 | 上下文窗口 | `-c 8192` 下长 prompt 不 OOM |

### 4.4 关键参数

- `-ngl 99`：尽可能多地将层卸载到 GPU
- `-c 8192`：上下文长度
- `--spec-type mtp --spec-draft-n-max 3`：MTP 推测解码加速

## 5. 方案 B 测试：Unsloth 量化 + MacaRT-vLLM

### 5.1 原理

Unsloth 对 Qwen3.6 做 4-bit 量化后，由沐曦适配的 **MacaRT-vLLM**（vLLM 后端插件）提供更高吞吐与更低延迟。

### 5.2 测试步骤

```bash
# 1. 安装 MacaRT-vLLM
pip install macart-vllm-metax==0.11.0

# 2. Unsloth 量化并导出（见 scripts/quantize-qwen36.py）
python scripts/quantize-qwen36.py \
  --model Qwen/Qwen3.6-27B \
  --output ./qwen3.6-27b-4bit

# 3. 启动 vLLM API 服务
python -m vllm.entrypoints.api_server \
  --model ./qwen3.6-27b-4bit \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --dtype auto \
  --trust-remote-code

# 4. 功能验证
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-27b-4bit","prompt":"你好，我是","max_tokens":128}'
```

或使用脚本：

```bash
export VLLM_MODEL=./qwen3.6-27b-4bit
./scripts/test-scheme-b.sh
```

### 5.3 验收标准

| 编号 | 测试项 | 预期结果 |
|------|--------|----------|
| B01 | 量化导出 | `save_pretrained_merged` 生成 safetensors 与 tokenizer |
| B02 | vLLM 启动 | API server 监听 8000，日志显示 GPU 被使用 |
| B03 | 补全接口 | OpenAI 兼容 `/v1/completions` 返回正常文本 |
| B04 | 吞吐 | 批量请求下 tokens/s 优于方案 A（同硬件同模型） |
| B05 | 显存 | 4-bit 27B 在 ≥18GB 显存下可稳定运行 |

## 6. 模型与量化选型

### 6.1 按显存选模型

| 显存 | 推荐模型 | 量化档位 |
|------|----------|----------|
| 8–16GB | Qwen3.6-7B | Q4_K_M / Unsloth Dynamic 2.0 |
| 16–24GB | Qwen3.6-14B | Q4_K_M |
| 24–32GB | Qwen3.6-27B | Q4_K_M（方案 A）/ 4-bit（方案 B） |
| ≥32GB | Qwen3.6-35B-A3B | Q4_K_M 或 Q6_K_XL |

### 6.2 量化策略

- **Q4_K_M**：速度与精度平衡（方案 A 默认）
- **Q6_K_XL**：更高精度，显存占用更大
- **Unsloth Dynamic 2.0**：极致压缩，适合显存紧张
- **4-bit（方案 B）**：配合 vLLM 生产部署

## 7. 性能优化与注意事项

1. **关闭显存压缩**，确保 MXMACA 后端正常加载。
2. **MTP**：方案 A 务必启用 `--spec-type mtp`。
3. **Unsloth 训练**：目前仅支持 NVIDIA；**推理**可通过 Vulkan / 沐曦后端完成。
4. **显存不足**：降低量化（如 Q2_K），或 llama.cpp 使用 `--cpu-moe` 做 CPU+GPU 混合。

## 8. 常见问题

| 现象 | 排查 | 处理 |
|------|------|------|
| Vulkan 初始化失败 | `vulkaninfo` | 重装 MXMACA 驱动与 Vulkan SDK |
| 显存 OOM | 降低 `-ngl` 或换更小量化 | Q2_K / 更小模型 |
| 推理慢 | 检查 `-ngl`、GPU 利用率 | 确保层在 GPU 上 |
| vLLM 无法加载沐曦 | 日志与 `macart-vllm-metax` 版本 | 使用 `==0.11.0` 并与 MXMACA 版本匹配 |

## 9. 测试结果

**实机测试记录见 [TEST_RESULTS.md](./TEST_RESULTS.md)**（MetaX C500 / 32GB / 2026-07-06）。

### 实机验证摘要

| 方案 | 状态 | 说明 |
|------|------|------|
| A（GGUF） | **PASS（CPU）** | `Qwen3.6-27B-Q4_K_M.gguf` + llama-server；GPU Vulkan 未启用 |
| B（vLLM） | **PASS** | `QuantTrio/Qwen3.6-27B-AWQ` + vllm_metax 0.17.0，~9.5 tok/s |

### Qwen3.6 特别注意

`Qwen3.6` 在 `config.json` 中声明 `model_type: qwen3_5`，**transformers 4.57.x 无法加载**。实机需：

```bash
pip install "git+https://github.com/huggingface/transformers.git"
```

### 32GB 显存（MetaX C500 sGPU）推荐

- **模型**：`QuantTrio/Qwen3.6-27B-AWQ`（INT4，推理占用约 28GB）
- **参数**：`--max-model-len 8192`，`--tensor-parallel-size 1`
- **备选**：`Qwen/Qwen3.6-35B-A3B`（MoE，需单独实测）

### 结果记录模板

在沐曦实机上执行后，可将结果追加到 [TEST_RESULTS.md](./TEST_RESULTS.md)：

```markdown
## 测试执行记录

- **日期**：
- **机器**：显卡型号 / 显存 / MXMACA 版本 / OS
- **执行人**：

### 环境检查 (E01–E05)

| 编号 | 结果 | 备注 |
|------|------|------|
| E01  | PASS/FAIL | |
| ...  | | |

### 方案 A (A01–A05)

| 编号 | 结果 | tokens/s | 备注 |
|------|------|----------|------|
| A01  | | | |
| ...  | | | |

### 方案 B (B01–B05)

| 编号 | 结果 | tokens/s | 备注 |
|------|------|----------|------|
| B01  | | | |
| ...  | | | |

### 结论

- [ ] 方案 A 可用于快速验证
- [ ] 方案 B 可用于生产部署
- **阻塞问题**：
```

## 10. 仓库脚本索引

| 脚本 | 用途 |
|------|------|
| `scripts/test-env-check.sh` | 测试前环境检查 (E01–E05) |
| `scripts/test-scheme-a.sh` | 方案 A 自动化冒烟测试 |
| `scripts/test-scheme-b.sh` | 方案 B 自动化冒烟测试 |
| `scripts/quantize-qwen36.py` | Unsloth 4-bit 量化并导出 |
| `scripts/remote_test_scheme_b.sh` | 实机方案 B 端到端测试（含模型下载） |
| `scripts/remote_test_scheme_a.sh` | 实机方案 A 端到端测试（GGUF + llama-server） |
| `scripts/remote_test_vllm_fix.sh` | 升级 transformers 后启动 vLLM 并验证 |

## 11. 总结

- **快速验证**：优先 **方案 A**（Unsloth GGUF + llama.cpp / Vulkan）。
- **生产部署**：采用 **方案 B**（Unsloth 量化 + MacaRT-vLLM）。
- 核心依赖沐曦 MXMACA 的 Vulkan 与 PyTorch 兼容能力；按本文档与脚本在实机上执行即可完成测试闭环。

---

*实机测试结果见 [TEST_RESULTS.md](./TEST_RESULTS.md)；远程冒烟脚本见 `scripts/remote_test_*.sh`。*
