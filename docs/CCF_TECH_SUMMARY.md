# MetaX-TrustClaw 技术文档摘要

**第八届 CCF 开源创新大赛 · 初赛提交材料**

---

## 1. 系统架构

### 1.1 三层架构

| 层级 | 组件 | 职责 |
|------|------|------|
| 应用层 | TrustClaw Gateway + trustclaw-tra | Agent Pack 编排、consent 策略、证据链、TRA Console |
| 推理层 | vLLM + vllm_metax + metax_kernels | Qwen3.6 本地推理、自定义算子、OpenAI 兼容 API |
| 算力层 | MetaX MACA (C500) | GPU 调度、AWQ 量化 GEMM、Flash Attention |

### 1.2 请求链路

1. 用户通过 TRA Console 或 Control UI 发起对话
2. Gateway 调用 `POST /api/agent/chat`，进入 trustclaw-tra 审计 pipeline
3. Agent Pack 通过 `trustclaw_tra_query` / `trustclaw_tra_write` 访问本地 SQLite
4. LLM 推理请求转发至 `http://vllm:8000/v1`（Qwen3.6-27B-AWQ）
5. 响应与 evidence 写入本地 audit log，结论可追溯

### 1.3 部署拓扑（Docker 全栈）

| 容器镜像 | 端口 | 说明 |
|----------|------|------|
| `metax-vllm:local` | 8000 | vLLM 推理服务（挂载宿主机 MACA/conda） |
| `metax-trustclaw:local` | 19001 | TrustClaw Gateway + 配置初始化 |
| `metax-openclaw:local` | — | TrustClaw + TRA UI 基础镜像 |

配置文件：`docker/metax-full/docker-compose.yml`、`configs/trustclaw-metax-vllm.json`

---

## 2. 推理引擎（metaX-inference）

### 2.1 软件栈

| 组件 | 版本 |
|------|------|
| GPU | MetaX C500，32GB sGPU |
| MACA | 3.5.3.20 |
| vLLM | 0.17.0 + vllm_metax 0.17.0 |
| PyTorch | 2.8.0+metax3.5.3.9 |
| Python | 3.10 |
| 模型 | QuantTrio/Qwen3.6-27B-AWQ（INT4，~28GB 显存） |

### 2.2 自研模块

```text
metax_kernels/qwen36/
  fused_rope_rms.py    # RoPE + RMSNorm 融合
  gqa_attention.py     # GQA + SDPA / flash_attn
  awq_gemm.py          # AWQ GEMM 封装
  fused_mlp.py         # SwiGLU MLP
engine/vllm_metax_plugin/  # METAX_KERNELS=1 CustomOp 加载
scripts/bench_qwen36.py    # 端到端 tok/s / 并发 benchmark
```

启用方式：`METAX_KERNELS=1 ./scripts/serve_qwen36_metax.sh`

### 2.3 分阶段验收

| 阶段 | 脚本 | 目标 |
|------|------|------|
| Phase 0 | `bench_qwen36.py` 单请求 | ≥ 9.5 tok/s |
| Phase 1 | `run_phase1_concurrent_bench.sh` | 并发 c=8 ≥ 40 tok/s |
| Phase 2 | `run_op_bench.sh` | fused_rope_rms ≤ 0.5 ms |
| Phase 3 | `run_phase3_mtp_bench.sh` | MTP 等效 ≥ 20 tok/s |

最佳配置：`configs/qwen36-phase1-tuned.yaml`、`configs/qwen36-c8160-tuned.yaml`

---

## 3. 推理效果（实机数据）

来源：`TEST_RESULTS.md`（MetaX C500，2026-07-06 ~ 2026-07-07）

### 3.1 端到端 benchmark

| 指标 | 实测 | 备注 |
|------|------|------|
| 单请求 tok/s | **31.85** | temperature=0，warmup 后 |
| 并发 c=8 聚合 tok/s | **81.02** | Phase 1 PASS |
| MTP 等效 tok/s | **23.65** | MTP-2 + warmup |
| 并发 c=18 聚合 tok/s | **155.89** | C8160 最佳 |
| TTFT | **0.08s** | temperature=0（原 13s） |
| 显存占用 | ~28GB / 32GB | AWQ INT4 |

### 3.2 算子 micro-benchmark（seq-len=256）

| Kernel | avg ms |
|--------|--------|
| qwen36.fused_rope_rms (fused) | 0.525 |
| qwen36.gqa_attention:sdpa | 0.12 |
| qwen36.gqa_attention:fused | 0.16 |

### 3.3 关键调参结论

- Qwen3.6 默认 thinking 块严重影响 tok/s；completions API + `temperature=0` 可跳过
- AWQ 模型上 MTP speculative acceptance 低；生产推荐无 MTP c=8 配置（~81 tok/s）
- sGPU 50% 算力配额为云平台限制；整卡后 c=8 有望 ~160 tok/s

---

## 4. 可信 Agent 运行时（TrustClaw）

### 4.1 TRA 五平面

| 平面 | 实现 |
|------|------|
| Data | 本地 SQLite `state/local_tra.db` |
| Policy | consent 策略控制读写 |
| Agent | Agent Pack 可插拔（`trustclaw/agents/`） |
| Evidence | 不可篡改证据链 |
| Operator | TRA Console / Control UI |

### 4.2 核心 API

| Endpoint | 用途 |
|----------|------|
| `POST /api/tra/init` | 挂载本地 TRA 个人数据 |
| `POST /api/tra/reset` | 清除个人数据 + audit/ledger |
| `POST /api/agent/chat` | 可审计 Agent Pack 流水线 |
| `GET /api/tra/domain-agents` | Agent 目录 |

Agent 工具：`trustclaw_tra_query`（读）、`trustclaw_tra_write`（经 consent 写）

### 4.3 vLLM 后端配置

- `baseUrl`: `http://127.0.0.1:8000/v1`（Docker 内为 `http://vllm:8000/v1`）
- `primary model`: `vllm/Qwen3.6-27B-AWQ`
- `chat_template_kwargs.enable_thinking`: `false`

配置模板：`configs/trustclaw-metax-vllm.json`

---

## 5. 开源与第三方依赖

| 项目 | 协议 | 用途 |
|------|------|------|
| metaX-inference | MIT | 推理内核、benchmark、集成部署 |
| TrustClaw | MIT | TRA 运行时、Agent Pack、Gateway |
| vLLM / vllm_metax | Apache-2.0 | 推理引擎 |
| transformers | Apache-2.0 | Qwen3.6 模型加载 |
| OpenClaw (TrustClaw fork base) | MIT | Gateway、Control UI 基础 |

---

## 6. 复现步骤

```bash
./scripts/test-env-check.sh
./scripts/serve_qwen36_metax.sh
export VLLM_API_KEY=sk-your-key
./scripts/deploy_trustclaw_metax.sh
python scripts/bench_qwen36.py --concurrency 8 --temperature 0 --warmup-requests 1 --stream --json
python scripts/bench_acceptance.py . --markdown
```

---

## 7. 参考文档

| 文件 | 内容 |
|------|------|
| `README.md` | 服务框架 + 推理效果总览 |
| `AGENT.md` | 测试方案 + §12 MACA 架构设计 |
| `TEST_RESULTS.md` | 完整实机测试记录 |
| `docker/metax-full/README.md` | Docker 构建与离线部署 |

---

*导出 PDF：`pandoc docs/CCF_TECH_SUMMARY.md -o MetaX-TrustClaw-技术文档.pdf`*
