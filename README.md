# metaX-inference

在沐曦显卡上运行 **Qwen3.6-27B-AWQ** 的高性能推理引擎，并与 **[TrustClaw](https://gitlink.org.cn/xingjian/trustclaw)** 集成构成 **MetaX-TrustClaw** 全栈：**国产 GPU 本地推理 + 可信 Agent 运行时（TRA）**。

**GitLink:** [metaX-inference](https://gitlink.org.cn/xingjian/metaX-inference) · [TrustClaw](https://gitlink.org.cn/xingjian/trustclaw)  
**Release:** [v0.1.1](https://github.com/chenxingqiang/metaX-inference/releases/tag/v0.1.1) · [v0.1.0](https://github.com/chenxingqiang/metaX-inference/releases/tag/v0.1.0)

---

## 项目简介

**MetaX-TrustClaw** 参加 **第八届 CCF 开源创新大赛 · 国产开源 GPU AI 创新生态赛**。

- **战队：** YiRage · **队长：** 陈星强 · **厦门大学**
- **定位：** 基于沐曦国产 GPU 的本地可信 AI Agent 全栈，不调用 OpenAI 等外部云端 API
- **双仓库：** [metaX-inference](https://gitlink.org.cn/xingjian/metaX-inference)（推理层）+ [TrustClaw](https://gitlink.org.cn/xingjian/trustclaw)（Agent 运行时，OpenClaw 生态）

## 功能说明

| 功能 | 说明 |
|------|------|
| 沐曦 Qwen3.6 推理 | vLLM + vllm_metax，OpenAI 兼容 API（:8000/v1） |
| 自定义 MACA 算子 | fused RoPE/GQA/AWQ/MLP，`METAX_KERNELS=1` 启用 |
| 全链路 benchmark | Phase 0→3 验收脚本，实机 tok/s / 并发 / MTP |
| TrustClaw TRA 集成 | Agent Pack、Evidence 链、TRA Console |
| 模型接口调用 | `configs/trustclaw-metax-vllm.json` 配置本地 vLLM provider |
| Docker 一键部署 | vLLM + TrustClaw 全栈 compose |

## 模型与算力环境

| 项 | 配置 |
|----|------|
| **模型** | QuantTrio/Qwen3.6-27B-AWQ（INT4） |
| **GPU** | 沐曦 MetaX C500，32GB sGPU，MACA 3.5.3 |
| **推理引擎** | vLLM 0.17.0 + vllm_metax 0.17.0 |
| **Agent 框架** | TrustClaw（OpenClaw fork + trustclaw-tra） |
| **算力来源** | 沐曦算力卡 / Gitee.AI 沐曦资源包（赛题要求） |

## 示例输入输出

### vLLM Completions API

**请求**（见 [docs/samples/vllm_completion_request.json](./docs/samples/vllm_completion_request.json)）：

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -d '{"prompt":"你好，我是","max_tokens":64,"temperature":0.7}'
```

**响应**（摘录，见 [docs/samples/vllm_completion_response.json](./docs/samples/vllm_completion_response.json)）：

```json
{
  "choices": [{"text": "小雅。你好，小雅！很高兴认识你。", "finish_reason": "length"}],
  "usage": {"completion_tokens": 64}
}
```

### Agent Chat API（TrustClaw）

```bash
curl -X POST http://127.0.0.1:19001/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"请根据我的数据回答","agentPack":"glp1-eligibility"}'
```

## 演示材料

| 材料 | 位置 |
|------|------|
| **创意规划 PPT（阶段一）** | `docs/` 目录内 PPT 文件 |
| **技术说明 PDF** | [docs/CCF_TECH_SUMMARY.pdf](./docs/CCF_TECH_SUMMARY.pdf) |
| **性能测试报告 PDF** | [docs/CCF_TEST_REPORT.pdf](./docs/CCF_TEST_REPORT.pdf) |
| **项目简介 PDF** | [docs/CCF_PROJECT_BRIEF.pdf](./docs/CCF_PROJECT_BRIEF.pdf) |
| **真实调用日志** | [TEST_RESULTS.md](./TEST_RESULTS.md)、[docs/samples/inference_call_log.txt](./docs/samples/inference_call_log.txt) |
| **运行截图** | [docs/demo/](./docs/demo/)（待上传截图） |
| **Demo 演示视频** | **待填链接：** `<!-- 上传后填写，例如：https://... -->` |

## 参考来源说明

本项目参考并扩展以下开源项目，**非简单复制**：

| 项目 | 协议 | 本仓库改造 |
|------|------|-----------|
| vLLM / vllm_metax | Apache-2.0 | 沐曦 kernel + benchmark 体系 |
| OpenClaw / TrustClaw | MIT | TRA 扩展 + 沐曦 vLLM 后端集成 |
| transformers | Apache-2.0 | Qwen3.6 qwen3_5 适配 |

完整说明见 **[docs/REFERENCES.md](./docs/REFERENCES.md)**。

## 开发过程记录

- **Issue 模板：** [.github/ISSUE_TEMPLATE/development_record.yml](./.github/ISSUE_TEMPLATE/development_record.yml)
- **开发记录摘要：** [docs/DEVELOPMENT_LOG.md](./docs/DEVELOPMENT_LOG.md)
- **Commit 历史：** GitLink 仓库提交记录
- **提交对照表：** [docs/SUBMISSION.md](./docs/SUBMISSION.md)


---

## 服务框架

MetaX-TrustClaw 分为 **应用层 / 推理层 / 算力层** 三层，数据与审计全程本地闭环：

```text
┌─────────────────────────────────────────────────────────────────┐
│  应用层 — TrustClaw TRA（可信 Agent 运行时）                       │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ TRA Console  │  │ Control UI   │  │ Agent Pack (可插拔)     │ |
│  │ :19001/      │  │ :19001/      │  │ glp1-eligibility 等    │ |
│  │  trustclaw/  │  │  ?token=…    │  │                        │ |
│  └──────┬───────┘  └──────┬───────┘  └───────────┬────────────┘ │
│         │                 │                      │              │
│         └─────────────────┴──────────────────────┘              |
│                           │                                     │
│              trustclaw-tra 插件 · Evidence 链 · local_tra.db     │
│             POST /api/agent/chat · trustclaw_tra_query/write    │
├───────────────────────────┼─────────────────────────────────────┐
│  推理层 — metaX-inference（本仓库）                                │
│                           ▼                                      |
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ vLLM OpenAI API (:8000/v1)                                  │ │
│  │  · vllm_metax 0.17.0                                        │ │
│  │  · engine/vllm_metax_plugin（METAX_KERNELS=1 自定义算子）     │ │
│  │  · metax_kernels（fused RoPE / GQA / AWQ GEMM / SwiGLU）     │ │
│  └─────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  算力层 — MetaX MACA                                             │
│  GPU: MetaX C500 · MACA 3.5+ · 模型: Qwen3.6-27B-AWQ (INT4)      │
└─────────────────────────────────────────────────────────────────┘
```

### 服务入口

| 入口 | 地址 | 说明 |
|------|------|------|
| **vLLM API** | `http://<host>:8000/v1` | OpenAI 兼容推理接口，供 Agent / 应用直接调用 |
| **TrustClaw Control UI** | `http://<host>:19001/?token=<TOKEN>` | Gateway 控制台、模型与会话管理 |
| **TRA Console** | `http://<host>:19001/trustclaw/` | 可信运行时：数据初始化、Agent 对话、审计查看 |
| **OpenAI Completions** | `POST /v1/completions` | 底层推理验证（见 [TEST_RESULTS.md](./TEST_RESULTS.md)） |
| **Agent Chat** | `POST /api/agent/chat` | 经 TRA 审计的 Agent Pack 流水线 |

### 部署模式

| 模式 | 命令 | 适用 |
|------|------|------|
| **仅推理** | `./scripts/serve_qwen36_metax.sh` | benchmark、API 服务 |
| **推理 + Agent** | `./scripts/deploy_trustclaw_metax.sh` | vLLM 已运行后叠加 Gateway |
| **Docker 全栈** | `docker/metax-full/docker compose up -d` | 生产一键部署（见 [docker/metax-full/README.md](./docker/metax-full/README.md)） |

### TrustClaw 可信原则

| 原则 | 实现 |
|------|------|
| 个人数据不出域 | Raw data 仅存本地 SQLite `state/local_tra.db` |
| 凡答必有据 | 结论追溯到 data + rules + evidence chain |
| 凡行必审计 | 每次 data access / agent action 写日志 |
| Agent 与平台解耦 | 垂直逻辑以 Agent Pack 交付，非平台硬编码 |

---

## 推理效果

以下数据来自 **MetaX C500 实机**（32GB sGPU，MACA 3.5.3，vLLM 0.17.0 + vllm_metax），模型 **QuantTrio/Qwen3.6-27B-AWQ**。完整记录见 [TEST_RESULTS.md](./TEST_RESULTS.md)。

### 测试环境

| 项目 | 值 |
|------|-----|
| GPU | MetaX C500，32GB VRAM（sGPU 50% 算力配额） |
| 模型 | Qwen3.6-27B-AWQ（INT4，显存 ~28GB） |
| 引擎 | vLLM 0.17.0 + vllm_metax 0.17.0 |
| Python | 3.10 · PyTorch 2.8.0+metax |

### 分阶段验收（Phase 0 → 3）

| 阶段 | 指标 | 目标 | **实测** | 状态 |
|------|------|------|----------|------|
| Phase 0 | 单请求 decode tok/s | ≥ 9.5 | **31.85** | PASS |
| Phase 1 | 并发 8 req 聚合 tok/s | ≥ 40 | **81.02** | PASS |
| Phase 3 | MTP speculative 等效 tok/s | ≥ 20 | **23.65** | PASS |
| C8160 | 并发 18 req 聚合 tok/s | ~160 | **155.89** | 接近目标 |

### 关键优化发现

| 优化项 | 效果 |
|--------|------|
| `temperature=0` + completions API | TTFT **13s → 0.08s**，跳过 Qwen3.6 thinking 预填充 |
| Phase 1 并发调参（`configs/qwen36-phase1-tuned.yaml`） | c=8 聚合 **21 → 81 tok/s**（3.8×） |
| 生产推荐（无 MTP） | c=8 **~81 tok/s**；c=18 **~156 tok/s** |
| BF16 MTP graft | 单请求 MTP **23 tok/s**；并发 c=8 MTP ~40–47 tok/s |

### 算子 micro-benchmark（Phase 2）

`PYTHONPATH=. python -m metax_kernels.bench.op_bench --seq-len 256`

| Kernel | avg ms | 说明 |
|--------|--------|------|
| qwen36.fused_rope_rms (fused) | **0.525** | RoPE+RMSNorm 融合（目标 ≤0.5 ms） |
| qwen36.gqa_attention:sdpa | **0.12** | GQA SDPA，当前最快 attention 路径 |
| qwen36.gqa_attention:fused | 0.16 | mcflashattn 路径 |

复现命令：

```bash
# 单请求吞吐（需 MetaX 实机 + vLLM 已启动）
python scripts/bench_qwen36.py --temperature 0 --max-tokens 128 --warmup-requests 1 --stream --json

# 并发 8 请求
python scripts/bench_qwen36.py --concurrency 8 --temperature 0 --warmup-requests 1 --stream --json

# 算子 benchmark
PYTHONPATH=. ./scripts/run_op_bench.sh --seq-len 256 --json

# 自动验收报告
python scripts/bench_acceptance.py . --markdown
```

---

## 文档

- **[AGENT.md](./AGENT.md)** — 测试方案 + §12 MACA 最佳推理架构设计
- **[TEST_RESULTS.md](./TEST_RESULTS.md)** — 实机 A/B 测试与调参全记录
- **[docker/metax-full/README.md](./docker/metax-full/README.md)** — Docker 全栈构建与离线部署
- **[docs/DEPLOYMENT.md](./docs/DEPLOYMENT.md)** — 部署指南
- **[docs/REFERENCES.md](./docs/REFERENCES.md)** — 参考来源与原创说明
- **[docs/SUBMISSION.md](./docs/SUBMISSION.md)** — 作品提交对照表
- **[docs/DEVELOPMENT_LOG.md](./docs/DEVELOPMENT_LOG.md)** — 功能开发记录
- **[docs/CCF_PROJECT_BRIEF.pdf](./docs/CCF_PROJECT_BRIEF.pdf)** — 项目简介 PDF
- **[docs/CCF_TECH_SUMMARY.pdf](./docs/CCF_TECH_SUMMARY.pdf)** — 技术说明 PDF
- **[docs/CCF_TEST_REPORT.pdf](./docs/CCF_TEST_REPORT.pdf)** — 性能测试报告 PDF

---

## 快速开始

### 推理层（MetaX 实机）

```bash
# 环境检查
./scripts/test-env-check.sh

# 启动 vLLM 生产服务
./scripts/serve_qwen36_metax.sh
METAX_KERNELS=1 ENABLE_MTP=1 ./scripts/serve_qwen36_metax.sh   # 启用自定义 kernel + MTP

# 冒烟测试
export VLLM_MODEL=/data/models/Qwen3.6-27B-AWQ
./scripts/test-scheme-b.sh

# 实机一键 benchmark
./scripts/remote_run_all_benches.sh
curl -fsSL "https://raw.githubusercontent.com/chenxingqiang/metaX-inference/main/scripts/metax_paste_and_run.sh" | bash

# 单元测试 + 验收
python -m unittest discover -s tests -v
python scripts/bench_acceptance.py . --markdown
pip install -e .
```

### 全栈（推理 + TrustClaw Agent）

```bash
# 方式 A：脚本（vLLM 已运行）
export VLLM_API_KEY=sk-your-key
./scripts/deploy_trustclaw_metax.sh

# 方式 B：Docker 一键栈
git clone https://gitlink.org.cn/xingjian/trustclaw.git ../TrustClaw
./scripts/build_metax_full_image.sh
cd docker/metax-full && cp app.env.example app.env
docker compose up -d
```

| 组件 | 路径 |
|------|------|
| Gateway 配置 | [configs/trustclaw-metax-vllm.json](./configs/trustclaw-metax-vllm.json) |
| 实机部署脚本 | [scripts/deploy_trustclaw_metax.sh](./scripts/deploy_trustclaw_metax.sh) |
| Docker compose | [docker/metax-full/](./docker/metax-full/) |
| TrustClaw 源码 | [gitlink.org.cn/xingjian/trustclaw](https://gitlink.org.cn/xingjian/trustclaw) |

默认 Agent Pack：`glp1-eligibility`（`app.env` 中 `TRUSTCLAW_DEFAULT_AGENT_PACK` 可改）。

---

## 包结构

```text
metaX-inference/
├── metax_kernels/              # MACA 自定义算子
│   ├── qwen36/                 #   fused RoPE, GQA, AWQ GEMM, SwiGLU MLP
│   └── bench/op_bench.py       #   算子 micro-benchmark
├── engine/vllm_metax_plugin/   # vLLM CustomOp + METAX_KERNELS=1 加载器
├── configs/                    # 调优 yaml + trustclaw-metax-vllm.json
├── docker/metax-full/          # vLLM + TrustClaw Docker 全栈
├── scripts/
│   ├── serve_qwen36_metax.sh   #   生产 vLLM 启动
│   ├── bench_qwen36.py         #   端到端 tok/s / 并发 benchmark
│   └── deploy_trustclaw_metax.sh
├── tests/                      # 单元测试 + acceptance baseline
├── AGENT.md                    # 架构设计与测试方案
└── TEST_RESULTS.md             # 实机 benchmark 数据
```

详细步骤、模型选型与故障排查见 [AGENT.md](./AGENT.md)。
