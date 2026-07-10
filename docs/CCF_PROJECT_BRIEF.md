# MetaX-TrustClaw 项目简介

**第八届 CCF 开源创新大赛 · 初赛提交材料**

---

## 1. 项目概述

**MetaX-TrustClaw** 是在沐曦国产 GPU 上运行的 **本地可信 AI Agent 全栈**：

- **metaX-inference**（本仓库）：Qwen3.6-27B-AWQ 高性能推理引擎、自定义 MACA 算子、全链路 benchmark
- **TrustClaw**（配套仓库）：local-first 可信 Agent 运行时（TRA），支持数据不出域、凡答必有据、凡行必审计

两层通过 vLLM OpenAI 兼容 API 对接，支持脚本部署与 Docker 一键全栈。

| 项 | 内容 |
|----|------|
| 项目名称 | MetaX-TrustClaw |
| 开源协议 | MIT |
| 推理模型 | Qwen3.6-27B-AWQ（INT4） |
| 目标硬件 | 沐曦 MetaX C500（MACA 3.5+） |
| GitLink | [metaX-inference](https://gitlink.org.cn/xingjian/metaX-inference) · [TrustClaw](https://gitlink.org.cn/xingjian/trustclaw) |

---

## 2. 解决的问题

1. **国产算力推理缺口**：沐曦 GPU 上缺少 Qwen3.6 可复现 benchmark 与生产级 vLLM 部署方案
2. **Agent 可信治理缺口**：合规场景要求个人数据不出域、决策可审计、结论有证据链
3. **栈割裂**：推理优化与 Agent 治理各自独立，缺少端到端本地开源方案

---

## 3. 服务框架

```text
应用层   TrustClaw TRA — TRA Console / Control UI / Agent Pack
         trustclaw-tra 插件 · Evidence 链 · local_tra.db
推理层   metaX-inference — vLLM (:8000/v1) + metax_kernels + vllm_metax_plugin
算力层   MetaX C500 + Qwen3.6-27B-AWQ
```

**服务入口**

| 入口 | 地址 |
|------|------|
| vLLM API | `http://<host>:8000/v1` |
| Control UI | `http://<host>:19001/?token=<TOKEN>` |
| TRA Console | `http://<host>:19001/trustclaw/` |
| Agent Chat | `POST /api/agent/chat` |

---

## 4. 推理效果（MetaX C500 实机）

测试环境：MetaX C500 · 32GB sGPU · MACA 3.5.3 · vLLM 0.17.0 + vllm_metax

| 阶段 | 指标 | 目标 | 实测 | 状态 |
|------|------|------|------|------|
| Phase 0 | 单请求 tok/s | ≥ 9.5 | **31.85** | PASS |
| Phase 1 | 并发 c=8 聚合 tok/s | ≥ 40 | **81.02** | PASS |
| Phase 3 | MTP 等效 tok/s | ≥ 20 | **23.65** | PASS |
| C8160 | 并发 c=18 聚合 tok/s | ~160 | **155.89** | 接近 |

关键优化：`temperature=0` 使 TTFT 从 13s 降至 0.08s；Phase 1 调参使并发 c=8 从 21 提升至 81 tok/s。

---

## 5. 核心创新

1. **国产 GPU + 可信 Agent 全栈开源**：推理与 TRA 治理工程化打通
2. **可复现 benchmark 体系**：Phase 0→3 分阶段验收脚本与实机数据
3. **TRA 五平面可信运行时**：Data · Policy · Agent · Evidence · Operator
4. **Agent Pack 可插拔**：垂直场景与平台解耦（默认 `glp1-eligibility`）
5. **一键部署**：Docker compose 全栈 + 内网离线镜像导出

---

## 6. 应用场景

- 医疗健康 Agent（个人数据不出域）
- 企业知识库问答（本地推理 + 审计）
- 政务 / 金融合规场景（国产算力 + 证据链）
- 开发者二次扩展（OpenClaw 多通道 + MetaX 本地后端）

---

## 7. 快速复现

```bash
# 推理层
./scripts/serve_qwen36_metax.sh

# 全栈（Docker）
git clone https://gitlink.org.cn/xingjian/trustclaw.git ../TrustClaw
./scripts/build_metax_full_image.sh
cd docker/metax-full && cp app.env.example app.env && docker compose up -d
```

详细文档见仓库 `README.md`、`AGENT.md`、`TEST_RESULTS.md`。

---

*导出 PDF：可使用 Typora / VS Code Markdown PDF / pandoc 将此文件转为 PDF 上传。*
