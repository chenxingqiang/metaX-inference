# MetaX-TrustClaw 项目简介

**第八届 CCF 开源创新大赛 · 国产开源 GPU AI 创新生态赛 · 初赛提交材料**

---

| 字段 | 内容 |
|------|------|
| 战队名称 | YiRage |
| 队长 | 陈星强 |
| 单位 | 厦门大学 |
| 联系电话 | 15001334125 |
| 提交日期 | 2026 年 7 月 10 日 |
| 开源协议 | MIT |

**GitLink：**

- metaX-inference：https://gitlink.org.cn/xingjian/metaX-inference
- TrustClaw：https://gitlink.org.cn/xingjian/trustclaw

---

## 1. 项目概述

MetaX-TrustClaw 是基于国产沐曦 GPU 的开源 AI Agent 全栈。下层 metaX-inference 在沐曦 C500 上部署 Qwen3.6-27B-AWQ 高性能推理；上层 TrustClaw 基于 OpenClaw 构建可信 Agent 运行时（TRA）。全链路在沐曦算力环境完成，不依赖 OpenAI 等外部云端 API。

## 2. 赛题对齐

| 赛题要求 | 实现 |
|----------|------|
| 沐曦算力卡 / Gitee.AI | MetaX C500 实机，MACA 3.5.3 + vllm_metax |
| 禁止外部云端 API | 本地 vLLM :8000/v1 |
| OpenClaw 生态 | TrustClaw = OpenClaw fork + trustclaw-tra |
| Skill / MCP | OpenClaw Skills、extensions/trustclaw-tra |
| GitLink 开源 | 双仓库 MIT，Docker 一键部署 |
| 性能数据 | TEST_RESULTS.md + bench_acceptance |

## 3. 服务框架

应用层：TrustClaw TRA（TRA Console、Control UI、Agent Pack、Evidence 链）  
推理层：vLLM + vllm_metax + metax_kernels（:8000/v1）  
算力层：MetaX C500 + Qwen3.6-27B-AWQ

## 4. 推理效果（实机）

环境：MetaX C500 · 32GB · MACA 3.5.3 · vLLM 0.17.0

| 阶段 | 指标 | 目标 | 实测 | 状态 |
|------|------|------|------|------|
| Phase 0 | 单请求 tok/s | ≥9.5 | 31.85 | PASS |
| Phase 1 | 并发 c=8 tok/s | ≥40 | 81.02 | PASS |
| Phase 3 | MTP tok/s | ≥20 | 23.65 | PASS |
| C8160 | c=18 tok/s | ~160 | 155.89 | 接近 |

TTFT：13s → 0.08s（temperature=0）；c=8：21 → 81 tok/s（3.8×）

## 5. 核心创新

1. 国产 GPU + OpenClaw 全栈开源  
2. Phase 0→3 可复现 benchmark  
3. TRA 五平面可信运行时  
4. Agent Pack 可插拔  
5. Docker 一键部署  

## 6. 应用场景

医疗 Agent、企业知识库、政务金融合规、Skill/MCP 开发者扩展

## 7. 复现

    ./scripts/serve_qwen36_metax.sh
    ./scripts/deploy_trustclaw_metax.sh
    docker/metax-full: docker compose up -d

---

YiRage 战队 · 厦门大学 · 陈星强
