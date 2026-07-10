# MetaX-TrustClaw 技术文档

**第八届 CCF 开源创新大赛 · 国产开源 GPU AI 创新生态赛 · 初赛提交材料**

**战队：** YiRage · **队长：** 陈星强 · **单位：** 厦门大学  
**GitLink：** gitlink.org.cn/xingjian/metaX-inference · gitlink.org.cn/xingjian/trustclaw

---

## 1. 系统架构

### 1.1 三层架构

| 层级 | 组件 | 职责 |
|------|------|------|
| 应用层 | TrustClaw Gateway + trustclaw-tra | Agent Pack、证据链、TRA Console |
| 推理层 | vLLM + vllm_metax + metax_kernels | Qwen3.6 本地推理、OpenAI 兼容 API |
| 算力层 | MetaX MACA C500 | AWQ GEMM、Flash Attention |

### 1.2 请求链路

1. 用户通过 TRA Console 发起对话  
2. POST /api/agent/chat 进入 trustclaw-tra 审计 pipeline  
3. Agent Pack 访问本地 SQLite（trustclaw_tra_query/write）  
4. LLM 请求转发至本地 vLLM http://vllm:8000/v1（沐曦 GPU）  
5. 响应与 evidence 写入 audit log  

**合规：** 不调用 OpenAI / Anthropic 等外部云端 API。

### 1.3 Docker 部署

| 镜像 | 端口 | 说明 |
|------|------|------|
| metax-vllm:local | 8000 | vLLM 推理 |
| metax-trustclaw:local | 19001 | TrustClaw Gateway |
| metax-openclaw:local | — | TRA UI 基础镜像 |

---

## 2. 推理引擎

| 组件 | 版本 |
|------|------|
| GPU | MetaX C500，32GB sGPU |
| MACA | 3.5.3.20 |
| vLLM | 0.17.0 + vllm_metax |
| 模型 | Qwen3.6-27B-AWQ INT4 |

自研模块：fused_rope_rms、gqa_attention、awq_gemm、fused_mlp、vllm_metax_plugin

---

## 3. 推理效果

| 指标 | 实测 |
|------|------|
| 单请求 tok/s | 31.85 |
| 并发 c=8 tok/s | 81.02 |
| MTP 等效 tok/s | 23.65 |
| 并发 c=18 tok/s | 155.89 |
| TTFT | 0.08s |

---

## 4. TrustClaw / OpenClaw

TrustClaw 基于 OpenClaw fork，新增 TRA 五平面：Data · Policy · Agent · Evidence · Operator

核心 API：/api/tra/init、/api/agent/chat、trustclaw_tra_query、trustclaw_tra_write

vLLM 配置：configs/trustclaw-metax-vllm.json

---

## 5. 复现

    ./scripts/test-env-check.sh
    ./scripts/serve_qwen36_metax.sh
    ./scripts/deploy_trustclaw_metax.sh
    python scripts/bench_acceptance.py . --markdown

**YiRage · 厦门大学 · 陈星强 · 2026-07-10**
