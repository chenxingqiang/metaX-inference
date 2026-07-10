# MetaX-TrustClaw 性能测试报告

**第八届 CCF 开源创新大赛 · 国产开源 GPU AI 创新生态赛**

**战队：** YiRage · **队长：** 陈星强 · **单位：** 厦门大学  
**测试日期：** 2026-07-06 ~ 2026-07-07

---

## 1. 测试环境

| 项目 | 值 |
|------|-----|
| GPU | MetaX C500，32GB sGPU，50% 算力配额 |
| MACA | 3.5.3.20 |
| vLLM | 0.17.0 + vllm_metax |
| 模型 | Qwen3.6-27B-AWQ INT4 |

---

## 2. 分阶段验收

| 阶段 | 指标 | 目标 | 实测 | 状态 |
|------|------|------|------|------|
| Phase 0 | 单请求 tok/s | ≥ 9.5 | 31.85 | PASS |
| Phase 1 | 并发 c=8 tok/s | ≥ 40 | 81.02 | PASS |
| Phase 2 | fused_rope ms | ≤ 0.5 | 0.525 | 接近 |
| Phase 3 | MTP tok/s | ≥ 20 | 23.65 | PASS |
| C8160 | c=18 tok/s | ~160 | 155.89 | 接近 |

---

## 3. Phase 0 调参

| 配置 | tok/s | TTFT |
|------|-------|------|
| t0.7 + thinking | 7.50 | 13.0s |
| **t0 无 thinking** | **31.85** | **0.08s** |

---

## 4. Phase 1 并发

| 配置 | c8 tok/s |
|------|----------|
| 默认 | 39.78 |
| **t0 调优** | **81.02** |

---

## 5. 算子 benchmark (S=256)

| Kernel | ms |
|--------|-----|
| fused_rope_rms | 0.525 |
| gqa_attention:sdpa | 0.12 |
| gqa_attention:fused | 0.16 |

---

## 6. 合规声明

- 全部数据在沐曦 MetaX C500 实机采集  
- 本地 vLLM 推理，未调用外部云端 API  
- 详见仓库 TEST_RESULTS.md  

**YiRage · 厦门大学 · 陈星强 · 2026-07-10**
