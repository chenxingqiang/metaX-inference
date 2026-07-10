# 功能开发记录

> 评审用开发过程记录。建议在 GitLink Issue 中同步创建对应条目。

## Issue #1 — 项目规划与架构设计

- **阶段**：设计
- **内容**：确定 MetaX-TrustClaw 双层架构（推理层 + TRA 应用层）
- **产出**：AGENT.md §12 MACA 最佳推理架构设计
- **Commit**：初始 benchmark 脚本与测试方案

## Issue #2 — MetaX C500 实机环境验证

- **阶段**：开发 / 测试
- **内容**：vLLM + vllm_metax 部署 Qwen3.6-27B-AWQ
- **难点**：transformers 不支持 qwen3_5 → 升级 5.x dev
- **产出**：TEST_RESULTS.md 方案 B PASS
- **Commit**：test-scheme-b.sh、serve_qwen36_metax.sh

## Issue #3 — 模型接口调用与性能调优

- **阶段**：优化
- **内容**：Phase 0→3 benchmark；temperature=0 跳过 thinking
- **数据**：单请求 31.85 tok/s；c=8 聚合 81.02 tok/s
- **产出**：bench_qwen36.py、configs/qwen36-phase1-tuned.yaml
- **Commit**：tune_targets_loop、acceptance baseline

## Issue #4 — metax_kernels 算子开发

- **阶段**：开发
- **内容**：fused_rope_rms、gqa_attention、awq_gemm
- **产出**：metax_kernels/、op_bench.py
- **Commit**：vllm_metax_plugin METAX_KERNELS=1

## Issue #5 — TrustClaw 集成（OpenClaw 生态）

- **阶段**：集成
- **内容**：TrustClaw Gateway + 本地 vLLM provider
- **产出**：configs/trustclaw-metax-vllm.json、deploy_trustclaw_metax.sh
- **Commit**：docker/metax-full 全栈

## Issue #6 — 大赛提交材料

- **阶段**：文档
- **内容**：README、部署指南、PDF、PPT
- **产出**：docs/CCF_*.pdf、DEPLOYMENT.md
