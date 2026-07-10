# 参考来源与原创说明

## 自研内容（本仓库）

- `metax_kernels/`：MACA 自定义算子（fused RoPE、GQA、AWQ GEMM、MLP）
- `engine/vllm_metax_plugin/`：vLLM CustomOp 加载器
- `scripts/bench_*.py`、`scripts/serve_qwen36_metax.sh`：benchmark 与部署脚本
- `docker/metax-full/`：vLLM + TrustClaw Docker 集成
- `configs/trustclaw-metax-vllm.json`：沐曦 vLLM 后端配置模板

## 参考的开源项目

| 项目 | 协议 | 用途 | 本仓库改造 |
|------|------|------|-----------|
| [vLLM](https://github.com/vllm-project/vllm) | Apache-2.0 | 推理引擎框架 | 集成 vllm_metax + 自研 kernel |
| [HuggingFace transformers](https://github.com/huggingface/transformers) | Apache-2.0 | Qwen3.6 模型加载 | 适配 qwen3_5 架构 |
| [OpenClaw](https://github.com/openclaw/openclaw) | MIT | Gateway / Control UI 基础 | TrustClaw fork + TRA 扩展（见 TrustClaw 仓库） |
| [TrustClaw](https://gitlink.org.cn/xingjian/trustclaw) | MIT | 可信 Agent 运行时 | 本仓库提供沐曦 vLLM 后端集成 |
| QuantTrio/Qwen3.6-27B-AWQ | 模型许可 | 预量化权重 | 推理评测与部署 |

## 原创性声明

本项目在开源组件基础上进行了**功能扩展与场景改造**：

1. 国产沐曦 GPU 上 Qwen3.6 全链路 benchmark 与调参体系（Phase 0→3）
2. metaX-inference 自研 MACA 算子与 vLLM 插件
3. TrustClaw TRA 与沐曦本地 vLLM 的工程化打通（非简单配置拼接）
4. 可信 Agent 场景（数据不出域、证据链）+ 国产算力合规部署

**非**直接复制已有项目或仅做简单修改。
