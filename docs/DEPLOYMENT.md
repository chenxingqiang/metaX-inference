# MetaX-TrustClaw 部署指南

**第八届 CCF 开源创新大赛 · 国产开源 GPU AI 创新生态赛**

## 1. 前置条件

| 项 | 要求 |
|----|------|
| 硬件 | 沐曦 MetaX C500（推荐 32GB 显存） |
| 驱动 | MACA 3.5+（/opt/maca、/opt/mxdriver） |
| Python | 3.10 |
| 模型 | QuantTrio/Qwen3.6-27B-AWQ（~28GB 显存） |
| 算力来源 | 沐曦算力卡 / Gitee.AI 沐曦资源包（赛题要求） |

## 2. 依赖清单

见 [requirements-metax.txt](../requirements-metax.txt) 与 [pyproject.toml](../pyproject.toml)。

核心：vLLM 0.17.0 + vllm_metax、transformers 5.x（qwen3_5）、PyTorch metax 版。

TrustClaw 侧：Node.js 22.19+、pnpm 11.x（见 TrustClaw 仓库 package.json）。

## 3. 方式 A：仅推理层（vLLM）

```bash
git clone https://gitlink.org.cn/xingjian/metaX-inference.git
cd metaX-inference
./scripts/test-env-check.sh
./scripts/serve_qwen36_metax.sh
curl http://127.0.0.1:8000/v1/models -H "Authorization: Bearer $VLLM_API_KEY"
```

## 4. 方式 B：推理 + TrustClaw Agent

```bash
# vLLM 已运行后
export VLLM_API_KEY=sk-your-key
./scripts/deploy_trustclaw_metax.sh
# Control UI: http://<host>:19001/?token=<TOKEN>
# TRA Console: http://<host>:19001/trustclaw/
```

## 5. 方式 C：Docker 全栈

```bash
git clone https://gitlink.org.cn/xingjian/trustclaw.git ../TrustClaw
./scripts/build_metax_full_image.sh
cd docker/metax-full
cp app.env.example app.env   # 编辑 OPENCLAW_GATEWAY_TOKEN、VLLM_API_KEY
docker compose up -d
```

## 6. 模型接口配置

TrustClaw → vLLM 配置模板：[configs/trustclaw-metax-vllm.json](../configs/trustclaw-metax-vllm.json)

- baseUrl: http://127.0.0.1:8000/v1
- model: vllm/Qwen3.6-27B-AWQ
- enable_thinking: false

## 7. 验证与 benchmark

```bash
python scripts/bench_qwen36.py --temperature 0 --max-tokens 128 --warmup-requests 1 --stream --json
python scripts/bench_acceptance.py . --markdown
```

## 8. 故障排查

| 现象 | 处理 |
|------|------|
| qwen3_5 无法加载 | 升级 transformers 5.x dev |
| OOM | 使用 AWQ INT4，降低 max-model-len |
| TrustClaw 无法连 vLLM | 检查 VLLM_API_KEY 与 deploy 日志 |

详见 [AGENT.md](../AGENT.md)、[TEST_RESULTS.md](../TEST_RESULTS.md)。
