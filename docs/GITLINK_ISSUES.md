# GitLink Issue 创建指南

在 [metaX-inference 仓库](https://gitlink.org.cn/xingjian/metaX-inference) 点击 **「+ 疑修(Issue)」**，依次创建以下 6 条。
每条复制 **标题** 和 **正文** 即可。

---

## Issue 1

**标题：** `[Dev] 项目规划与 MetaX-TrustClaw 双层架构设计`

**正文：**

```markdown
## 阶段
设计

## 背景
参加第八届 CCF 开源创新大赛 · 国产开源 GPU AI 创新生态赛，需基于沐曦算力 + OpenClaw 生态构建可复现的开源 Agent 项目。

## 内容
确定 **MetaX-TrustClaw** 双层架构：
- **推理层（metaX-inference）**：沐曦 C500 + vLLM/vllm_metax + Qwen3.6-27B-AWQ
- **应用层（TrustClaw）**：基于 OpenClaw 的 TRA 可信 Agent 运行时

## 产出
- `AGENT.md` §12 MACA 最佳推理架构设计
- Phase 0→3 分阶段验收目标定义
- 初始 benchmark 脚本与测试方案

## 相关文件
- `AGENT.md`
- `README.md`
- `scripts/bench_qwen36.py`

## 状态
已完成
```

---

## Issue 2

**标题：** `[Dev] MetaX C500 实机环境验证与 vLLM 部署`

**正文：**

```markdown
## 阶段
开发 / 测试

## 内容
在沐曦 MetaX C500（32GB sGPU，MACA 3.5.3）上部署 Qwen3.6-27B-AWQ：
- vLLM 0.17.0 + vllm_metax 0.17.0
- `/v1/completions` API 功能验证

## 技术难点
- `config.json` 中 `model_type=qwen3_5`，系统 transformers 4.x 无法识别
- **解决方案**：升级 transformers 5.x dev（`pip install git+https://github.com/huggingface/transformers.git`）

## 验证结果
| 项 | 结果 |
|----|------|
| 模型加载 | PASS |
| vLLM 启动 | PASS |
| API 返回 | HTTP 200，中文续写正常 |
| 显存 | ~28604/32000 MiB，GPU 利用率 ~70% |

## 产出
- `TEST_RESULTS.md` 方案 B PASS
- `scripts/test-scheme-b.sh`
- `scripts/serve_qwen36_metax.sh`

## 状态
已完成
```

---

## Issue 3

**标题：** `[Dev] 模型接口调用与 Phase 0→3 性能调优`

**正文：**

```markdown
## 阶段
优化

## 内容
建立 Phase 0→3 全链路 benchmark 体系，优化 Qwen3.6 在沐曦上的推理吞吐。

## 关键发现
1. Qwen3.6 默认 thinking 块导致 TTFT ~13s、tok/s 偏低
2. **temperature=0** + completions API 可跳过 thinking，TTFT **13s → 0.08s**
3. Phase 1 并发调参使 c=8 聚合 **21 → 81 tok/s**（3.8×）

## 性能数据（MetaX C500 实机）
| 阶段 | 指标 | 目标 | 实测 | 状态 |
|------|------|------|------|------|
| Phase 0 | 单请求 tok/s | ≥9.5 | **31.85** | PASS |
| Phase 1 | 并发 c=8 tok/s | ≥40 | **81.02** | PASS |
| Phase 3 | MTP 等效 tok/s | ≥20 | **23.65** | PASS |

## 产出
- `scripts/bench_qwen36.py`
- `configs/qwen36-phase1-tuned.yaml`
- `configs/acceptance_baseline.json`
- `scripts/tune_targets_loop.sh`

## 相关 Commit
tune_targets_loop、acceptance baseline 相关提交

## 状态
已完成
```

---

## Issue 4

**标题：** `[Dev] metax_kernels 自定义算子与 vLLM 插件`

**正文：**

```markdown
## 阶段
开发

## 内容
在 metaX-inference 中实现 MACA 自定义算子，并通过 vLLM CustomOp 插件加载：
- `fused_rope_rms`：RoPE + RMSNorm 融合
- `gqa_attention`：GQA + SDPA / flash_attn 路径
- `awq_gemm`：AWQ GEMM 封装
- `fused_mlp`：SwiGLU MLP

## 启用方式
```bash
METAX_KERNELS=1 ./scripts/serve_qwen36_metax.sh
```

## 算子 benchmark（seq-len=256）
| Kernel | avg ms |
|--------|--------|
| fused_rope_rms (fused) | 0.525 |
| gqa_attention:sdpa | 0.12 |
| gqa_attention:fused | 0.16 |

## 产出
- `metax_kernels/qwen36/`
- `metax_kernels/bench/op_bench.py`
- `engine/vllm_metax_plugin/`

## 状态
已完成（fused_rope 距 0.5ms 目标差 5%，后续 mcoplib 深度融合）
```

---

## Issue 5

**标题：** `[Dev] TrustClaw 集成 — OpenClaw 生态 + 沐曦本地 vLLM 后端`

**正文：**

```markdown
## 阶段
集成

## 背景
赛题要求基于 OpenClaw 生态开发，且必须使用沐曦算力、禁止外部云端 API。

## 内容
将 TrustClaw（OpenClaw fork + trustclaw-tra）与 metaX-inference 本地 vLLM 打通：
- TrustClaw Gateway 配置 vLLM OpenAI 兼容 provider
- `POST /api/agent/chat` 经 TRA 审计 pipeline 调用本地 Qwen3.6
- Docker compose 一键全栈部署

## 数据流
```
用户 → TRA Console → trustclaw-tra → Agent Pack
     → vLLM API (:8000/v1, 沐曦 GPU) → Evidence 链
```

## 产出
- `configs/trustclaw-metax-vllm.json`
- `scripts/deploy_trustclaw_metax.sh`
- `docker/metax-full/docker-compose.yml`
- TrustClaw 仓库：https://gitlink.org.cn/xingjian/trustclaw

## 服务入口
- vLLM API: http://host:8000/v1
- TRA Console: http://host:19001/trustclaw/
- Control UI: http://host:19001/?token=TOKEN

## 状态
已完成
```

---

## Issue 6

**标题：** `[Dev] 大赛提交材料整理 — README / 部署指南 / PDF / PPT`

**正文：**

```markdown
## 阶段
文档

## 内容
按第八届 CCF 开源创新大赛作品提交要求整理材料：
- README 六大章节（简介、功能、模型算力、部署、示例 I/O、参考来源）
- 部署指南、参考来源说明、开发记录
- 项目简介 / 技术说明 / 性能测试 PDF
- 创意规划 PPT（阶段一）

## 产出
| 材料 | 路径 |
|------|------|
| README | README.md |
| 部署指南 | docs/DEPLOYMENT.md |
| 参考来源 | docs/REFERENCES.md |
| 项目简介 PDF | docs/CCF_PROJECT_BRIEF.pdf |
| 技术说明 PDF | docs/CCF_TECH_SUMMARY.pdf |
| 测试报告 PDF | docs/CCF_TEST_REPORT.pdf |
| 提交对照表 | docs/SUBMISSION.md |
| 调用日志样例 | docs/samples/inference_call_log.txt |

## 战队信息
- 战队：YiRage
- 队长：陈星强 · 厦门大学
- GitLink：gitlink.org.cn/xingjian/metaX-inference

## 待完成
- [ ] Demo 演示视频链接填入 README
- [ ] 运行截图上传 docs/demo/

## 状态
进行中
```

---

## 创建步骤

1. 打开 https://gitlink.org.cn/xingjian/metaX-inference/issues
2. 点击「+ 疑修」或「新建 Issue」
3. 复制上面对应 **标题** 和 **正文**（去掉 markdown 代码块围栏后粘贴）
4. 创建 6 条 Issue 后关闭（状态选「已完成」或留 Open 均可）
