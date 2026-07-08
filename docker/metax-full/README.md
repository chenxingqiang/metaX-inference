# MetaX 完整 Docker 栈

vLLM (Qwen3.6-27B-AWQ) + TrustClaw Gateway，一键构建与部署。

## 镜像组成

| 镜像 | 说明 |
|------|------|
| `metax-openclaw:local` | TrustClaw + TRA UI（基于 TrustClaw 官方 Dockerfile） |
| `metax-vllm:local` | vLLM 启动包装器（运行时挂载宿主机 MACA/conda） |
| `metax-trustclaw:local` | Gateway + vLLM 配置初始化 |

## 前置条件

- MetaX 宿主机已安装：`/opt/maca`、`/opt/mxdriver`、`/opt/conda`（含 vllm_metax）
- 模型权重：`/data/models/Qwen3.6-27B-AWQ`
- Docker + docker compose
- TrustClaw 源码（与 metaX-inference 同级目录，或通过 `TRUSTCLAW_DIR` 指定）

## 构建

```bash
# 克隆 TrustClaw（若尚未克隆）
git clone https://github.com/chenxingqiang/TrustClaw.git ../TrustClaw

# 构建三个镜像（首次约 20–40 分钟）
./scripts/build_metax_full_image.sh

# 导出离线包（内网部署）
./scripts/build_metax_full_image.sh --save
# → docker/metax-full/dist/metax-full-images.tar
```

## 运行

```bash
cd docker/metax-full
cp app.env.example app.env
# 编辑 app.env：OPENCLAW_GATEWAY_TOKEN、VLLM_API_KEY

docker compose up -d
docker compose logs -f
```

## 访问

| 入口 | URL |
|------|-----|
| TrustClaw Control UI | `http://<host>:19001/?token=<OPENCLAW_GATEWAY_TOKEN>` |
| TRA Console | `http://<host>:19001/trustclaw/` |
| vLLM OpenAI API | `http://<host>:8000/v1` |
| Unsloth Studio 连接 | vLLM → Base URL 同上，Bearer `VLLM_API_KEY` |

## 离线部署

目标 MetaX 机器：

```bash
docker load -i metax-full-images.tar
# 拷贝 docker/metax-full/{docker-compose.yml,app.env}
docker compose up -d
```

## 说明

- vLLM 容器**不内置** MACA 驱动，通过 volume 挂载宿主机 `/opt/maca`、`/opt/conda`
- 模型目录默认只读挂载 `/data/models`
- TrustClaw 状态持久化在 volume `trustclaw-data`
- 默认模型：`vllm//data/models/Qwen3.6-27B-AWQ`，thinking 关闭

## 停止

```bash
cd docker/metax-full
docker compose down
```
