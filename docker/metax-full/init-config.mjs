#!/usr/bin/env node
/**
 * First-boot OpenClaw config for MetaX full stack (TrustClaw + vLLM).
 */
import { cpSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

const stateDir = process.env.OPENCLAW_STATE_DIR ?? "/home/node/.openclaw";
const configPath = process.env.OPENCLAW_CONFIG_PATH ?? path.join(stateDir, "openclaw.json");
const seedPath = process.env.TRUSTCLAW_CONFIG_SEED ?? "/opt/metax-full/config/openclaw.json.seed";
const appRoot = process.env.TRUSTCLAW_APP_ROOT ?? "/app";

function envTrim(key) {
  const value = process.env[key];
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function loadJson(filePath, fallback) {
  if (!existsSync(filePath)) {
    return structuredClone(fallback);
  }
  try {
    return JSON.parse(readFileSync(filePath, "utf8"));
  } catch {
    return structuredClone(fallback);
  }
}

function saveJson(filePath, data) {
  mkdirSync(path.dirname(filePath), { recursive: true });
  writeFileSync(filePath, `${JSON.stringify(data, null, 2)}\n`);
}

function syncWorkspaceTemplate(templateDir, targetDir) {
  if (!existsSync(templateDir)) {
    return;
  }
  mkdirSync(targetDir, { recursive: true });
  for (const name of ["SOUL.md", "IDENTITY.md", "AGENTS.md"]) {
    const src = path.join(templateDir, name);
    if (existsSync(src)) {
      cpSync(src, path.join(targetDir, name), { force: true });
    }
  }
}

function syncTrustclawWorkspaces() {
  const workspaceRoot = path.join(appRoot, "trustclaw", "workspace");
  const mappings = [
    { template: "dev", target: path.join(stateDir, "workspace") },
    { template: "nrdl-reimburse", target: path.join(stateDir, "workspace-nrdl-reimburse") },
    { template: "compliance-auditor", target: path.join(stateDir, "workspace-compliance-auditor") },
  ];
  for (const entry of mappings) {
    syncWorkspaceTemplate(path.join(workspaceRoot, entry.template), entry.target);
  }
}

function buildControlUiOrigins() {
  const explicit = envTrim("TRUSTCLAW_CONTROL_UI_ORIGINS");
  if (explicit) {
    return explicit
      .split(/[,\s]+/)
      .map((origin) => origin.trim())
      .filter(Boolean);
  }
  const hostPorts = [
    envTrim("APP_PORT") ?? "19001",
    envTrim("TRUSTCLAW_UI_PORT") ?? "19001",
  ];
  const origins = new Set();
  for (const port of hostPorts) {
    origins.add(`http://127.0.0.1:${port}`);
    origins.add(`http://localhost:${port}`);
  }
  return [...origins];
}

function applyEnvToConfig(config) {
  const gatewayToken = envTrim("OPENCLAW_GATEWAY_TOKEN") ?? "change-me-trustclaw-metax";
  const gatewayPort = Number(envTrim("OPENCLAW_GATEWAY_PORT") ?? "19001");
  const vllmBaseUrl = envTrim("VLLM_BASE_URL") ?? "http://vllm:8000/v1";
  const vllmApiKey = envTrim("VLLM_API_KEY") ?? "sk-unsloth-metax-docker";
  const vllmModelId = envTrim("VLLM_MODEL_ID") ?? "/data/models/Qwen3.6-27B-AWQ";
  const primaryModel = envTrim("OPENCLAW_PRIMARY_MODEL") ?? `vllm/${vllmModelId}`;
  const agentPacksDir =
    envTrim("TRUSTCLAW_AGENT_PACKS_DIR") ?? path.join(appRoot, "trustclaw", "agents");

  config.gateway = {
    mode: "local",
    bind: "lan",
    ...config.gateway,
    port: Number.isFinite(gatewayPort) ? gatewayPort : 19001,
    auth: {
      mode: "token",
      token: gatewayToken,
    },
    controlUi: {
      ...(config.gateway?.controlUi ?? {}),
      allowInsecureAuth: true,
      allowedOrigins: buildControlUiOrigins(),
    },
  };

  config.plugins = {
    ...config.plugins,
    entries: {
      ...(config.plugins?.entries ?? {}),
      "trustclaw-tra": {
        enabled: true,
        config: {
          agentPacksDir,
          defaultAgentPack: envTrim("TRUSTCLAW_DEFAULT_AGENT_PACK") ?? "glp1-eligibility",
        },
      },
      acpx: { enabled: false },
      workboard: { enabled: true },
    },
  };

  config.env = {
    ...(config.env ?? {}),
    VLLM_API_KEY: vllmApiKey,
  };

  config.models = {
    ...(config.models ?? {}),
    providers: {
      ...(config.models?.providers ?? {}),
      vllm: {
        baseUrl: vllmBaseUrl,
        apiKey: "${VLLM_API_KEY}",
        api: "openai-completions",
        timeoutSeconds: 300,
        request: { allowPrivateNetwork: true },
        models: [
          {
            id: vllmModelId,
            name: "Qwen3.6-27B-AWQ (MetaX vLLM)",
            reasoning: false,
            input: ["text"],
            compat: { thinkingFormat: "qwen-chat-template" },
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
            contextWindow: Number(envTrim("VLLM_MAX_MODEL_LEN") ?? "8192"),
            maxTokens: 4096,
          },
        ],
      },
    },
  };

  config.agents = {
    ...(config.agents ?? {}),
    defaults: {
      ...(config.agents?.defaults ?? {}),
      workspace: path.join(stateDir, "workspace"),
      model: { primary: primaryModel, fallbacks: [] },
      models: {
        ...(config.agents?.defaults?.models ?? {}),
        [primaryModel]: {
          alias: "qwen36-awq",
          params: { chat_template_kwargs: { enable_thinking: false } },
        },
        "vllm/*": {},
      },
      compaction: { mode: "safeguard" },
    },
    list:
      Array.isArray(config.agents?.list) && config.agents.list.length > 0
        ? config.agents.list
        : [
            {
              id: "main",
              default: true,
              workspace: path.join(stateDir, "workspace"),
              agentDir: path.join(stateDir, "agents", "main", "agent"),
            },
          ],
  };

  return config;
}

function main() {
  mkdirSync(stateDir, { recursive: true });
  mkdirSync(path.join(stateDir, "agents", "main", "agent"), { recursive: true });

  const seed = loadJson(seedPath, {});
  const existing = existsSync(configPath) ? loadJson(configPath, seed) : seed;
  const merged = applyEnvToConfig(existing);
  saveJson(configPath, merged);
  syncTrustclawWorkspaces();

  console.log(`[metax-full] Config ready at ${configPath}`);
  console.log(`[metax-full] Primary model: ${merged.agents?.defaults?.model?.primary}`);
}

main();
