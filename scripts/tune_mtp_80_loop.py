#!/usr/bin/env python3
"""MTP speculative tuning loop toward 80 tok/s on MetaX C500.

Sweeps MTP draft tokens, concurrency, vLLM batch params, and prompts.
Requires warmup (bench --warmup-requests 1) for accurate t=0 measurement.

Usage:
  bash scripts/tune_mtp_80_loop.sh
  MTP_TARGET=80 MAX_LOOPS=16 bash scripts/tune_mtp_80_loop.sh
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
COMP_CONFIG_NONE = '{"cudagraph_mode":"none"}'
DEFAULT_TARGET = 80.0
PROMPT_LONG = "请用中文写一段约120字的自我介绍，不要换行。"
PROMPT_SHORT = "你好，请用一句话介绍你自己。"


@dataclass
class VllmCfg:
    name: str
    gpu_mem: float = 0.92
    max_seqs: int = 64
    max_batched: int = 8192
    prefix_cache: bool = True


@dataclass
class MtpLoop:
    loop_id: int
    vllm: str
    mtp_tokens: int
    concurrency: int
    prompt: str
    max_tokens: int
    tok_s: Optional[float] = None
    pass_target: bool = False
    error: Optional[str] = None


def _wait_vllm(url: str, timeout_s: int = 900) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url.rstrip('/')}/v1/models", timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(5)
    return False


def _stop_vllm() -> None:
    subprocess.run(["pkill", "-f", "vllm serve"], check=False)
    time.sleep(3)


def _start_vllm(model: str, host: str, port: int, cfg: VllmCfg, mtp_tokens: int, log: Path) -> None:
    spec = json.dumps({"method": "mtp", "num_speculative_tokens": mtp_tokens})
    cmd = [
        "vllm", "serve", model,
        "--host", host, "--port", str(port),
        "--tensor-parallel-size", "1",
        "--max-model-len", "8192",
        "--dtype", "auto",
        "--gpu-memory-utilization", str(cfg.gpu_mem),
        "--max-num-batched-tokens", str(cfg.max_batched),
        "--max-num-seqs", str(cfg.max_seqs),
        "--enable-chunked-prefill",
        "--trust-remote-code",
        "--compilation-config", COMP_CONFIG_NONE,
        "--speculative-config", spec,
        "--reasoning-parser", "qwen3",
    ]
    if cfg.prefix_cache:
        cmd.append("--enable-prefix-caching")
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8") as f:
        subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)


def _bench(
    repo: Path,
    url: str,
    prompt: str,
    concurrency: int,
    max_tokens: int,
    out: Path,
) -> float:
    cmd = [
        sys.executable, str(repo / "scripts" / "bench_qwen36.py"),
        "--url", url,
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--temperature", "0",
        "--warmup-requests", "1",
        "--concurrency", str(concurrency),
        "--requests", str(concurrency),
        "--stream", "--json", "--output", str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    data = json.loads(out.read_text())
    return float(data["summary"]["aggregate_tokens_per_s"])


def _build_grid(max_loops: int) -> List[tuple[VllmCfg, int, int, str, int]]:
    vllm_cfgs = [
        VllmCfg("base"),
        VllmCfg("high-mem", gpu_mem=0.95, max_seqs=128),
        VllmCfg("aggressive", gpu_mem=0.97, max_seqs=128, max_batched=16384),
    ]
    grid: List[tuple[VllmCfg, int, int, str, int]] = []
    # Prioritize high concurrency + long prompt (Phase 1 recipe under MTP)
    for v in vllm_cfgs:
        for mtp in (2, 3, 4, 5, 8):
            for conc in (8, 4, 1):
                for prompt, max_tok in ((PROMPT_LONG, 128), (PROMPT_SHORT, 128)):
                    grid.append((v, mtp, conc, prompt, max_tok))
    return grid[:max_loops]


def run(args: argparse.Namespace) -> int:
    repo = Path(args.repo)
    log_root = Path(args.log_root)
    log_root.mkdir(parents=True, exist_ok=True)
    url = f"http://{args.host}:{args.port}"
    target = args.target
    grid = _build_grid(args.max_loops)

    results: List[MtpLoop] = []
    best: Optional[MtpLoop] = None

    for i, (vcfg, mtp, conc, prompt, max_tok) in enumerate(grid, start=1):
        loop = MtpLoop(i, vcfg.name, mtp, conc, prompt[:30], max_tok)
        print(f"\n=== MTP loop {i}/{len(grid)}: {vcfg.name} mtp={mtp} c={conc} ===", flush=True)
        _stop_vllm()
        try:
            log = log_root / f"vllm-mtp80-{i}.log"
            _start_vllm(args.model, args.host, args.port, vcfg, mtp, log)
            if not _wait_vllm(url, timeout_s=args.startup_timeout):
                raise RuntimeError("vLLM startup timeout")
            out = log_root / f"mtp80-loop{i}.json"
            loop.tok_s = _bench(repo, url, prompt, conc, max_tok, out)
            loop.pass_target = loop.tok_s >= target
            print(f"  tok/s={loop.tok_s:.2f} target={target} pass={loop.pass_target}", flush=True)
        except Exception as exc:
            loop.error = str(exc)
            print(f"  ERROR: {exc}", flush=True)
        finally:
            _stop_vllm()
        results.append(loop)
        if best is None or (loop.tok_s or 0) > (best.tok_s or 0):
            best = loop
        if loop.pass_target:
            print(f"  Target {target} tok/s reached — continuing sweep for best.", flush=True)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target_tok_s": target,
        "best": asdict(best) if best else None,
        "results": [asdict(r) for r in results],
    }
    json_path = log_root / "MTP80_LOOP_RESULTS.json"
    md_path = log_root / "MTP80_LOOP_RESULTS.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    lines = [
        f"# MTP 80 tok/s Tuning Loop — {report['timestamp']}",
        "",
        f"Target: **{target} tok/s** aggregate",
        f"Best: **{best.tok_s if best else '—'} tok/s** "
        f"({best.vllm if best else '—'} mtp={best.mtp_tokens if best else '—'} c={best.concurrency if best else '—'})",
        "",
        "| Loop | vLLM | MTP | c | tok/s | Pass | Error |",
        "|------|------|-----|---|-------|------|-------|",
    ]
    for r in results:
        lines.append(
            f"| {r.loop_id} | {r.vllm} | {r.mtp_tokens} | {r.concurrency} | "
            f"{r.tok_s or '-'} | {r.pass_target} | {r.error or ''} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {json_path}\nWrote {md_path}")
    return 0 if best and (best.tok_s or 0) >= target else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default=str(REPO_ROOT))
    p.add_argument("--model", default="/data/models/Qwen3.6-27B-AWQ")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--log-root", default="/data/metax-test-logs/tune/mtp80")
    p.add_argument("--target", type=float, default=float(os.environ.get("MTP_TARGET", DEFAULT_TARGET)))
    p.add_argument("--max-loops", type=int, default=int(os.environ.get("MAX_LOOPS", "12")))
    p.add_argument("--startup-timeout", type=int, default=900)
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
