#!/usr/bin/env python3
"""No-MTP c=8 aggregate tuning loop toward 160 tok/s on MetaX C500.

Focus: vLLM batch/KV settings, concurrency, prompt length, max_tokens.
Does NOT use MTP (MTP caps ~40-47 tok/s on AWQ).

Usage:
  bash scripts/tune_c8_160_loop.sh
  C8_TARGET=160 MAX_LOOPS=20 bash scripts/tune_c8_160_loop.sh
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
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = 160.0
PROMPT_LONG = "请用中文写一段约120字的自我介绍，不要换行。"
PROMPT_SHORT = "你好，请用一句话介绍你自己。"


@dataclass
class VllmCfg:
    name: str
    gpu_mem: float = 0.92
    max_seqs: int = 64
    max_batched: int = 8192
    max_model_len: int = 8192
    prefix_cache: bool = True


@dataclass
class BenchCfg:
    name: str
    prompt: str
    max_tokens: int
    api: str = "completions"
    no_think: bool = False
    temperature: float = 0.0


@dataclass
class C8Loop:
    loop_id: int
    vllm: str
    bench: str
    concurrency: int
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


def _start_vllm(model: str, host: str, port: int, cfg: VllmCfg, log: Path) -> None:
    cmd = [
        "vllm", "serve", model,
        "--host", host, "--port", str(port),
        "--tensor-parallel-size", "1",
        "--max-model-len", str(cfg.max_model_len),
        "--dtype", "auto",
        "--gpu-memory-utilization", str(cfg.gpu_mem),
        "--max-num-batched-tokens", str(cfg.max_batched),
        "--max-num-seqs", str(cfg.max_seqs),
        "--enable-chunked-prefill",
        "--trust-remote-code",
    ]
    if cfg.prefix_cache:
        cmd.append("--enable-prefix-caching")
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8") as f:
        subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)


def _bench(
    repo: Path,
    url: str,
    bench: BenchCfg,
    concurrency: int,
    out: Path,
) -> float:
    cmd = [
        sys.executable, str(repo / "scripts" / "bench_qwen36.py"),
        "--url", url,
        "--prompt", bench.prompt,
        "--max-tokens", str(bench.max_tokens),
        "--temperature", str(bench.temperature),
        "--warmup-requests", "1",
        "--concurrency", str(concurrency),
        "--requests", str(concurrency),
        "--api", bench.api,
        "--stream", "--json", "--output", str(out),
    ]
    if bench.no_think:
        cmd.append("--no-think")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    data = json.loads(out.read_text())
    return float(data["summary"]["aggregate_tokens_per_s"])


def _build_refine_grid() -> List[Tuple[VllmCfg, BenchCfg, int]]:
    """Second-pass grid around best c=16 configs (~151 tok/s)."""
    bench_long = BenchCfg("t0-long", PROMPT_LONG, 128)
    bench_64 = BenchCfg("t0-long-64", PROMPT_LONG, 64)
    cfgs = [
        (VllmCfg("aggressive-seq256", 0.97, 256, 16384), bench_long, 16),
        (VllmCfg("aggressive-seq256", 0.97, 256, 16384), bench_long, 18),
        (VllmCfg("aggressive", 0.97, 128, 16384), bench_long, 16),
        (VllmCfg("aggressive", 0.97, 128, 16384), bench_long, 18),
        (VllmCfg("aggressive-seq256", 0.97, 256, 16384), bench_64, 16),
        (VllmCfg("aggressive-seq256", 0.97, 256, 16384), bench_64, 18),
        (VllmCfg("high-mem-batch16k", 0.95, 128, 16384), bench_long, 16),
        (VllmCfg("high-mem-batch16k", 0.95, 128, 16384), bench_long, 18),
        (VllmCfg("aggressive-seq256", 0.97, 256, 16384), bench_long, 20),
        (VllmCfg("tight-kv", 0.97, 256, 16384, 4096), bench_long, 16),
    ]
    return cfgs


def _build_grid(max_loops: int) -> List[Tuple[VllmCfg, BenchCfg, int]]:
    if os.environ.get("REFINE", "0") == "1":
        return _build_refine_grid()[:max_loops]
    vllm_cfgs = [
        VllmCfg("base"),
        VllmCfg("high-mem", gpu_mem=0.95, max_seqs=128),
        VllmCfg("high-mem-batch16k", gpu_mem=0.95, max_seqs=128, max_batched=16384),
        VllmCfg("aggressive", gpu_mem=0.97, max_seqs=128, max_batched=16384),
        VllmCfg("aggressive-seq256", gpu_mem=0.97, max_seqs=256, max_batched=16384),
        VllmCfg("aggressive-32k", gpu_mem=0.97, max_seqs=128, max_batched=32768),
        VllmCfg("tight-kv", gpu_mem=0.97, max_seqs=256, max_batched=16384, max_model_len=4096),
        VllmCfg("no-prefix", gpu_mem=0.97, max_seqs=128, max_batched=16384, prefix_cache=False),
    ]
    bench_cfgs = [
        BenchCfg("t0-long", PROMPT_LONG, 128),
        BenchCfg("t0-long-64", PROMPT_LONG, 64),
        BenchCfg("t0-short", PROMPT_SHORT, 128),
        BenchCfg("chat-long", PROMPT_LONG, 128, api="chat", no_think=True),
        BenchCfg("chat-short", PROMPT_SHORT, 128, api="chat", no_think=True),
    ]
    concurrencies = (8, 12, 16, 24, 32)

    grid: List[Tuple[VllmCfg, BenchCfg, int]] = []
    # Prioritize known-good baseline first, then scale concurrency
    priority: List[Tuple[VllmCfg, BenchCfg, int]] = [
        (VllmCfg("high-mem-batch16k", 0.95, 128, 16384), bench_cfgs[0], 8),
        (VllmCfg("aggressive", 0.97, 128, 16384), bench_cfgs[0], 8),
        (VllmCfg("aggressive", 0.97, 128, 16384), bench_cfgs[0], 16),
        (VllmCfg("aggressive-seq256", 0.97, 256, 16384), bench_cfgs[0], 16),
        (VllmCfg("aggressive-seq256", 0.97, 256, 16384), bench_cfgs[0], 24),
        (VllmCfg("aggressive-32k", 0.97, 128, 32768), bench_cfgs[0], 16),
        (VllmCfg("tight-kv", 0.97, 256, 16384, 4096), bench_cfgs[1], 16),
        (VllmCfg("aggressive", 0.97, 128, 16384), bench_cfgs[1], 16),
    ]
    seen = set()
    for item in priority:
        key = (item[0].name, item[1].name, item[2])
        if key not in seen:
            seen.add(key)
            grid.append(item)

    for v in vllm_cfgs:
        for b in bench_cfgs:
            for c in concurrencies:
                key = (v.name, b.name, c)
                if key not in seen:
                    seen.add(key)
                    grid.append((v, b, c))
    return grid[:max_loops]


def run(args: argparse.Namespace) -> int:
    repo = Path(args.repo)
    log_root = Path(args.log_root)
    log_root.mkdir(parents=True, exist_ok=True)
    url = f"http://{args.host}:{args.port}"
    target = args.target
    grid = _build_grid(args.max_loops)

    results: List[C8Loop] = []
    best: Optional[C8Loop] = None

    for i, (vcfg, bcfg, conc) in enumerate(grid, start=1):
        loop = C8Loop(i, vcfg.name, bcfg.name, conc)
        print(
            f"\n=== C8-160 loop {i}/{len(grid)}: {vcfg.name} {bcfg.name} c={conc} ===",
            flush=True,
        )
        _stop_vllm()
        try:
            log = log_root / f"vllm-c8160-{i}.log"
            _start_vllm(args.model, args.host, args.port, vcfg, log)
            if not _wait_vllm(url, timeout_s=args.startup_timeout):
                raise RuntimeError("vLLM startup timeout")
            out = log_root / f"c8160-loop{i}.json"
            loop.tok_s = _bench(repo, url, bcfg, conc, out)
            loop.pass_target = loop.tok_s >= target
            print(f"  aggregate tok/s={loop.tok_s:.2f} target={target} pass={loop.pass_target}", flush=True)
        except Exception as exc:
            loop.error = str(exc)
            print(f"  ERROR: {exc}", flush=True)
        finally:
            _stop_vllm()
        results.append(loop)
        if best is None or (loop.tok_s or 0) > (best.tok_s or 0):
            best = loop
        if loop.pass_target:
            print(f"  Target {target} tok/s reached.", flush=True)
            if args.stop_on_pass:
                break

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target_aggregate_tok_s": target,
        "note": "aggregate_tokens_per_s at given concurrency (no MTP)",
        "best": asdict(best) if best else None,
        "results": [asdict(r) for r in results],
    }
    json_path = log_root / "C8160_LOOP_RESULTS.json"
    md_path = log_root / "C8160_LOOP_RESULTS.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    lines = [
        f"# C8 Aggregate 160 tok/s Tuning Loop — {report['timestamp']}",
        "",
        f"Target: **{target} tok/s** aggregate (no MTP)",
        f"Best: **{best.tok_s if best else '—'} tok/s** "
        f"({best.vllm if best else '—'} / {best.bench if best else '—'} c={best.concurrency if best else '—'})",
        "",
        "| Loop | vLLM | Bench | c | tok/s | Pass | Error |",
        "|------|------|-------|---|-------|------|-------|",
    ]
    for r in results:
        lines.append(
            f"| {r.loop_id} | {r.vllm} | {r.bench} | {r.concurrency} | "
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
    p.add_argument("--log-root", default=os.environ.get("LOG_ROOT", "/data/metax-test-logs/tune/c8160"))
    p.add_argument("--target", type=float, default=float(os.environ.get("C8_TARGET", DEFAULT_TARGET)))
    p.add_argument("--max-loops", type=int, default=int(os.environ.get("MAX_LOOPS", "20")))
    p.add_argument("--startup-timeout", type=int, default=900)
    p.add_argument("--stop-on-pass", action="store_true", default=os.environ.get("STOP_ON_PASS", "0") == "1")
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
