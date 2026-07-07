#!/usr/bin/env python3
"""Automated tuning loop toward AGENT.md §12.5 acceptance targets on MetaX C500.

Runs a grid of vLLM configs + bench modes, keeps best results, stops early on PASS.

Usage (MetaX server):
  bash scripts/tune_targets_loop.sh
  MAX_LOOPS=12 bash scripts/tune_targets_loop.sh
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
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    "phase0_single_tok_s": 9.5,
    "phase1_concurrent_8_tok_s": 40.0,
    "phase3_mtp_tok_s": 20.0,
    "fused_rope_rms_ms": 0.5,
}


@dataclass
class VllmConfig:
    name: str
    gpu_mem: float
    max_seqs: int
    max_batched: int
    prefix_cache: bool = True
    max_model_len: int = 8192


@dataclass
class BenchConfig:
    name: str
    api: str = "completions"
    no_think: bool = False
    temperature: float = 0.0
    prompt: str = "请用中文写一段约120字的自我介绍，不要换行。"
    max_tokens: int = 128


@dataclass
class LoopResult:
    loop_id: int
    vllm: VllmConfig
    bench: BenchConfig
    phase0_tok_s: Optional[float] = None
    phase1_c8_tok_s: Optional[float] = None
    phase0_pass: bool = False
    phase1_pass: bool = False
    startup_s: Optional[float] = None
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


def _start_vllm(model: str, host: str, port: int, cfg: VllmConfig, log_path: Path) -> subprocess.Popen[Any]:
    cmd = [
        "vllm", "serve", model,
        "--host", host,
        "--port", str(port),
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
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        return subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)


def _stop_vllm() -> None:
    subprocess.run(["pkill", "-f", "vllm serve"], check=False)
    time.sleep(3)


def _run_bench(
    repo: Path,
    url: str,
    bench: BenchConfig,
    concurrency: int,
    output: Path,
) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(repo / "scripts" / "bench_qwen36.py"),
        "--url", url,
        "--prompt", bench.prompt,
        "--max-tokens", str(bench.max_tokens),
        "--temperature", str(bench.temperature),
        "--concurrency", str(concurrency),
        "--requests", str(concurrency),
        "--warmup-requests", "1",
        "--api", bench.api,
        "--stream",
        "--json",
        "--output", str(output),
    ]
    if bench.no_think:
        cmd.append("--no-think")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"bench exit {proc.returncode}")
    if output.exists():
        return json.loads(output.read_text())
    return json.loads(proc.stdout)


def _build_grid() -> List[tuple[VllmConfig, BenchConfig]]:
    vllm_grid = [
        VllmConfig("base", 0.92, 64, 8192),
        VllmConfig("high-mem", 0.95, 64, 8192),
        VllmConfig("high-mem-seq128", 0.95, 128, 8192),
        VllmConfig("high-mem-batch16k", 0.95, 128, 16384),
        VllmConfig("aggressive", 0.97, 128, 16384, prefix_cache=False, max_model_len=4096),
    ]
    bench_grid = [
        BenchConfig("completions-default", api="completions", temperature=0.7,
                    prompt="你好，请用一句话介绍你自己。"),
        BenchConfig("completions-t0", api="completions", temperature=0.0,
                    prompt="请用中文写一段约120字的自我介绍，不要换行。"),
        BenchConfig("chat-no-think", api="chat", no_think=True, temperature=0.0,
                    prompt="请用中文写一段约120字的自我介绍，不要换行。"),
        BenchConfig("chat-no-think-short", api="chat", no_think=True, temperature=0.0,
                    prompt="1+1等于几？请只回答数字。", max_tokens=16),
    ]
    combos: List[tuple[VllmConfig, BenchConfig]] = []
    for v in vllm_grid:
        for b in bench_grid:
            combos.append((v, b))
    return combos


def _score(result: LoopResult) -> float:
    s = 0.0
    if result.phase0_tok_s:
        s += result.phase0_tok_s
    if result.phase1_c8_tok_s:
        s += result.phase1_c8_tok_s * 2
    if result.phase0_pass:
        s += 100
    if result.phase1_pass:
        s += 200
    return s


def run_loop(args: argparse.Namespace) -> int:
    repo = Path(args.repo).resolve()
    log_root = Path(args.log_root)
    log_root.mkdir(parents=True, exist_ok=True)
    url = f"http://{args.host}:{args.port}"
    grid = _build_grid()[: args.max_loops]

    results: List[LoopResult] = []
    best: Optional[LoopResult] = None

    for i, (v_cfg, b_cfg) in enumerate(grid, start=1):
        loop = LoopResult(loop_id=i, vllm=v_cfg, bench=b_cfg)
        print(f"\n=== Loop {i}/{len(grid)}: vllm={v_cfg.name} bench={b_cfg.name} ===", flush=True)
        _stop_vllm()
        vllm_log = log_root / f"vllm-loop{i}-{v_cfg.name}.log"
        t0 = time.time()
        try:
            proc = _start_vllm(args.model, args.host, args.port, v_cfg, vllm_log)
            if not _wait_vllm(url, timeout_s=args.startup_timeout):
                loop.error = "vLLM startup timeout"
                proc.kill()
                results.append(loop)
                continue
            loop.startup_s = round(time.time() - t0, 1)

            p0_out = log_root / f"loop{i}-c1.json"
            p0 = _run_bench(repo, url, b_cfg, 1, p0_out)
            loop.phase0_tok_s = p0["summary"].get("aggregate_tokens_per_s")
            loop.phase0_pass = bool(loop.phase0_tok_s and loop.phase0_tok_s >= TARGETS["phase0_single_tok_s"])

            p1_out = log_root / f"loop{i}-c8.json"
            p1 = _run_bench(repo, url, b_cfg, 8, p1_out)
            loop.phase1_c8_tok_s = p1["summary"].get("aggregate_tokens_per_s")
            loop.phase1_pass = bool(loop.phase1_c8_tok_s and loop.phase1_c8_tok_s >= TARGETS["phase1_concurrent_8_tok_s"])

            print(
                f"  phase0={loop.phase0_tok_s} ({'PASS' if loop.phase0_pass else 'fail'}) "
                f"phase1_c8={loop.phase1_c8_tok_s} ({'PASS' if loop.phase1_pass else 'fail'})",
                flush=True,
            )
        except Exception as exc:
            loop.error = str(exc)
            print(f"  ERROR: {exc}", flush=True)
        finally:
            _stop_vllm()
            results.append(loop)
            if best is None or _score(loop) > _score(best):
                best = loop
            if loop.phase0_pass and loop.phase1_pass:
                print("All Phase 0/1 targets met — stopping early.", flush=True)
                break

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "targets": TARGETS,
        "loops_run": len(results),
        "best": asdict(best) if best else None,
        "results": [asdict(r) for r in results],
    }
    json_path = log_root / "TUNE_LOOP_RESULTS.json"
    md_path = log_root / "TUNE_LOOP_RESULTS.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    lines = [
        f"# Target Tuning Loop — {report['timestamp']}",
        "",
        "## Targets",
        "",
        "| Metric | Target | Best | Status |",
        "|--------|--------|------|--------|",
    ]
    if best:
        lines += [
            f"| phase0_single | {TARGETS['phase0_single_tok_s']} | {best.phase0_tok_s} | {'PASS' if best.phase0_pass else 'FAIL'} |",
            f"| phase1_c8 | {TARGETS['phase1_concurrent_8_tok_s']} | {best.phase1_c8_tok_s} | {'PASS' if best.phase1_pass else 'FAIL'} |",
            "",
            f"**Best combo:** vllm=`{best.vllm.name}` bench=`{best.bench.name}`",
            "",
            "## All loops",
            "",
            "| Loop | vLLM | Bench | c1 tok/s | c8 tok/s | Error |",
            "|------|------|-------|----------|----------|-------|",
        ]
        for r in results:
            lines.append(
                f"| {r.loop_id} | {r.vllm.name} | {r.bench.name} | "
                f"{r.phase0_tok_s or '-'} | {r.phase1_c8_tok_s or '-'} | {r.error or ''} |"
            )
    md_path.write_text("\n".join(lines) + "\n")

    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    if best and best.phase0_pass and best.phase1_pass:
        return 0
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune vLLM configs toward acceptance targets")
    parser.add_argument("--repo", default=str(REPO_ROOT))
    parser.add_argument("--model", default="/data/models/Qwen3.6-27B-AWQ")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-root", default="/data/metax-test-logs/tune")
    parser.add_argument("--max-loops", type=int, default=int(os.environ.get("MAX_LOOPS", "8")))
    parser.add_argument("--startup-timeout", type=int, default=900)
    return run_loop(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
