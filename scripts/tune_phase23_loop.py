#!/usr/bin/env python3
"""Phase 2 (op bench) + Phase 3 (speculative) tuning loops toward remaining targets."""

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
TARGETS = {
    "fused_rope_rms_ms": 0.5,
    "phase3_mtp_tok_s": 20.0,
}
COMP_CONFIG_NONE = '{"cudagraph_mode":"none"}'
PHASE1_PROMPT = "请用中文写一段约120字的自我介绍，不要换行。"


@dataclass
class OpLoop:
    loop_id: int
    seq_len: int
    dtype: str
    impl: str
    iters: int
    avg_ms: Optional[float] = None
    pass_target: bool = False
    error: Optional[str] = None


@dataclass
class Phase3Loop:
    loop_id: int
    mode: str
    spec_json: str
    temperature: float
    prompt: str
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


def _run_op_bench(repo: Path, seq_len: int, dtype: str, impl: str, iters: int, out: Path) -> float:
    code = f"""
import json, sys
sys.path.insert(0, {str(repo)!r})
from metax_kernels.bench.op_bench import bench_fused_rope_rms
import torch
dtype_map = {{"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}}
device = torch.device("cuda")
dtype = dtype_map[{dtype!r}]
r = bench_fused_rope_rms(1, {seq_len}, 5120, 40, 8, dtype, device, {impl!r}, 10, {iters})
print(json.dumps(r))
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    line = proc.stdout.strip().splitlines()[-1]
    data = json.loads(line)
    if "error" in data:
        raise RuntimeError(data["error"])
    out.write_text(json.dumps(data, indent=2))
    return float(data["avg_ms"])


def _start_vllm(model: str, host: str, port: int, extra: List[str], log_path: Path, no_cg: bool) -> None:
    cmd = [
        "vllm", "serve", model,
        "--host", host, "--port", str(port),
        "--tensor-parallel-size", "1",
        "--max-model-len", "8192",
        "--dtype", "auto",
        "--gpu-memory-utilization", "0.92",
        "--max-num-batched-tokens", "8192",
        "--max-num-seqs", "64",
        "--enable-chunked-prefill",
        "--trust-remote-code",
    ]
    if no_cg:
        cmd += ["--compilation-config", COMP_CONFIG_NONE]
    cmd += extra
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)


def _run_e2e_bench(
    repo: Path, url: str, prompt: str, temperature: float, out: Path
) -> float:
    cmd = [
        sys.executable, str(repo / "scripts" / "bench_qwen36.py"),
        "--url", url,
        "--prompt", prompt,
        "--max-tokens", "128",
        "--temperature", str(temperature),
        "--concurrency", "1",
        "--stream", "--json", "--output", str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    data = json.loads(out.read_text())
    return float(data["summary"]["aggregate_tokens_per_s"])


def _build_op_grid(max_loops: int) -> List[tuple[int, str, str, int]]:
    """Prioritize S=256 (acceptance target) before decode S=1 sweeps."""
    s256 = [
        (256, "bfloat16", impl, iters)
        for impl in ("fast", "eager", "compiled", "fused", "opt_eager")
        for iters in (50, 30)
    ]
    s1 = [
        (1, "bfloat16", impl, 30)
        for impl in ("fast", "eager", "fused")
    ]
    grid = s256 + s1
    return grid[:max_loops]


def run_phase2_loops(repo: Path, log_root: Path, max_loops: int) -> List[OpLoop]:
    grid = _build_op_grid(max_loops)
    results: List[OpLoop] = []
    best_ms: Optional[float] = None

    for i, (seq_len, dtype, impl, iters) in enumerate(grid, start=1):
        loop = OpLoop(i, seq_len, dtype, impl, iters)
        print(f"\n=== Phase2 loop {i}: S={seq_len} dtype={dtype} impl={impl} iters={iters} ===", flush=True)
        try:
            out = log_root / f"phase2-loop{i}.json"
            loop.avg_ms = _run_op_bench(repo, seq_len, dtype, impl, iters, out)
            if seq_len == 256:
                loop.pass_target = loop.avg_ms <= TARGETS["fused_rope_rms_ms"]
            print(f"  avg_ms={loop.avg_ms:.4f} pass@S256={loop.pass_target}", flush=True)
            if seq_len == 256 and (best_ms is None or loop.avg_ms < best_ms):
                best_ms = loop.avg_ms
                if loop.pass_target:
                    print("  Phase2 target met at S=256 — continuing sweep for margin.", flush=True)
        except Exception as exc:
            loop.error = str(exc)
            print(f"  ERROR: {exc}", flush=True)
        results.append(loop)
    return results


def run_phase3_loops(repo: Path, log_root: Path, model: str, host: str, port: int, max_loops: int) -> List[Phase3Loop]:
    url = f"http://{host}:{port}"
    grid: List[tuple[str, str, float, str, bool]] = [
        ("baseline-default-t0", "", 0.0, "你好，请用一句话介绍你自己。", False),
        ("baseline-long-t0", "", 0.0, PHASE1_PROMPT, False),
        ("ngram-8-default", '{"method":"ngram","num_speculative_tokens":8,"prompt_lookup_max":8}', 0.0,
         "你好，请用一句话介绍你自己。", True),
        ("ngram-8-long", '{"method":"ngram","num_speculative_tokens":8,"prompt_lookup_max":8}', 0.0, PHASE1_PROMPT, True),
        ("ngram-repeat", '{"method":"ngram","num_speculative_tokens":8,"prompt_lookup_max":8}', 0.0,
         "重复测试：" + "你好 " * 20 + "请继续写下去。", True),
        ("mtp-2-default", '{"method":"mtp","num_speculative_tokens":2}', 0.0,
         "你好，请用一句话介绍你自己。", True),
        ("mtp-2-long", '{"method":"mtp","num_speculative_tokens":2}', 0.0, PHASE1_PROMPT, True),
    ]
    grid = grid[:max_loops]

    results: List[Phase3Loop] = []
    for i, (mode, spec, temp, prompt, no_cg) in enumerate(grid, start=1):
        loop = Phase3Loop(i, mode, spec, temp, prompt)
        print(f"\n=== Phase3 loop {i}: mode={mode} t={temp} ===", flush=True)
        _stop_vllm()
        extra: List[str] = []
        if spec:
            extra += ["--speculative-config", spec]
        if mode.startswith("mtp"):
            extra += ["--reasoning-parser", "qwen3"]
        try:
            log = log_root / f"phase3-loop{i}-{mode}.log"
            _start_vllm(model, host, port, extra, log, no_cg=no_cg)
            if not _wait_vllm(url):
                raise RuntimeError("vLLM startup timeout")
            out = log_root / f"phase3-loop{i}.json"
            loop.tok_s = _run_e2e_bench(repo, url, prompt, temp, out)
            loop.pass_target = bool(loop.tok_s >= TARGETS["phase3_mtp_tok_s"])
            print(f"  tok/s={loop.tok_s} pass={loop.pass_target}", flush=True)
        except Exception as exc:
            loop.error = str(exc)
            print(f"  ERROR: {exc}", flush=True)
        finally:
            _stop_vllm()
        results.append(loop)
        if loop.pass_target:
            print("  Phase3 target met.", flush=True)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(REPO_ROOT))
    parser.add_argument("--log-root", default="/data/metax-test-logs/tune/phase23")
    parser.add_argument("--model", default="/data/models/Qwen3.6-27B-AWQ")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--phase2-loops", type=int, default=int(os.environ.get("PHASE2_LOOPS", "12")))
    parser.add_argument("--phase3-loops", type=int, default=int(os.environ.get("PHASE3_LOOPS", "5")))
    parser.add_argument("--skip-phase2", action="store_true")
    parser.add_argument("--skip-phase3", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo)
    log_root = Path(args.log_root)
    log_root.mkdir(parents=True, exist_ok=True)

    phase2: List[OpLoop] = []
    phase3: List[Phase3Loop] = []

    if not args.skip_phase2:
        phase2 = run_phase2_loops(repo, log_root, args.phase2_loops)
    if not args.skip_phase3:
        phase3 = run_phase3_loops(repo, log_root, args.model, args.host, args.port, args.phase3_loops)

    best_op = min((r for r in phase2 if r.seq_len == 256 and r.avg_ms), key=lambda r: r.avg_ms, default=None)
    best_p3 = max((r for r in phase3 if r.tok_s), key=lambda r: r.tok_s or 0, default=None)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "targets": TARGETS,
        "best_phase2_s256_ms": best_op.avg_ms if best_op else None,
        "best_phase2_impl": best_op.impl if best_op else None,
        "best_phase3_tok_s": best_p3.tok_s if best_p3 else None,
        "best_phase3_mode": best_p3.mode if best_p3 else None,
        "phase2": [asdict(r) for r in phase2],
        "phase3": [asdict(r) for r in phase3],
    }
    json_path = log_root / "PHASE23_LOOP_RESULTS.json"
    md_path = log_root / "PHASE23_LOOP_RESULTS.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    lines = [
        f"# Phase 2/3 Tuning Loop — {report['timestamp']}",
        "",
        f"- Best fused_rope @ S=256: **{report['best_phase2_s256_ms']}** ms ({report['best_phase2_impl']}) target ≤{TARGETS['fused_rope_rms_ms']}",
        f"- Best Phase3 tok/s: **{report['best_phase3_tok_s']}** ({report['best_phase3_mode']}) target ≥{TARGETS['phase3_mtp_tok_s']}",
        "",
    ]
    md_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {json_path}\nWrote {md_path}")

    p2_ok = best_op and best_op.avg_ms <= TARGETS["fused_rope_rms_ms"]
    p3_ok = best_p3 and (best_p3.tok_s or 0) >= TARGETS["phase3_mtp_tok_s"]
    return 0 if p2_ok and p3_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
