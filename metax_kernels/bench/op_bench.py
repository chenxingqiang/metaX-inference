#!/usr/bin/env python3
"""Operator-level benchmark for Qwen3.6 MACA kernels (AGENT.md §12 Phase 2)."""

from __future__ import annotations

import argparse
import json
import time
from typing import Callable, Dict, List

import torch

from metax_kernels.qwen36.fused_rope_rms import fused_rope_rmsnorm
from metax_kernels.qwen36.gqa_attention import gqa_attention
from metax_kernels.qwen36.awq_gemm import awq_gemm
from metax_kernels.qwen36.fused_mlp import fused_mlp
from metax_kernels.registry import KernelRegistry


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _bench_fn(fn: Callable, warmup: int, iters: int) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for _ in range(warmup):
        fn()
    _sync(device)

    times: List[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t0)

    avg = sum(times) / len(times)
    return {
        "avg_ms": avg * 1000,
        "min_ms": min(times) * 1000,
        "max_ms": max(times) * 1000,
        "iters": iters,
    }


def bench_fused_rope_rms(
    batch: int,
    seq_len: int,
    hidden: int,
    num_heads: int,
    num_kv_heads: int,
    dtype: torch.dtype,
    device: torch.device,
    impl: str,
    warmup: int,
    iters: int,
) -> Dict[str, float]:
    head_dim = hidden // num_heads
    kv_dim = num_kv_heads * head_dim
    x = torch.randn(batch, seq_len, hidden, device=device, dtype=dtype)
    qw = torch.randn(hidden, hidden, device=device, dtype=dtype)
    kw = torch.randn(kv_dim, hidden, device=device, dtype=dtype)
    vw = torch.randn(kv_dim, hidden, device=device, dtype=dtype)
    qnw = torch.ones(head_dim, device=device, dtype=dtype)
    knw = torch.ones(head_dim, device=device, dtype=dtype)

    def run():
        fused_rope_rmsnorm(
            x,
            impl=impl,
            q_proj_weight=qw,
            k_proj_weight=kw,
            v_proj_weight=vw,
            q_norm_weight=qnw,
            k_norm_weight=knw,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )

    stats = _bench_fn(run, warmup, iters)
    stats["kernel"] = f"qwen36.fused_rope_rms:{impl}"
    stats["shape"] = f"B={batch},S={seq_len},H={hidden}"
    return stats


def bench_gqa_attention(
    batch: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    impl: str,
    warmup: int,
    iters: int,
) -> Dict[str, float]:
    q = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)

    def run():
        gqa_attention(q, k, v, impl=impl, is_causal=True)

    stats = _bench_fn(run, warmup, iters)
    stats["kernel"] = f"qwen36.gqa_attention:{impl}"
    stats["shape"] = f"B={batch},H={num_heads},S={seq_len},D={head_dim}"
    return stats


def bench_awq_gemm(
    batch: int,
    seq_len: int,
    hidden: int,
    out_features: int,
    dtype: torch.dtype,
    device: torch.device,
    impl: str,
    warmup: int,
    iters: int,
) -> Dict[str, float]:
    x = torch.randn(batch, seq_len, hidden, device=device, dtype=dtype)
    w = torch.randn(out_features, hidden, device=device, dtype=dtype)

    def run():
        awq_gemm(x, w, impl=impl)

    stats = _bench_fn(run, warmup, iters)
    stats["kernel"] = f"qwen36.awq_gemm:{impl}"
    stats["shape"] = f"B={batch},S={seq_len},H={hidden},O={out_features}"
    return stats


def bench_fused_mlp(
    batch: int,
    seq_len: int,
    hidden: int,
    intermediate: int,
    dtype: torch.dtype,
    device: torch.device,
    impl: str,
    warmup: int,
    iters: int,
) -> Dict[str, float]:
    x = torch.randn(batch, seq_len, hidden, device=device, dtype=dtype)
    gw = torch.randn(intermediate, hidden, device=device, dtype=dtype)
    uw = torch.randn(intermediate, hidden, device=device, dtype=dtype)
    dw = torch.randn(hidden, intermediate, device=device, dtype=dtype)

    def run():
        fused_mlp(x, gw, uw, dw, impl=impl)

    stats = _bench_fn(run, warmup, iters)
    stats["kernel"] = f"qwen36.fused_mlp:{impl}"
    stats["shape"] = f"B={batch},S={seq_len},H={hidden},I={intermediate}"
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="MACA Qwen3.6 operator micro-benchmark")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--intermediate", type=int, default=17408)
    parser.add_argument("--num-heads", type=int, default=40)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "registered_kernels": KernelRegistry.list_kernels(),
        "benchmarks": [],
    }

    for impl in ("eager", "compiled", "fused"):
        try:
            results["benchmarks"].append(
                bench_fused_rope_rms(
                    args.batch, args.seq_len, args.hidden,
                    args.num_heads, args.num_kv_heads,
                    dtype, device, impl, args.warmup, args.iters,
                )
            )
        except Exception as exc:
            results["benchmarks"].append({"kernel": f"fused_rope_rms:{impl}", "error": str(exc)})

    head_dim = args.hidden // args.num_heads
    for impl in ("eager", "sdpa", "fused"):
        try:
            results["benchmarks"].append(
                bench_gqa_attention(
                    args.batch, args.seq_len, args.num_heads, head_dim,
                    dtype, device, impl, args.warmup, args.iters,
                )
            )
        except Exception as exc:
            results["benchmarks"].append({"kernel": f"gqa:{impl}", "error": str(exc)})

    for impl in ("eager", "fused"):
        try:
            results["benchmarks"].append(
                bench_awq_gemm(
                    args.batch, args.seq_len, args.hidden, args.hidden,
                    dtype, device, impl, args.warmup, args.iters,
                )
            )
        except Exception as exc:
            results["benchmarks"].append({"kernel": f"awq_gemm:{impl}", "error": str(exc)})

    for impl in ("eager", "fused"):
        try:
            results["benchmarks"].append(
                bench_fused_mlp(
                    args.batch, args.seq_len, args.hidden, args.intermediate,
                    dtype, device, impl, args.warmup, args.iters,
                )
            )
        except Exception as exc:
            results["benchmarks"].append({"kernel": f"fused_mlp:{impl}", "error": str(exc)})

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"Device: {results['gpu']}")
        for b in results["benchmarks"]:
            if "error" in b:
                print(f"  {b['kernel']}: ERROR {b['error']}")
            else:
                print(f"  {b['kernel']} [{b['shape']}]: {b['avg_ms']:.3f} ms avg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
