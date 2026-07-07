#!/usr/bin/env python3
"""Decode-phase operator profiler for Qwen3.6 metax_kernels (AGENT.md §12.6).

Runs PyTorch profiler on fused RoPE + GQA attention to identify hotspots.
Usage on MetaX server:
  PYTHONPATH=. python scripts/profile_decode.py --seq-len 256 --json
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

import torch

from metax_kernels.qwen36.fused_rope_rms import fused_rope_rmsnorm
from metax_kernels.qwen36.gqa_attention import gqa_attention


def _device_time_us(evt: Any) -> float:
    """Return GPU time in microseconds (PyTorch 2.8 uses device_time_total)."""
    for attr in ("cuda_time_total", "device_time_total", "self_cuda_time_total"):
        val = getattr(evt, attr, None)
        if val is not None:
            return float(val)
    return float(getattr(evt, "cpu_time_total", 0.0))


def _top_ops(prof: torch.profiler.profile, limit: int = 15) -> List[Dict[str, Any]]:
    events = prof.key_averages()
    rows: List[Dict[str, Any]] = []
    for evt in sorted(events, key=_device_time_us, reverse=True)[:limit]:
        rows.append(
            {
                "name": evt.key,
                "cuda_time_us": round(_device_time_us(evt), 1),
                "cpu_time_us": round(evt.cpu_time_total, 1),
                "count": evt.count,
            }
        )
    return rows


def profile_decode_step(
    batch: int,
    seq_len: int,
    hidden: int,
    num_heads: int,
    num_kv_heads: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Dict[str, Any]:
    head_dim = hidden // num_heads
    kv_dim = num_kv_heads * head_dim

    x = torch.randn(batch, seq_len, hidden, device=device, dtype=dtype)
    qw = torch.randn(hidden, hidden, device=device, dtype=dtype)
    kw = torch.randn(kv_dim, hidden, device=device, dtype=dtype)
    vw = torch.randn(kv_dim, hidden, device=device, dtype=dtype)
    qnw = torch.ones(head_dim, device=device, dtype=dtype)
    knw = torch.ones(head_dim, device=device, dtype=dtype)

    def decode_step():
        q, k, v = fused_rope_rmsnorm(
            x,
            q_proj_weight=qw,
            k_proj_weight=kw,
            v_proj_weight=vw,
            q_norm_weight=qnw,
            k_norm_weight=knw,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )
        gqa_attention(q, k, v, impl="fused", is_causal=True)

    # Warmup
    for _ in range(3):
        decode_step()
    if device.type == "cuda":
        torch.cuda.synchronize()

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        with_stack=False,
    ) as prof:
        decode_step()
        if device.type == "cuda":
            torch.cuda.synchronize()

    return {
        "shape": f"B={batch},S={seq_len},H={hidden}",
        "top_ops": _top_ops(prof),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile Qwen3.6 decode operators")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--num-heads", type=int, default=40)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "profile": profile_decode_step(
            args.batch, args.seq_len, args.hidden,
            args.num_heads, args.num_kv_heads, dtype, device,
        ),
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Device: {result['gpu']}")
        print(f"Shape: {result['profile']['shape']}")
        print("Top ops (CUDA time):")
        for row in result["profile"]["top_ops"]:
            print(f"  {row['cuda_time_us']:>10.1f} us  {row['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
