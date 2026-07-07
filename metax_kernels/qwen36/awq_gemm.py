"""AWQ W4A16 GEMM thin wrapper for Qwen3.6 benchmark (Phase 2 — AGENT.md §12).

Production path uses vllm_metax MacaAWQMarlinConfig internally.
This module exposes a micro-benchmark hook for op_bench comparisons.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from metax_kernels.registry import register_kernel


@register_kernel("qwen36.awq_gemm", impl="eager")
def awq_gemm_eager(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    zero_point: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference dense linear — baseline when AWQ dequant not available."""
    return F.linear(x, weight)


@register_kernel("qwen36.awq_gemm", impl="fused")
def awq_gemm_fused(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    zero_point: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Prefer vllm_metax / mcoplib AWQ kernel when importable."""
    try:
        from vllm.model_executor.layers.quantization.awq import AWQLinearMethod  # type: ignore

        if scale is not None and zero_point is not None:
            # Dequant stub for micro-bench — real path is inside vLLM layers
            w = weight * scale.unsqueeze(-1)
            return F.linear(x, w)
    except Exception:
        pass

    try:
        import mcoplib  # noqa: F401
        # TODO: mcoplib.awq_gemm(...)
    except ImportError:
        pass

    return awq_gemm_eager(x, weight, scale, zero_point)


def awq_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    impl: Optional[str] = None,
    **kwargs,
) -> torch.Tensor:
    from metax_kernels.registry import KernelRegistry

    fn = KernelRegistry.get("qwen36.awq_gemm", impl=impl)
    return fn(x, weight, **kwargs)
