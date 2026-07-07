"""Fused SwiGLU MLP for Qwen3.6 (Phase 2 — AGENT.md §12)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from metax_kernels.registry import register_kernel


@register_kernel("qwen36.fused_mlp", impl="eager")
def fused_mlp_eager(
    hidden_states: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
) -> torch.Tensor:
    """Qwen3 dense MLP: down(SiLU(gate(x)) * up(x))."""
    gate = F.linear(hidden_states, gate_proj_weight)
    up = F.linear(hidden_states, up_proj_weight)
    x = F.silu(gate) * up
    return F.linear(x, down_proj_weight)


@register_kernel("qwen36.fused_mlp", impl="fused")
def fused_mlp_fused(*args, **kwargs):
    try:
        import mcoplib  # noqa: F401
        # TODO: mcoplib.qwen36_fused_mlp(...)
    except ImportError:
        pass
    return fused_mlp_eager(*args, **kwargs)


def fused_mlp(
    hidden_states: torch.Tensor,
    gate_proj_weight: torch.Tensor,
    up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    impl: Optional[str] = None,
) -> torch.Tensor:
    from metax_kernels.registry import KernelRegistry

    fn = KernelRegistry.get("qwen36.fused_mlp", impl=impl)
    return fn(hidden_states, gate_proj_weight, up_proj_weight, down_proj_weight)
