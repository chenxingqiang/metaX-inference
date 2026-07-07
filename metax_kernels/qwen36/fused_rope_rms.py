"""Fused RoPE + RMSNorm for Qwen3.6 (Phase 2 — AGENT.md §12).

Reference eager implementation runs on MACA PyTorch.
Replace `fused` registration with mcoplib custom op when available.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from metax_kernels.registry import register_kernel

# Qwen3.6-27B typical dims (override at runtime from model config)
DEFAULT_HIDDEN = 5120
DEFAULT_HEADS = 40
DEFAULT_KV_HEADS = 8
DEFAULT_HEAD_DIM = 128


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return x * weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _build_rope_cache(
    seq_len: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    base: float = 1_000_000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()[None, :, None, :]
    sin = emb.sin()[None, :, None, :]
    return cos, sin


@register_kernel("qwen36.fused_rope_rms", impl="eager")
def fused_rope_rmsnorm_eager(
    hidden_states: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    num_heads: int = DEFAULT_HEADS,
    num_kv_heads: int = DEFAULT_KV_HEADS,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Qwen3-style: RMSNorm on Q/K projections, then RoPE."""
    bsz, seq_len, hidden = hidden_states.shape
    head_dim = hidden // num_heads
    kv_dim = num_kv_heads * head_dim

    q = F.linear(hidden_states, q_proj_weight)
    k = F.linear(hidden_states, k_proj_weight)
    v = F.linear(hidden_states, v_proj_weight)

    q = q.view(bsz, seq_len, num_heads, head_dim)
    k = k.view(bsz, seq_len, num_kv_heads, head_dim)

    q = _rms_norm(q, q_norm_weight, eps)
    k = _rms_norm(k, k_norm_weight, eps)

    cos, sin = _build_rope_cache(seq_len, head_dim, hidden_states.device, hidden_states.dtype)
    q, k = _apply_rope(q, k, cos, sin)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    if num_kv_heads != num_heads:
        repeat = num_heads // num_kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)

    return q, k, v


@register_kernel("qwen36.fused_rope_rms", impl="fused")
def fused_rope_rmsnorm_fused(*args, **kwargs):
    """Placeholder for mcoplib fused kernel. Falls back to eager until registered."""
    try:
        import mcoplib  # noqa: F401 — future MACA fused op

        # TODO: mcoplib.qwen36_fused_rope_rms(...)
        pass
    except ImportError:
        pass
    return fused_rope_rmsnorm_eager(*args, **kwargs)


def fused_rope_rmsnorm(hidden_states: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from metax_kernels.registry import KernelRegistry

    fn = KernelRegistry.get("qwen36.fused_rope_rms")
    return fn(hidden_states, **kwargs)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return _rms_norm(x, weight, eps)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _apply_rope(q, k, cos, sin)
