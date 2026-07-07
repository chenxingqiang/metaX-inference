"""GQA attention for Qwen3.6 — SDPA baseline, flash path when available."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from metax_kernels.registry import register_kernel

DEFAULT_HEADS = 40
DEFAULT_KV_HEADS = 8


def _repeat_kv(hidden: torch.Tensor, num_heads: int) -> torch.Tensor:
    b, n_kv, s, d = hidden.shape
    if n_kv == num_heads:
        return hidden
    repeat = num_heads // n_kv
    return hidden.repeat_interleave(repeat, dim=1)


@register_kernel("qwen36.gqa_attention", impl="eager")
def gqa_attention_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = True,
    scale: Optional[float] = None,
) -> torch.Tensor:
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if is_causal:
        seq = q.shape[-2]
        mask = torch.triu(
            torch.full((seq, seq), float("-inf"), device=q.device, dtype=scores.dtype),
            diagonal=1,
        )
        scores = scores + mask
    if attn_mask is not None:
        scores = scores + attn_mask

    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(probs, v)


@register_kernel("qwen36.gqa_attention", impl="sdpa")
def gqa_attention_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = True,
    scale: Optional[float] = None,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=is_causal, scale=scale
    )


@register_kernel("qwen36.gqa_attention", impl="fused")
def gqa_attention_fused(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    is_causal: bool = True,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Prefer flash-attn on MACA when importable, else SDPA."""
    try:
        import flash_attn  # noqa: F401

        # flash_attn expects (B, S, H, D); q is (B, H, S, D)
        q_f = q.transpose(1, 2).contiguous()
        k_f = k.transpose(1, 2).contiguous()
        v_f = v.transpose(1, 2).contiguous()
        from flash_attn import flash_attn_func

        out = flash_attn_func(q_f, k_f, v_f, causal=is_causal)
        return out.transpose(1, 2)
    except Exception:
        return gqa_attention_sdpa(q, k, v, attn_mask, is_causal, scale)


def gqa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    impl: Optional[str] = None,
    **kwargs,
) -> torch.Tensor:
    from metax_kernels.registry import KernelRegistry

    fn = KernelRegistry.get("qwen36.gqa_attention", impl=impl or "fused")
    return fn(q, k, v, **kwargs)
