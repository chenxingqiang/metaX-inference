"""mcoplib Op interface contract for Qwen3.6 fused kernels.

沐曦 mcoplib 实现以下函数后，调用 bootstrap_mcoplib() 即可自动接入 metax_kernels。

Expected mcoplib Python API (MACA C500 / MACA 3.5.x)
======================================================

qwen36_fused_rope_rms(
    hidden_states,      # [B, S, H]  bfloat16/float16
    q_proj_weight,      # [H, H]
    k_proj_weight,      # [kv_dim, H]
    v_proj_weight,      # [kv_dim, H]
    q_norm_weight,      # [head_dim]
    k_norm_weight,      # [head_dim]
    num_heads=40,
    num_kv_heads=8,
    eps=1e-6,
) -> (q, k, v)
    q: [B, num_heads, S, head_dim]
    k: [B, num_heads, S, head_dim]  (GQA expanded)
    v: [B, num_heads, S, head_dim]

Target latency (C500, S=256, B=1): < 0.5 ms (current eager: ~0.94 ms)

qwen36_gqa_attention(q, k, v, is_causal=True) -> out
    Prefer mcflashattn when available.

qwen36_awq_gemm(x, weight, scale, zero_point) -> out
    W4A16 AWQ matmul — decode bottleneck candidate.

qwen36_fused_mlp(hidden_states, gate_w, up_w, down_w) -> out
    SwiGLU: down(silu(gate(x)) * up(x))

Validation on MetaX server:
    PYTHONPATH=. python -m metax_kernels.bench.op_bench --seq-len 256 --json
"""

from __future__ import annotations

# Re-export for documentation discovery
MCOPLIB_OP_NAMES = (
    "qwen36_fused_rope_rms",
    "qwen36_gqa_attention",
    "qwen36_awq_gemm",
    "qwen36_fused_mlp",
)
