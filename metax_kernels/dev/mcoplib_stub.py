"""Development mcoplib stub — delegates to opt_eager for integration testing.

Enable on MetaX server before mcoplib Op lands:
  METAX_MCOP_STUB=1 PYTHONPATH=. python -m metax_kernels.bench.op_bench --json

Implementation: `metax_kernels/dev/mcoplib_stub.py` (delegates to opt_eager).
"""

from __future__ import annotations

from typing import Tuple

import torch

from metax_kernels.qwen36.fused_rope_rms import fused_rope_rmsnorm_opt_eager


def qwen36_fused_rope_rms(
    hidden_states: torch.Tensor,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    num_heads: int = 40,
    num_kv_heads: int = 8,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return fused_rope_rmsnorm_opt_eager(
        hidden_states,
        q_proj_weight,
        k_proj_weight,
        v_proj_weight,
        q_norm_weight,
        k_norm_weight,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        eps=eps,
    )


def install_stub() -> None:
    """Register stub as mcoplib module in sys.modules."""
    import sys
    import types

    mod = types.ModuleType("mcoplib")
    mod.qwen36_fused_rope_rms = qwen36_fused_rope_rms
    sys.modules["mcoplib"] = mod
