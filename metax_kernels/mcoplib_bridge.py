"""Runtime bridge to mcoplib custom ops on MetaX MACA."""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# KernelRegistry key -> mcoplib attribute name
MCOPLIB_KERNEL_MAP: Dict[str, str] = {
    "qwen36.fused_rope_rms": "qwen36_fused_rope_rms",
    "qwen36.gqa_attention": "qwen36_gqa_attention",
    "qwen36.awq_gemm": "qwen36_awq_gemm",
    "qwen36.fused_mlp": "qwen36_fused_mlp",
}


def _get_mcoplib_fn(name: str) -> Optional[Callable]:
    try:
        import mcoplib
    except ImportError:
        return None
    fn = getattr(mcoplib, name, None)
    return fn if callable(fn) else None


def bootstrap_mcoplib(impl: str = "fused") -> List[str]:
    """Wire mcoplib functions into KernelRegistry when available.

    Returns list of kernel keys successfully overridden.
    """
    from metax_kernels.registry import KernelRegistry

    wired: List[str] = []
    for kernel_key, mcop_attr in MCOPLIB_KERNEL_MAP.items():
        fn = _get_mcoplib_fn(mcop_attr)
        if fn is None:
            continue
        KernelRegistry.override(kernel_key, impl, fn)
        wired.append(kernel_key)
        logger.info("mcoplib wired: %s -> %s", kernel_key, mcop_attr)

    if wired:
        KernelRegistry.set_default_impl(impl)
    return wired


def mcoplib_available() -> bool:
    try:
        import mcoplib  # noqa: F401

        return True
    except ImportError:
        return False


def list_mcoplib_ops() -> List[str]:
    """Return mcoplib attributes that match our kernel map."""
    try:
        import mcoplib
    except ImportError:
        return []
    return [name for name in MCOPLIB_KERNEL_MAP.values() if callable(getattr(mcoplib, name, None))]
