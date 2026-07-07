"""Register metax_kernels custom ops with vLLM / vllm_metax (Phase 2 integration)."""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Kernel name -> (registry_key, preferred_impl)
QWEN36_KERNEL_MAP: Dict[str, tuple[str, str]] = {
    "fused_rope_rmsnorm": ("qwen36.fused_rope_rms", "fused"),
    "gqa_attention": ("qwen36.gqa_attention", "fused"),
}

_registered: bool = False


def _try_register_vllm_custom_op(name: str, fn: Callable) -> bool:
    """Attempt vLLM CustomOp registration (v0.17+ uses class decorator at import time)."""
    try:
        from vllm.model_executor.custom_op import CustomOp  # type: ignore

        if not hasattr(CustomOp, "register"):
            return False
        # vLLM expects @CustomOp.register("name") on a subclass at module load.
        # Store fn for a future static patch / PR to vllm_metax model code.
        registry = getattr(CustomOp, "_metax_kernel_fns", None)
        if registry is None:
            CustomOp._metax_kernel_fns = {}  # type: ignore[attr-defined]
            registry = CustomOp._metax_kernel_fns  # type: ignore[attr-defined]
        registry[name] = fn  # type: ignore[index]
        logger.info("staged vLLM CustomOp fn: %s", name)
        return True
    except Exception as exc:
        logger.debug("vLLM CustomOp unavailable for %s: %s", name, exc)
        return False


def _try_register_vllm_metax_hook(name: str, fn: Callable) -> bool:
    """Attempt vllm_metax-specific plugin hook if present."""
    try:
        import vllm_metax  # noqa: F401

        registry = getattr(vllm_metax, "CUSTOM_OP_REGISTRY", None)
        if registry is not None and hasattr(registry, "register"):
            registry.register(name, fn)
            return True
    except Exception as exc:
        logger.debug("vllm_metax CUSTOM_OP_REGISTRY unavailable for %s: %s", name, exc)
    return False


def register_qwen36_kernels(
    impl: str = "fused",
    force: bool = False,
) -> List[str]:
    """Register Qwen3.6 metax_kernels with vLLM / vllm_metax runtimes.

    Returns list of successfully registered op names.
    """
    global _registered
    if _registered and not force:
        return []

    from metax_kernels.registry import KernelRegistry

    KernelRegistry.set_default_impl(impl)
    ok: List[str] = []

    for op_name, (kernel_key, preferred_impl) in QWEN36_KERNEL_MAP.items():
        use_impl = impl or preferred_impl
        try:
            fn = KernelRegistry.get(kernel_key, impl=use_impl)
        except KeyError:
            logger.warning("kernel not found: %s:%s", kernel_key, use_impl)
            continue

        vllm_name = f"metax_{op_name}"
        if _try_register_vllm_custom_op(vllm_name, fn):
            ok.append(vllm_name)
            continue
        if _try_register_vllm_metax_hook(vllm_name, fn):
            ok.append(vllm_name)
            continue

        # Fallback: keep kernel in metax_kernels registry only (MacaFastModel path)
        logger.info("registered locally only: %s -> %s", vllm_name, kernel_key)

    _registered = True
    return ok


def patch_qwen36_attention_layer(model: object, impl: str = "fused") -> int:
    """Monkey-patch Qwen3.5 attention modules to use metax_kernels (notebook / debug).

    Returns number of layers patched. Does not modify vLLM server weights.
    """
    from metax_kernels.qwen36.fused_rope_rms import fused_rope_rmsnorm
    from metax_kernels.qwen36.gqa_attention import gqa_attention

    patched = 0
    for module in model.modules():  # type: ignore[attr-defined]
        cls_name = module.__class__.__name__
        if "Attention" not in cls_name:
            continue
        if hasattr(module, "_metax_kernels_patched"):
            continue

        orig_forward = module.forward

        def _wrapped_forward(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return orig_forward(*args, **kwargs)

        module.forward = _wrapped_forward.__get__(module, module.__class__)
        module._metax_kernels_patched = True
        module._metax_fused_rope_rms = fused_rope_rmsnorm
        module._metax_gqa_attention = gqa_attention
        patched += 1

    if patched:
        from metax_kernels.registry import KernelRegistry

        KernelRegistry.set_default_impl(impl)
    return patched
