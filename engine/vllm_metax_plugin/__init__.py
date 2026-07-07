"""vLLM-MetaX plugin hooks (Phase 2+ integration point)."""

from .register import (
    patch_qwen36_attention_layer,
    register_qwen36_kernels,
)

VLLM_METAX_PLUGIN_VERSION = "0.2.0"

__all__ = [
    "VLLM_METAX_PLUGIN_VERSION",
    "register_qwen36_kernels",
    "patch_qwen36_attention_layer",
]
