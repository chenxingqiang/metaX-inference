"""Auto-load metax_kernels when vLLM starts (set METAX_KERNELS=1)."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def maybe_bootstrap() -> None:
    """Called at vLLM import time if METAX_KERNELS env is set."""
    if os.environ.get("METAX_KERNELS", "").lower() not in ("1", "true", "yes"):
        return

    impl = os.environ.get("METAX_KERNEL_IMPL", "fused")

    if os.environ.get("METAX_MCOP_STUB", "").lower() in ("1", "true", "yes"):
        from metax_kernels.dev.mcoplib_stub import install_stub

        install_stub()
        logger.info("METAX_MCOP_STUB=1: using opt_eager mcoplib stub")

    from engine.vllm_metax_plugin.register import register_qwen36_kernels
    from metax_kernels.mcoplib_bridge import bootstrap_mcoplib

    mcop = bootstrap_mcoplib(impl=impl)
    registered = register_qwen36_kernels(impl=impl, force=True)
    logger.info(
        "metaX-inference kernels loaded (impl=%s, mcoplib=%s, vllm=%s)",
        impl,
        mcop,
        registered,
    )


# Side-effect import hook — enable with: METAX_KERNELS=1 vllm serve ...
maybe_bootstrap()
