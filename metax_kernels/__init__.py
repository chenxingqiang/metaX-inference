"""MACA inference kernel plugins for metaX-inference (AGENT.md §12)."""

from metax_kernels.registry import KernelRegistry, register_kernel
from metax_kernels.mcoplib_bridge import bootstrap_mcoplib, mcoplib_available

__all__ = ["KernelRegistry", "register_kernel", "bootstrap_mcoplib", "mcoplib_available"]
