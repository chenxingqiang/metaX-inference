"""Tests for mcoplib_bridge with injected mock mcoplib."""

from __future__ import annotations

import sys
import types
import unittest

import torch


class TestMcoplibBridge(unittest.TestCase):
    def test_bootstrap_wires_mock_op(self) -> None:
        from metax_kernels.registry import KernelRegistry
        from metax_kernels.mcoplib_bridge import bootstrap_mcoplib, MCOPLIB_KERNEL_MAP

        calls = []

        def mock_rope(*args, **kwargs):
            calls.append("rope")
            return args[0], args[0], args[0]

        mock = types.ModuleType("mcoplib")
        mock.qwen36_fused_rope_rms = mock_rope
        sys.modules["mcoplib"] = mock

        try:
            wired = bootstrap_mcoplib(impl="fused")
            self.assertIn("qwen36.fused_rope_rms", wired)
            fn = KernelRegistry.get("qwen36.fused_rope_rms", impl="fused")
            x = torch.randn(1, 4, 8)
            fn(x, x, x, x, x, x)
            self.assertEqual(calls, ["rope"])
        finally:
            sys.modules.pop("mcoplib", None)
            # restore original fused registration by reimporting
            import importlib
            import metax_kernels.qwen36.fused_rope_rms as fr

            importlib.reload(fr)


if __name__ == "__main__":
    unittest.main()
