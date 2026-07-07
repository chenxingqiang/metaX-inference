"""Tests for development mcoplib stub."""

from __future__ import annotations

import sys
import unittest

import torch


class TestMcoplibStub(unittest.TestCase):
    def test_stub_wires_and_runs(self) -> None:
        from metax_kernels.dev.mcoplib_stub import install_stub
        from metax_kernels.mcoplib_bridge import bootstrap_mcoplib
        from metax_kernels.registry import KernelRegistry

        install_stub()
        try:
            wired = bootstrap_mcoplib(impl="fused")
            self.assertIn("qwen36.fused_rope_rms", wired)
            fn = KernelRegistry.get("qwen36.fused_rope_rms", impl="fused")
            b, s, h = 1, 8, 5120
            hd = h // 40
            kv = 8 * hd
            x = torch.randn(b, s, h)
            qw = torch.randn(h, h)
            kw = torch.randn(kv, h)
            vw = torch.randn(kv, h)
            qnw = torch.ones(hd)
            knw = torch.ones(hd)
            q, k, v = fn(x, qw, kw, vw, qnw, knw, num_heads=40, num_kv_heads=8)
            self.assertEqual(q.shape, (b, 40, s, hd))
        finally:
            sys.modules.pop("mcoplib", None)


if __name__ == "__main__":
    unittest.main()
