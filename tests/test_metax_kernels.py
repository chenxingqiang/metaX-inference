"""Unit tests for metax_kernels Qwen3.6 operators."""

from __future__ import annotations

import unittest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestFusedRopeRms(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.batch, self.seq_len, self.hidden = 1, 16, 5120
        self.num_heads, self.num_kv_heads = 40, 8
        self.head_dim = self.hidden // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim

    def _make_inputs(self):
        x = torch.randn(self.batch, self.seq_len, self.hidden, device=self.device, dtype=self.dtype)
        qw = torch.randn(self.hidden, self.hidden, device=self.device, dtype=self.dtype)
        kw = torch.randn(self.kv_dim, self.hidden, device=self.device, dtype=self.dtype)
        vw = torch.randn(self.kv_dim, self.hidden, device=self.device, dtype=self.dtype)
        qnw = torch.ones(self.head_dim, device=self.device, dtype=self.dtype)
        knw = torch.ones(self.head_dim, device=self.device, dtype=self.dtype)
        return x, qw, kw, vw, qnw, knw

    def test_output_shapes(self) -> None:
        from metax_kernels.qwen36.fused_rope_rms import fused_rope_rmsnorm

        x, qw, kw, vw, qnw, knw = self._make_inputs()
        q, k, v = fused_rope_rmsnorm(
            x,
            impl="eager",
            q_proj_weight=qw,
            k_proj_weight=kw,
            v_proj_weight=vw,
            q_norm_weight=qnw,
            k_norm_weight=knw,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
        )
        self.assertEqual(q.shape, (self.batch, self.num_heads, self.seq_len, self.head_dim))
        self.assertEqual(k.shape, (self.batch, self.num_heads, self.seq_len, self.head_dim))
        self.assertEqual(v.shape, (self.batch, self.num_heads, self.seq_len, self.head_dim))

    def test_rope_cache_reuse(self) -> None:
        from metax_kernels.qwen36 import fused_rope_rms as mod

        mod._ROPE_CACHE.clear()
        x, qw, kw, vw, qnw, knw = self._make_inputs()
        kwargs = dict(
            q_proj_weight=qw,
            k_proj_weight=kw,
            v_proj_weight=vw,
            q_norm_weight=qnw,
            k_norm_weight=knw,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
        )
        mod.fused_rope_rmsnorm(x, impl="eager", **kwargs)
        n_after_first = len(mod._ROPE_CACHE)
        mod.fused_rope_rmsnorm(x, impl="eager", **kwargs)
        self.assertEqual(len(mod._ROPE_CACHE), n_after_first)
        self.assertGreater(n_after_first, 0)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestGqaAttention(unittest.TestCase):
    def test_sdpa_matches_eager_shape(self) -> None:
        from metax_kernels.qwen36.gqa_attention import gqa_attention

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        b, h, s, d = 1, 40, 32, 128
        q = torch.randn(b, h, s, d, device=device, dtype=dtype)
        k = torch.randn(b, h, s, d, device=device, dtype=dtype)
        v = torch.randn(b, h, s, d, device=device, dtype=dtype)

        out_eager = gqa_attention(q, k, v, impl="eager", is_causal=True)
        out_sdpa = gqa_attention(q, k, v, impl="sdpa", is_causal=True)
        self.assertEqual(out_eager.shape, out_sdpa.shape)
        self.assertEqual(out_eager.shape, (b, h, s, d))


if __name__ == "__main__":
    unittest.main()
