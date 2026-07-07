#!/usr/bin/env python3
"""Lightweight MACA inference wrapper (Phase 4 — AGENT.md §12)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Union

import torch

from metax_kernels.mcoplib_bridge import bootstrap_mcoplib, list_mcoplib_ops
from metax_kernels.registry import KernelRegistry


@dataclass
class MacaFastModel:
    """Notebook-friendly single-request inference on MACA PyTorch.

    Production throughput should still use `vllm serve`.
    """

    model: Any
    tokenizer: Any
    device: torch.device
    maca_kernels_enabled: bool = False
    wired_kernels: List[str] = field(default_factory=list)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        max_seq_length: int = 8192,
        torch_dtype: Optional[torch.dtype] = torch.bfloat16,
        device_map: str = "auto",
        trust_remote_code: bool = True,
    ) -> "MacaFastModel":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            max_position_embeddings=max_seq_length,
        )
        device = next(model.parameters()).device
        return cls(model=model, tokenizer=tok, device=device)

    def enable_maca_inference(self, kernel_impl: str = "fused", patch_layers: bool = True) -> List[str]:
        """Enable fused MACA kernels and optionally patch attention modules."""
        KernelRegistry.set_default_impl(kernel_impl)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        wired = bootstrap_mcoplib(impl=kernel_impl)
        self.wired_kernels = wired

        if patch_layers:
            from engine.vllm_metax_plugin.register import patch_qwen36_attention_layer

            patch_qwen36_attention_layer(self.model, impl=kernel_impl)

        self.maca_kernels_enabled = True
        return wired

    def kernel_status(self) -> dict:
        return {
            "maca_kernels_enabled": self.maca_kernels_enabled,
            "default_impl": KernelRegistry._default_impl,
            "wired_mcoplib": self.wired_kernels,
            "available_mcoplib_ops": list_mcoplib_ops(),
            "registered": KernelRegistry.list_kernels(),
        }

    @torch.inference_mode()
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Union[str, List[str]]:
        single = isinstance(prompt, str)
        prompts = [prompt] if single else prompt
        results = []
        for p in prompts:
            inputs = self.tokenizer(p, return_tensors="pt").to(self.device)
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                **kwargs,
            )
            text = self.tokenizer.decode(out[0], skip_special_tokens=True)
            results.append(text)
        return results[0] if single else results
