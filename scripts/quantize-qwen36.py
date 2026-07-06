#!/usr/bin/env python3
"""Unsloth 4-bit quantize and export for MacaRT-vLLM (AGENT.md §5)."""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Quantize Qwen3.6 with Unsloth for vLLM")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.6-27B",
        help="HuggingFace model id or local path",
    )
    parser.add_argument(
        "--output",
        default="./qwen3.6-27b-4bit",
        help="Output directory for merged safetensors",
    )
    parser.add_argument("--max-seq-length", type=int, default=8192)
    args = parser.parse_args()

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("ERROR: unsloth not installed. pip install unsloth", file=sys.stderr)
        return 1

    print(f"Loading {args.model} (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        device_map="auto",
    )

    print(f"Saving merged model to {args.output} ...")
    model.save_pretrained_merged(args.output, tokenizer)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
