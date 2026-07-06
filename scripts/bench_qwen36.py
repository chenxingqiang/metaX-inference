#!/usr/bin/env python3
"""End-to-end benchmark stub for Qwen3.6 on MACA (AGENT.md §12).

Usage (on MetaX server with vLLM running):
  python scripts/bench_qwen36.py --url http://127.0.0.1:8000 --prompt "你好" --max-tokens 128

Phase 1: measure TTFT and tokens/s from OpenAI-compatible API.
Phase 2+: compare against baseline in TEST_RESULTS.md (~9.5 tok/s).
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Qwen3.6 via vLLM OpenAI API")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--prompt", default="你好，请用一句话介绍你自己。")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    payload = json.dumps(
        {
            "prompt": args.prompt,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
        }
    ).encode()

    req = urllib.request.Request(
        f"{args.url.rstrip('/')}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read().decode())
    elapsed = time.perf_counter() - t0

    text = body["choices"][0].get("text", "")
    usage = body.get("usage", {})
    completion_tokens = usage.get("completion_tokens") or args.max_tokens
    tok_s = completion_tokens / elapsed if elapsed > 0 else 0.0

    print(f"completion_tokens: {completion_tokens}")
    print(f"elapsed_s: {elapsed:.2f}")
    print(f"tokens_per_s: {tok_s:.2f}")
    print(f"sample: {text[:200]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
