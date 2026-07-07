#!/usr/bin/env python3
"""End-to-end benchmark for Qwen3.6 on MACA (AGENT.md §12).

Usage (on MetaX server with vLLM running):
  python scripts/bench_qwen36.py --url http://127.0.0.1:8000
  python scripts/bench_qwen36.py --concurrency 8 --json

Phase 1: measure TTFT and tokens/s from OpenAI-compatible API.
Phase 2+: compare against baseline in TEST_RESULTS.md (~9.5 tok/s single).
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RequestResult:
    request_id: int
    completion_tokens: int
    elapsed_s: float
    ttft_s: Optional[float]
    tokens_per_s: float
    error: Optional[str] = None


def _post_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _stream_completion(
    url: str,
    payload: Dict[str, Any],
    timeout: float,
) -> tuple[int, float, Optional[float]]:
    """Return (completion_tokens, elapsed_s, ttft_s) using SSE streaming."""
    payload = {**payload, "stream": True}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    ttft: Optional[float] = None
    completion_tokens = 0

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                event = json.loads(chunk)
            except json.JSONDecodeError:
                continue

            if ttft is None:
                ttft = time.perf_counter() - t0

            choices = event.get("choices") or []
            if not choices:
                continue
            text = choices[0].get("text") or ""
            if text:
                completion_tokens += max(1, len(text.split()))

            usage = event.get("usage") or {}
            if usage.get("completion_tokens"):
                completion_tokens = int(usage["completion_tokens"])

    elapsed = time.perf_counter() - t0
    return completion_tokens, elapsed, ttft


def run_single_request(
    base_url: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    request_id: int,
    timeout: float,
    stream: bool,
) -> RequestResult:
    url = f"{base_url.rstrip('/')}/v1/completions"
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        if stream:
            completion_tokens, elapsed, ttft = _stream_completion(url, payload, timeout)
        else:
            t0 = time.perf_counter()
            body = _post_json(url, payload, timeout)
            elapsed = time.perf_counter() - t0
            usage = body.get("usage", {})
            completion_tokens = int(usage.get("completion_tokens") or max_tokens)
            ttft = None

        tok_s = completion_tokens / elapsed if elapsed > 0 else 0.0
        return RequestResult(
            request_id=request_id,
            completion_tokens=completion_tokens,
            elapsed_s=elapsed,
            ttft_s=ttft,
            tokens_per_s=tok_s,
        )
    except Exception as exc:
        return RequestResult(
            request_id=request_id,
            completion_tokens=0,
            elapsed_s=0.0,
            ttft_s=None,
            tokens_per_s=0.0,
            error=str(exc),
        )


def summarize(results: List[RequestResult]) -> Dict[str, Any]:
    ok = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]

    if not ok:
        return {
            "requests": len(results),
            "success": 0,
            "failed": len(failed),
            "errors": [r.error for r in failed],
        }

    per_req_tok_s = [r.tokens_per_s for r in ok]
    total_tokens = sum(r.completion_tokens for r in ok)
    wall_s = max(r.elapsed_s for r in ok) if ok else 0.0
    aggregate_tok_s = total_tokens / wall_s if wall_s > 0 else 0.0

    ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
    summary: Dict[str, Any] = {
        "requests": len(results),
        "success": len(ok),
        "failed": len(failed),
        "total_completion_tokens": total_tokens,
        "wall_elapsed_s": round(wall_s, 3),
        "aggregate_tokens_per_s": round(aggregate_tok_s, 2),
        "per_request_tokens_per_s": {
            "mean": round(statistics.mean(per_req_tok_s), 2),
            "min": round(min(per_req_tok_s), 2),
            "max": round(max(per_req_tok_s), 2),
        },
        "per_request_elapsed_s": {
            "mean": round(statistics.mean([r.elapsed_s for r in ok]), 2),
            "p50": round(statistics.median([r.elapsed_s for r in ok]), 2),
        },
    }
    if ttfts:
        summary["ttft_s"] = {
            "mean": round(statistics.mean(ttfts), 3),
            "p50": round(statistics.median(ttfts), 3),
        }
    if failed:
        summary["errors"] = [r.error for r in failed]
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Qwen3.6 via vLLM OpenAI API")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--prompt", default="你好，请用一句话介绍你自己。")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--concurrency", type=int, default=1, help="Parallel requests")
    parser.add_argument("--requests", type=int, default=None, help="Total requests (default=concurrency)")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--stream", action="store_true", help="Use SSE streaming for TTFT")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    total = args.requests or args.concurrency
    concurrency = min(args.concurrency, total)

    results: List[RequestResult] = []
    t_wall0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                run_single_request,
                args.url,
                args.prompt,
                args.max_tokens,
                args.temperature,
                i,
                args.timeout,
                args.stream,
            )
            for i in range(total)
        ]
        for fut in as_completed(futures):
            results.append(fut.result())

    wall_total = time.perf_counter() - t_wall0
    results.sort(key=lambda r: r.request_id)
    summary = summarize(results)
    summary["concurrency"] = concurrency
    summary["total_wall_s"] = round(wall_total, 3)

    if args.json:
        print(json.dumps({"summary": summary, "results": [asdict(r) for r in results]}, indent=2))
    else:
        print(f"concurrency: {concurrency}, requests: {total}")
        print(f"success: {summary.get('success', 0)}/{summary.get('requests', total)}")
        if "aggregate_tokens_per_s" in summary:
            print(f"aggregate_tokens_per_s: {summary['aggregate_tokens_per_s']}")
            print(f"per_request_tokens_per_s (mean): {summary['per_request_tokens_per_s']['mean']}")
        if "ttft_s" in summary:
            print(f"ttft_s (mean): {summary['ttft_s']['mean']}")
        if summary.get("errors"):
            print(f"errors: {summary['errors'][:3]}")
    return 0 if summary.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
