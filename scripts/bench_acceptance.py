#!/usr/bin/env python3
"""Evaluate benchmark results against AGENT.md §12.5 acceptance targets."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# AGENT.md §12.5 targets (C500 32GB)
TARGETS = {
    "phase0_single_tok_s": 9.5,
    "phase1_concurrent_8_tok_s": 40.0,
    "phase2_single_tok_s": 14.0,
    "phase3_mtp_tok_s": 20.0,
    "fused_rope_rms_ms": 0.5,
}


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _find_tok_s_in_text(text: str, label: str) -> Optional[float]:
    # Match bench_qwen36 JSON blocks near section labels
    pattern = rf"{re.escape(label)}[\s\S]*?\"aggregate_tokens_per_s\":\s*([\d.]+)"
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def _parse_summary_md(path: Path) -> Dict[str, Optional[float]]:
    text = path.read_text(errors="replace")
    return {
        "single_baseline": _find_tok_s_in_text(text, "Single request baseline"),
        "concurrent_8": _find_tok_s_in_text(text, "Concurrent x8"),
        "mtp": _find_tok_s_in_text(text, "MTP"),
        "ngram": _find_tok_s_in_text(text, "N-gram"),
    }


def _parse_op_bench(data: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for b in data.get("benchmarks", []):
        k = b.get("kernel", "")
        if "avg_ms" in b:
            out[k] = float(b["avg_ms"])
    return out


def evaluate(log_root: Path) -> Dict[str, Any]:
    report: Dict[str, Any] = {"targets": TARGETS, "metrics": {}, "checks": []}

    summary = log_root / "ALL_BENCH_SUMMARY.md"
    if summary.exists():
        report["metrics"].update(_parse_summary_md(summary))

    parsed = log_root / "PARSED_RESULTS.md"
    if parsed.exists() and not report["metrics"]:
        report["metrics"].update(_parse_summary_md(parsed))

    op_json = log_root / "phase2_op_bench.json"
    if not op_json.exists():
        # try repo bundled result
        repo_op = Path(__file__).resolve().parents[1] / "metax_kernels/bench/results_op_bench_c500.json"
        if repo_op.exists():
            op_json = repo_op
    op_data = _load_json(op_json)
    if op_data:
        report["metrics"]["op_bench"] = _parse_op_bench(op_data)

    metrics = report["metrics"]

    def check(name: str, value: Optional[float], target: float, op: str = "ge") -> None:
        ok = value is not None and ((value >= target) if op == "ge" else (value <= target))
        report["checks"].append(
            {
                "name": name,
                "value": value,
                "target": target,
                "op": op,
                "pass": ok,
                "status": "PASS" if ok else ("SKIP" if value is None else "FAIL"),
            }
        )

    check("phase0_single_tok_s", metrics.get("single_baseline"), TARGETS["phase0_single_tok_s"])
    check("phase1_concurrent_8_tok_s", metrics.get("concurrent_8"), TARGETS["phase1_concurrent_8_tok_s"])
    check("phase3_mtp_tok_s", metrics.get("mtp"), TARGETS["phase3_mtp_tok_s"])

    op = metrics.get("op_bench", {})
    rope_ms = None
    for k, v in op.items():
        if "fused_rope_rms" in k and ("opt_eager" in k or k.endswith(":fused") or k.endswith(":eager")):
            rope_ms = v if rope_ms is None else min(rope_ms, v)
    if isinstance(op, dict):
        for k, v in op.items():
            if k.startswith("qwen36.fused_rope_rms"):
                rope_ms = v if rope_ms is None else min(rope_ms, v)
    check("fused_rope_rms_ms", rope_ms, TARGETS["fused_rope_rms_ms"], op="le")

    passed = sum(1 for c in report["checks"] if c["status"] == "PASS")
    failed = sum(1 for c in report["checks"] if c["status"] == "FAIL")
    skipped = sum(1 for c in report["checks"] if c["status"] == "SKIP")
    report["summary"] = {"pass": passed, "fail": failed, "skip": skipped}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark acceptance vs AGENT.md targets")
    parser.add_argument(
        "log_root",
        nargs="?",
        default="/data/metax-test-logs",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    report = evaluate(Path(args.log_root))
    text = json.dumps(report, indent=2)

    if args.output:
        Path(args.output).write_text(text)
    if args.json:
        print(text)
    else:
        print("Acceptance vs AGENT.md §12.5 (C500 32GB)")
        print("-" * 50)
        for c in report["checks"]:
            val = c["value"] if c["value"] is not None else "N/A"
            sym = ">=" if c["op"] == "ge" else "<="
            print(f"  [{c['status']:4}] {c['name']}: {val} (target {sym} {c['target']})")
        s = report["summary"]
        print("-" * 50)
        print(f"PASS={s['pass']} FAIL={s['fail']} SKIP={s['skip']}")

    return 1 if report["summary"]["fail"] > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
