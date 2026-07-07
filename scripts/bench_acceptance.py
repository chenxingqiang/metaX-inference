#!/usr/bin/env python3
"""Evaluate benchmark results against AGENT.md §12.5 acceptance targets."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]

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


def _load_baseline() -> Dict[str, Any]:
    path = REPO_ROOT / "configs" / "acceptance_baseline.json"
    return _load_json(path) or {}


def _find_tok_s_in_text(text: str, label: str) -> Optional[float]:
    pattern = rf"{re.escape(label)}[\s\S]*?\"aggregate_tokens_per_s\":\s*([\d.]+)"
    m = re.search(pattern, text, re.IGNORECASE)
    return float(m.group(1)) if m else None


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


def _best_rope_ms(op: Dict[str, float]) -> Optional[float]:
    rope_ms = None
    for k, v in op.items():
        if "fused_rope_rms" in k:
            rope_ms = v if rope_ms is None else min(rope_ms, v)
    return rope_ms


def evaluate(log_root: Path, use_baseline: bool = True) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "targets": TARGETS,
        "metrics": {},
        "checks": [],
        "sources": [],
    }

    baseline = _load_baseline() if use_baseline else {}
    if baseline:
        report["sources"].append("configs/acceptance_baseline.json")
        bm = baseline.get("metrics", {})
        if bm.get("phase0_single_tok_s") is not None:
            report["metrics"]["single_baseline"] = float(bm["phase0_single_tok_s"])
        if bm.get("phase0_single_tok_s_peak") is not None:
            report["metrics"]["single_peak"] = float(bm["phase0_single_tok_s_peak"])
        for key, live_key in [
            ("phase1_concurrent_8_tok_s", "concurrent_8"),
            ("phase3_mtp_tok_s", "mtp"),
        ]:
            if bm.get(key) is not None:
                report["metrics"][live_key] = float(bm[key])

    if log_root.is_dir():
        summary = log_root / "ALL_BENCH_SUMMARY.md"
        if summary.exists():
            report["sources"].append(str(summary))
            report["metrics"].update({k: v for k, v in _parse_summary_md(summary).items() if v is not None})

        parsed = log_root / "PARSED_RESULTS.md"
        if parsed.exists():
            report["sources"].append(str(parsed))
            for k, v in _parse_summary_md(parsed).items():
                if v is not None:
                    report["metrics"][k] = v

    op_json = log_root / "phase2_op_bench.json" if log_root.is_dir() else None
    if not op_json or not op_json.exists():
        rel = baseline.get("op_bench_path", "metax_kernels/bench/results_op_bench_c500.json")
        op_json = REPO_ROOT / rel
    op_data = _load_json(op_json) if op_json else None
    if op_data:
        report["sources"].append(str(op_json))
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

    # Phase 0: use live if available, else baseline; also check peak
    p0 = metrics.get("single_baseline")
    check("phase0_single_tok_s", p0, TARGETS["phase0_single_tok_s"])
    peak = metrics.get("single_peak")
    if peak is not None and peak != p0:
        check("phase0_single_tok_s_peak", peak, TARGETS["phase0_single_tok_s"])

    check("phase1_concurrent_8_tok_s", metrics.get("concurrent_8"), TARGETS["phase1_concurrent_8_tok_s"])
    check("phase3_mtp_tok_s", metrics.get("mtp"), TARGETS["phase3_mtp_tok_s"])

    op = metrics.get("op_bench", {})
    rope_ms = _best_rope_ms(op) if isinstance(op, dict) else None
    check("fused_rope_rms_ms", rope_ms, TARGETS["fused_rope_rms_ms"], op="le")

    passed = sum(1 for c in report["checks"] if c["status"] == "PASS")
    failed = sum(1 for c in report["checks"] if c["status"] == "FAIL")
    skipped = sum(1 for c in report["checks"] if c["status"] == "SKIP")
    report["summary"] = {"pass": passed, "fail": failed, "skip": skipped}
    return report


def to_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "## 自动验收报告（bench_acceptance.py）",
        "",
        "| 指标 | 实测 | 目标 | 状态 |",
        "|------|------|------|------|",
    ]
    for c in report["checks"]:
        val = c["value"] if c["value"] is not None else "—"
        sym = "≥" if c["op"] == "ge" else "≤"
        lines.append(f"| {c['name']} | {val} | {sym} {c['target']} | {c['status']} |")
    s = report["summary"]
    lines.extend(
        [
            "",
            f"**PASS={s['pass']} FAIL={s['fail']} SKIP={s['skip']}**",
            "",
            f"数据来源: {', '.join(report.get('sources', []))}",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark acceptance vs AGENT.md targets")
    parser.add_argument("log_root", nargs="?", default="/data/metax-test-logs")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("-o", "--output")
    args = parser.parse_args()

    report = evaluate(Path(args.log_root), use_baseline=not args.no_baseline)

    if args.markdown:
        text = to_markdown(report)
    else:
        text = json.dumps(report, indent=2)

    if args.output:
        Path(args.output).write_text(text)

    if args.json:
        print(json.dumps(report, indent=2))
    elif args.markdown and not args.output:
        print(text)
    elif not args.output:
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
