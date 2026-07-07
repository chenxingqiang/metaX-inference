#!/usr/bin/env python3
"""Update configs/acceptance_baseline.json from remote bench ACCEPTANCE.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / "configs" / "acceptance_baseline.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Update acceptance baseline from bench logs")
    parser.add_argument(
        "acceptance_json",
        nargs="?",
        default="/data/metax-test-logs/ACCEPTANCE.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    src = Path(args.acceptance_json)
    if not src.exists():
        print(f"Not found: {src}")
        return 1

    report = json.loads(src.read_text())
    checks = {c["name"]: c for c in report.get("checks", [])}
    baseline = json.loads(BASELINE.read_text()) if BASELINE.exists() else {"metrics": {}}
    metrics = baseline.setdefault("metrics", {})

    mapping = {
        "phase0_single_tok_s": "phase0_single_tok_s",
        "phase0_single_tok_s_peak": "phase0_single_tok_s_peak",
        "phase1_concurrent_8_tok_s": "phase1_concurrent_8_tok_s",
        "phase3_mtp_tok_s": "phase3_mtp_tok_s",
    }
    updated = []
    for check_name, metric_key in mapping.items():
        c = checks.get(check_name)
        if c and c.get("value") is not None:
            metrics[metric_key] = c["value"]
            updated.append(metric_key)

    op = report.get("metrics", {}).get("op_bench")
    if isinstance(op, dict) and op:
        baseline["op_bench_snapshot"] = op
        updated.append("op_bench_snapshot")

    baseline["source"] = f"Updated from {src}"

    if args.dry_run:
        print(json.dumps(baseline, indent=2))
    else:
        BASELINE.write_text(json.dumps(baseline, indent=2) + "\n")
        print(f"Updated {BASELINE}: {', '.join(updated) or 'no changes'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
