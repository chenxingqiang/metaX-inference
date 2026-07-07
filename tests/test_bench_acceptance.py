"""Tests for bench_acceptance.py."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "bench_acceptance", REPO / "scripts" / "bench_acceptance.py"
)
assert _spec and _spec.loader
bench_acceptance = importlib.util.module_from_spec(_spec)
sys.modules["bench_acceptance"] = bench_acceptance
_spec.loader.exec_module(bench_acceptance)
evaluate = bench_acceptance.evaluate


class TestBenchAcceptance(unittest.TestCase):
    def test_evaluate_with_op_bench_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            op = {
                "benchmarks": [
                    {"kernel": "qwen36.fused_rope_rms:opt_eager", "avg_ms": 0.45},
                ]
            }
            (root / "phase2_op_bench.json").write_text(json.dumps(op))
            report = evaluate(root)
            rope_check = next(c for c in report["checks"] if c["name"] == "fused_rope_rms_ms")
            self.assertEqual(rope_check["status"], "PASS")
            self.assertEqual(rope_check["value"], 0.45)

    def test_skip_when_no_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = evaluate(Path(tmp))
            self.assertGreater(report["summary"]["skip"], 0)


if __name__ == "__main__":
    unittest.main()
