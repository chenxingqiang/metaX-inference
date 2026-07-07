"""Tests for check_mtp_head script."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"
REPO = Path(__file__).resolve().parents[1]


class TestCheckMtpHead(unittest.TestCase):
    def _run(self, model_dir: Path) -> dict:
        proc = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "check_mtp_head.py"), str(model_dir), "--json"],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(proc.stdout)

    def test_no_mtp(self) -> None:
        r = self._run(FIXTURES / "model_no_mtp")
        self.assertFalse(r["has_mtp_head"])
        self.assertEqual(r["mtp_weight_count"], 0)

    def test_with_mtp(self) -> None:
        r = self._run(FIXTURES / "model_with_mtp")
        self.assertTrue(r["has_mtp_head"])
        self.assertGreater(r["mtp_weight_count"], 0)
        self.assertIn("warning", r)


if __name__ == "__main__":
    unittest.main()
