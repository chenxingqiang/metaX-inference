#!/usr/bin/env python3
"""Check whether a Qwen3.6 checkpoint includes usable MTP head weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _scan_safetensors_index(model_dir: Path) -> List[str]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return []
    data = json.loads(index_path.read_text())
    return list(data.get("weight_map", {}).keys())


def check_model(model_dir: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"model_dir": str(model_dir), "exists": model_dir.exists()}
    if not model_dir.exists():
        return result

    config_path = model_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        result["model_type"] = config.get("model_type")
        result["architectures"] = config.get("architectures", [])
        result["num_hidden_layers"] = config.get("num_hidden_layers")
        result["quantization_config"] = config.get("quantization_config")

    weight_names = _scan_safetensors_index(model_dir)
    mtp_keys = [k for k in weight_names if "mtp" in k.lower()]
    result["mtp_weight_keys"] = mtp_keys[:20]
    result["mtp_weight_count"] = len(mtp_keys)
    result["has_mtp_head"] = len(mtp_keys) > 0

    if result.get("quantization_config") and result["has_mtp_head"]:
        result["warning"] = (
            "MTP head present but model is quantized — verify mtp.* tensors remain BF16 "
            "or draft acceptance may be 0%"
        )
    elif not result["has_mtp_head"]:
        result["warning"] = (
            "No mtp.* weights found — use ngram speculative or a MTP-enabled checkpoint"
        )
    else:
        result["warning"] = None

    result["mtp_vllm_ready"] = result["has_mtp_head"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Qwen3.6 MTP head in checkpoint")
    parser.add_argument(
        "model_dir",
        nargs="?",
        default="/data/models/Qwen3.6-27B-AWQ",
        help="Local model directory",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = check_model(Path(args.model_dir))
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Model: {result.get('model_dir')}")
        print(f"  model_type: {result.get('model_type')}")
        print(f"  has_mtp_head: {result.get('has_mtp_head')}")
        print(f"  mtp_weight_count: {result.get('mtp_weight_count')}")
        if result.get("warning"):
            print(f"  WARNING: {result['warning']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
