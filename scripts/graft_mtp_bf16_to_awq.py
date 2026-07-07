#!/usr/bin/env python3
"""Graft BF16 mtp.* weights from a source checkpoint onto a MetaX-compatible AWQ base.

hampsonw/Qwen3.6-27B-AWQ-BF16-INT4-mtp-bf16 uses compressed-tensors WNA16 (uint4) which
MetaX Exllama cannot load. This script copies the AWQ base layout and replaces only the
15 mtp.* tensors from a BF16 MTP source (hampsonw or Qwen/Qwen3.6-27B).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from safetensors import safe_open
from safetensors.torch import save_file


def _load_index(model_dir: Path) -> Dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing index: {index_path}")
    data = json.loads(index_path.read_text())
    return dict(data.get("weight_map", {}))


def _mtp_keys(weight_map: Dict[str, str]) -> List[str]:
    return sorted(k for k in weight_map if k.startswith("mtp."))


def _collect_mtp_tensors(source: Path, keys: List[str], weight_map: Dict[str, str]) -> Dict[str, object]:
    by_file: Dict[str, List[str]] = {}
    for key in keys:
        by_file.setdefault(weight_map[key], []).append(key)

    tensors: Dict[str, object] = {}
    for filename, file_keys in by_file.items():
        shard = source / filename
        with safe_open(str(shard), framework="pt") as st:
            for key in file_keys:
                tensors[key] = st.get_tensor(key)
    return tensors


def _hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def graft(base: Path, source: Path, out: Path, force: bool = False) -> Dict[str, object]:
    if out.exists() and any(out.iterdir()) and not force:
        raise FileExistsError(f"Output dir not empty: {out} (use --force)")

    base_map = _load_index(base)
    src_map = _load_index(source)
    mtp_keys = _mtp_keys(src_map)
    if not mtp_keys:
        raise ValueError(f"No mtp.* keys in source: {source}")

    missing = [k for k in mtp_keys if k not in base_map]
    if missing:
        raise ValueError(f"Base missing mtp keys: {missing[:5]}")

    out.mkdir(parents=True, exist_ok=True)

    # Copy non-weight files from AWQ base (tokenizer, config, etc.)
    skip = {p.name for p in base.glob("model-*.safetensors")} | {
        "model.safetensors.index.json",
    }
    for item in base.iterdir():
        if item.name in skip:
            continue
        dst = out / item.name
        if item.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)

    mtp_tensors = _collect_mtp_tensors(source, mtp_keys, src_map)

    # Rewrite only the base shard(s) that contain mtp.* keys
    touched_shards = sorted({base_map[k] for k in mtp_keys})
    for shard_name in touched_shards:
        src_shard = base / shard_name
        dst_shard = out / shard_name
        shard_tensors: Dict[str, object] = {}
        with safe_open(str(src_shard), framework="pt") as st:
            for key in st.keys():
                shard_tensors[key] = st.get_tensor(key)
        for key in mtp_keys:
            if base_map[key] == shard_name:
                shard_tensors[key] = mtp_tensors[key]
        save_file(shard_tensors, str(dst_shard))

    # Hardlink untouched weight shards
    for shard in base.glob("model-*.safetensors"):
        if shard.name in touched_shards:
            continue
        _hardlink_or_copy(shard, out / shard.name)

    shutil.copy2(base / "model.safetensors.index.json", out / "model.safetensors.index.json")

    # Report dtype / diff summary
    report: Dict[str, object] = {
        "base": str(base),
        "source": str(source),
        "out": str(out),
        "mtp_keys": len(mtp_keys),
        "touched_shards": touched_shards,
    }
    dtypes = {str(mtp_tensors[k].dtype) for k in mtp_keys}
    report["mtp_dtypes"] = sorted(dtypes)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Graft BF16 MTP head onto AWQ base")
    parser.add_argument(
        "--base",
        default="/data/models/Qwen3.6-27B-AWQ",
        help="MetaX-compatible AWQ checkpoint",
    )
    parser.add_argument(
        "--source",
        default="/data/models/Qwen3.6-27B-MTP-BF16",
        help="BF16 MTP source (hampsonw or full Qwen3.6-27B)",
    )
    parser.add_argument(
        "--out",
        default="/data/models/Qwen3.6-27B-AWQ-MTP-BF16",
        help="Output grafted checkpoint",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output directory")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = graft(Path(args.base), Path(args.source), Path(args.out), force=args.force)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Grafted {report['mtp_keys']} mtp.* tensors -> {report['out']}")
        print(f"  dtypes: {report['mtp_dtypes']}")
        print(f"  touched shards: {report['touched_shards']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
