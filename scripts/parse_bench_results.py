#!/usr/bin/env python3
"""Parse remote bench logs and emit TEST_RESULTS snippet."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    for match in re.finditer(r"\{[\s\S]*?\}\s*(?=\n|$)", text):
        chunk = match.group(0)
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            continue
    return None


def _parse_bench_qwen36_json(data: Dict[str, Any]) -> str:
    summary = data.get("summary", {})
    lines = [
        f"- aggregate_tokens_per_s: **{summary.get('aggregate_tokens_per_s', 'N/A')}**",
        f"- per_request mean: {summary.get('per_request_tokens_per_s', {}).get('mean', 'N/A')}",
        f"- success: {summary.get('success', '?')}/{summary.get('requests', '?')}",
    ]
    ttft = summary.get("ttft_s")
    if ttft:
        lines.append(f"- ttft_s mean: {ttft.get('mean', 'N/A')}")
    return "\n".join(lines)


def parse_summary(path: Path) -> str:
    text = path.read_text(errors="replace")
    sections = re.split(r"^## ", text, flags=re.MULTILINE)
    out = [f"# Parsed from {path.name}\n"]

    for section in sections[1:]:
        title, _, body = section.partition("\n")
        title = title.strip()
        out.append(f"## {title}\n")

        data = _extract_json_block(body)
        if data and "summary" in data:
            out.append(_parse_bench_qwen36_json(data))
        elif "aggregate_tokens_per_s" in body:
            m = re.search(r'"aggregate_tokens_per_s":\s*([\d.]+)', body)
            if m:
                out.append(f"- aggregate_tokens_per_s: **{m.group(1)}**")
        else:
            snippet = body.strip()[:500]
            if snippet:
                out.append(f"```\n{snippet}\n```")
        out.append("")

    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse ALL_BENCH_SUMMARY.md")
    parser.add_argument(
        "summary",
        nargs="?",
        default="/data/metax-test-logs/ALL_BENCH_SUMMARY.md",
    )
    parser.add_argument("-o", "--output", help="Write markdown to file")
    args = parser.parse_args()

    path = Path(args.summary)
    if not path.exists():
        print(f"File not found: {path}")
        return 1

    md = parse_summary(path)
    if args.output:
        Path(args.output).write_text(md)
        print(f"Wrote {args.output}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
