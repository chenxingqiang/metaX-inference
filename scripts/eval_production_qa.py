#!/usr/bin/env python3
"""Production-quality Q&A evaluation — show full model responses for manual review."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class QAItem:
    id: str
    category: str
    question: str


@dataclass
class QAResult:
    id: str
    category: str
    question: str
    answer: str
    completion_tokens: int
    elapsed_s: float
    ttft_s: Optional[float]
    tokens_per_s: float
    error: Optional[str] = None


DEFAULT_CASES: List[QAItem] = [
    QAItem("q01", "基础对话", "你好，请用一句话介绍你自己。"),
    QAItem("q02", "中文写作", "请用中文写一段约120字的自我介绍，介绍你的能力和擅长领域，不要换行。"),
    QAItem("q03", "知识问答", "用三句话解释什么是大语言模型，以及它和普通搜索引擎的区别。"),
    QAItem("q04", "逻辑推理", "一个房间里有3盏灯和3个开关，你在门外只能进房间一次，如何确定哪个开关控制哪盏灯？"),
    QAItem("q05", "代码生成", "用 Python 写一个函数，判断字符串是否为回文，并给出两个测试用例。"),
    QAItem("q06", "数学计算", "小明有48个苹果，分给6个同学每人同样多，每人分到几个？请写出计算过程。"),
    QAItem("q07", "指令遵循", "请严格按照格式回答：先写一行「结论：」，再写一行「理由：」，内容关于为什么要做软件测试。"),
    QAItem("q08", "多轮上下文", "我在做 MetaX GPU 上的 Qwen3.6 推理部署。请列出3条生产环境部署建议，每条不超过30字。"),
]


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


def _extract_answer(body: Dict[str, Any], api: str) -> str:
    choices = body.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    if api == "chat":
        msg = choice.get("message") or {}
        return (msg.get("content") or "").strip()
    return (choice.get("text") or "").strip()


def run_qa(
    base_url: str,
    item: QAItem,
    *,
    api: str,
    max_tokens: int,
    temperature: float,
    no_think: bool,
    timeout: float,
) -> QAResult:
    if api == "chat":
        url = f"{base_url.rstrip('/')}/v1/chat/completions"
        payload: Dict[str, Any] = {
            "messages": [{"role": "user", "content": item.question}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if no_think:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        url = f"{base_url.rstrip('/')}/v1/completions"
        payload = {
            "prompt": item.question,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }

    t0 = time.perf_counter()
    try:
        body = _post_json(url, payload, timeout)
        elapsed = time.perf_counter() - t0
        usage = body.get("usage") or {}
        completion_tokens = int(usage.get("completion_tokens") or 0)
        answer = _extract_answer(body, api)
        tok_s = completion_tokens / elapsed if elapsed > 0 else 0.0
        return QAResult(
            id=item.id,
            category=item.category,
            question=item.question,
            answer=answer,
            completion_tokens=completion_tokens,
            elapsed_s=round(elapsed, 3),
            ttft_s=round(elapsed, 3),
            tokens_per_s=round(tok_s, 2),
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return QAResult(
            id=item.id,
            category=item.category,
            question=item.question,
            answer="",
            completion_tokens=0,
            elapsed_s=round(elapsed, 3),
            ttft_s=None,
            tokens_per_s=0.0,
            error=str(exc),
        )


def to_markdown(results: List[QAResult], meta: Dict[str, Any]) -> str:
    lines = [
        "# 生产级推理效果评测",
        "",
        f"- 时间: {meta.get('timestamp')}",
        f"- 模型: {meta.get('model', 'unknown')}",
        f"- API: {meta.get('api')}",
        f"- temperature: {meta.get('temperature')}",
        f"- max_tokens: {meta.get('max_tokens')}",
        f"- no_think: {meta.get('no_think')}",
        f"- 配置说明: {meta.get('config_note', '')}",
        "",
    ]
    for r in results:
        lines.extend(
            [
                f"## [{r.id}] {r.category}",
                "",
                "**问题：**",
                "",
                r.question,
                "",
                "**回答：**",
                "",
                r.answer if r.answer else f"_(错误: {r.error})_",
                "",
                f"_tokens={r.completion_tokens}, elapsed={r.elapsed_s}s, tok/s={r.tokens_per_s}_",
                "",
                "---",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Production Q&A quality evaluation")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--api", choices=["completions", "chat"], default="chat")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--no-think", action="store_true", default=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--warmup", action="store_true", help="Run one discard request first")
    parser.add_argument("--output", help="Write markdown report")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    cases = DEFAULT_CASES
    if args.warmup and cases:
        run_qa(
            args.url,
            cases[0],
            api=args.api,
            max_tokens=32,
            temperature=args.temperature,
            no_think=args.no_think,
            timeout=args.timeout,
        )

    results: List[QAResult] = []
    for item in cases:
        results.append(
            run_qa(
                args.url,
                item,
                api=args.api,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                no_think=args.no_think,
                timeout=args.timeout,
            )
        )

    meta = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "api": args.api,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "no_think": args.no_think,
        "config_note": "生产推荐: chat + no_think + temperature=0, 无 MTP",
    }

    md = to_markdown(results, meta)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(md)

    if args.json:
        print(json.dumps({"meta": meta, "results": [asdict(r) for r in results]}, ensure_ascii=False, indent=2))
    else:
        print(md)
    return 0 if all(r.error is None for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
