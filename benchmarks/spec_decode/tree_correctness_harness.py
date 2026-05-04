# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""TREE_ATTN / tree speculative decoding correctness harness.

This script talks to already-running OpenAI-compatible vLLM servers.  It sends
the same fixed prompt JSONL to each case, records token IDs and text, compares
outputs token-by-token, and optionally merges tree verifier traces written via
``VLLM_TREE_SPEC_TRACE_PATH``.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompts",
        default="benchmarks/spec_decode/tree_correctness_prompts.jsonl",
        help="JSONL file with {'id', 'prompt'} records.",
    )
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        help=(
            "Case in NAME=BASE_URL form. Example: "
            "target=http://127.0.0.1:8011"
        ),
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--trace", action="append", default=[])
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_prompts(path: str) -> list[dict[str, str]]:
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if "id" not in item or "prompt" not in item:
                raise ValueError(f"{path}:{line_no} must contain id and prompt")
            prompts.append({"id": str(item["id"]), "prompt": str(item["prompt"])})
    return prompts


def parse_cases(values: Iterable[str]) -> dict[str, str]:
    cases: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"case must be NAME=BASE_URL, got {value!r}")
        name, base_url = value.split("=", 1)
        cases[name] = base_url.rstrip("/")
    return cases


def parse_traces(values: Iterable[str]) -> dict[str, str]:
    traces: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"trace must be NAME=PATH, got {value!r}")
        name, trace_path = value.split("=", 1)
        traces[name] = trace_path
    return traces


def request_completion(
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "ignore_eos": True,
        "return_token_ids": True,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.load(resp)
    elapsed_ms = (time.perf_counter() - start) * 1000
    choice = body["choices"][0]
    token_ids = choice.get("token_ids")
    if token_ids is None:
        raise ValueError(
            "server did not return token_ids; ensure the vLLM OpenAI "
            "completion endpoint supports return_token_ids"
        )
    return {
        "text": choice["text"],
        "token_ids": token_ids,
        "usage": body.get("usage"),
        "latency_ms": elapsed_ms,
    }


def first_token_diff(lhs: list[int], rhs: list[int]) -> int | None:
    limit = min(len(lhs), len(rhs))
    for idx in range(limit):
        if lhs[idx] != rhs[idx]:
            return idx
    if len(lhs) != len(rhs):
        return limit
    return None


def load_trace_records(path: str) -> list[dict[str, Any]]:
    trace_path = Path(path)
    if not trace_path.exists():
        return []
    records = []
    with open(trace_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.prompts)
    cases = parse_cases(args.case)
    trace_paths = parse_traces(args.trace)
    if not cases:
        raise ValueError("at least one case is required")

    results: dict[str, Any] = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "prompts": prompts,
        "cases": {},
        "comparisons": {},
    }

    for case_name, base_url in cases.items():
        case_outputs = []
        for prompt in prompts:
            output = request_completion(
                base_url=base_url,
                model=args.model,
                prompt=prompt["prompt"],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout=args.timeout,
            )
            case_outputs.append({"id": prompt["id"], **output})
        results["cases"][case_name] = {
            "base_url": base_url,
            "outputs": case_outputs,
            "trace": load_trace_records(trace_paths[case_name])
            if case_name in trace_paths
            else [],
        }

    baseline_name = next(iter(cases))
    baseline_outputs = results["cases"][baseline_name]["outputs"]
    for case_name, case in results["cases"].items():
        if case_name == baseline_name:
            continue
        diffs = []
        for base_output, case_output in zip(baseline_outputs, case["outputs"]):
            base_token_ids = base_output.get("token_ids") or []
            case_token_ids = case_output.get("token_ids") or []
            diff_idx = first_token_diff(base_token_ids, case_token_ids)
            text_match = base_output["text"] == case_output["text"]
            token_match = diff_idx is None
            diffs.append(
                {
                    "id": base_output["id"],
                    "text_match": text_match,
                    "token_match": token_match,
                    "first_token_diff": diff_idx,
                    "baseline_text": base_output["text"],
                    "case_text": case_output["text"],
                    "baseline_token_ids": base_token_ids,
                    "case_token_ids": case_token_ids,
                }
            )
        results["comparisons"][case_name] = {
            "baseline": baseline_name,
            "all_text_match": all(diff["text_match"] for diff in diffs),
            "all_token_match": all(diff["token_match"] for diff in diffs),
            "diffs": diffs,
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, sort_keys=True)

    for case_name, comparison in results["comparisons"].items():
        print(
            f"{case_name}: text_match={comparison['all_text_match']} "
            f"token_match={comparison['all_token_match']}"
        )
        for diff in comparison["diffs"]:
            if not diff["token_match"]:
                print(
                    f"  {diff['id']}: first_token_diff="
                    f"{diff['first_token_diff']}"
                )


if __name__ == "__main__":
    main()
