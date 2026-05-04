# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Local TREE_ATTN / Dynamic Draft Tree runtime benchmark.

This benchmark is intentionally small and offline-friendly. It measures
end-to-end ``LLM.generate`` wall time after warmup and reports an approximate
inter-token latency from total generated tokens. It is not a streaming TTFT/ITL
benchmark.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import time
from collections.abc import Sequence
from typing import Any

from vllm import LLM, SamplingParams

DEFAULT_PROMPTS = [
    "The capital of France is",
    "Write three colors:",
    "Explain speculative decoding in one sentence:",
    "List two benefits of GPU batching:",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=[
            "vanilla",
            "tree_static",
            "ddt_bridge",
            "ddt_target_mask",
        ],
        required=True,
        help=(
            "Benchmark case to run. Run this script once per case to release "
            "GPU memory."
        ),
    )
    parser.add_argument("--model", default="eagle618/deepseek-v3-random")
    parser.add_argument("--draft-model", default="eagle618/eagle-deepseek-v3-random")
    parser.add_argument(
        "--method",
        default="eagle",
        choices=["eagle", "eagle3"],
        help="Speculative decoding method for TREE_ATTN cases.",
    )
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument(
        "--async-scheduling",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override scheduler async scheduling.",
    )
    parser.add_argument(
        "--tree",
        default="[(0,), (1,), (0, 0), (0, 1)]",
        help=(
            "Static speculative token tree for TREE_ATTN cases. Use 'auto' to "
            "omit and let vLLM resolve the default chain."
        ),
    )
    parser.add_argument(
        "--draft-attn-backend",
        default="TREE_ATTN",
        help="Attention backend for the speculative drafter. Use 'auto' to omit.",
    )
    parser.add_argument(
        "--target-attn-backend",
        default=None,
        help="Attention backend for the target model. Defaults to TREE_ATTN for "
        "ddt_bridge and auto for the other cases.",
    )
    parser.add_argument(
        "--ddt-max-draft-tokens",
        type=int,
        default=None,
        help=(
            "Maximum number of draft nodes selected into the runtime DDT "
            "verify subtree. Only used by ddt_bridge."
        ),
    )
    parser.add_argument(
        "--disable-mla",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Set VLLM_MLA_DISABLE=1 so DeepSeek random EAGLE can use TREE_ATTN.",
    )
    parser.add_argument(
        "--prompts",
        nargs="*",
        default=DEFAULT_PROMPTS,
        help="Prompts to benchmark.",
    )
    return parser.parse_args()


def build_llm(args: argparse.Namespace) -> LLM:
    if args.disable_mla:
        os.environ.setdefault("VLLM_MLA_DISABLE", "1")

    kwargs = dict(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_chunked_prefill=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_log_stats=False,
    )
    target_attn_backend = args.target_attn_backend
    if target_attn_backend is None and args.case in {
        "ddt_bridge",
        "ddt_target_mask",
    }:
        target_attn_backend = "TREE_ATTN"
    if target_attn_backend:
        kwargs["attention_config"] = {"backend": target_attn_backend}
    if args.async_scheduling is not None:
        kwargs["async_scheduling"] = args.async_scheduling
    if args.case == "vanilla":
        return LLM(**kwargs)

    spec_config = {
        "method": args.method,
        "model": args.draft_model,
        "max_model_len": args.max_model_len,
    }
    if args.tree == "auto":
        spec_config["num_speculative_tokens"] = 3
    else:
        spec_config["num_speculative_tokens"] = len(ast.literal_eval(args.tree))
        spec_config["speculative_token_tree"] = args.tree
    if args.draft_attn_backend != "auto":
        spec_config["attention_backend"] = args.draft_attn_backend
    if args.case in {"ddt_bridge", "ddt_target_mask"}:
        spec_config["enable_dynamic_draft_tree"] = True
        if args.ddt_max_draft_tokens is not None:
            spec_config["dynamic_draft_tree_max_draft_tokens"] = (
                args.ddt_max_draft_tokens
            )
    if args.case == "ddt_target_mask":
        spec_config["enable_dynamic_tree_target_mask"] = True

    return LLM(**kwargs, speculative_config=spec_config)


def generated_token_count(outputs) -> int:
    total = 0
    for output in outputs:
        total += len(output.outputs[0].token_ids)
    return total


def generated_texts(outputs) -> list[str]:
    return [output.outputs[0].text for output in outputs]


def metric_counter(metrics: Sequence[Any], name: str) -> int:
    total = 0
    for metric in metrics:
        if metric.name == name and hasattr(metric, "value"):
            total += int(metric.value)
    return total


def metric_vector(metrics: Sequence[Any], name: str) -> list[int]:
    total: list[int] = []
    for metric in metrics:
        if metric.name != name or not hasattr(metric, "values"):
            continue
        values = [int(value) for value in metric.values]
        if not total:
            total = [0] * len(values)
        for i, value in enumerate(values):
            total[i] += value
    return total


def spec_decode_metrics(
    before: Sequence[Any],
    after: Sequence[Any],
) -> dict[str, float | int | list[float] | None]:
    num_drafts = metric_counter(after, "vllm:spec_decode_num_drafts") - metric_counter(
        before, "vllm:spec_decode_num_drafts"
    )
    num_draft_tokens = metric_counter(
        after, "vllm:spec_decode_num_draft_tokens"
    ) - metric_counter(before, "vllm:spec_decode_num_draft_tokens")
    num_accepted_tokens = metric_counter(
        after, "vllm:spec_decode_num_accepted_tokens"
    ) - metric_counter(before, "vllm:spec_decode_num_accepted_tokens")

    before_per_pos = metric_vector(
        before, "vllm:spec_decode_num_accepted_tokens_per_pos"
    )
    after_per_pos = metric_vector(after, "vllm:spec_decode_num_accepted_tokens_per_pos")
    per_pos_acceptance_rates: list[float] = []
    if after_per_pos and num_drafts > 0:
        if not before_per_pos:
            before_per_pos = [0] * len(after_per_pos)
        per_pos_acceptance_rates = [
            (after_value - before_value) / num_drafts
            for before_value, after_value in zip(before_per_pos, after_per_pos)
        ]

    acceptance_rate = (
        num_accepted_tokens / num_draft_tokens if num_draft_tokens > 0 else None
    )
    acceptance_length = (
        1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else None
    )

    return {
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted_tokens,
        "acceptance_rate": acceptance_rate,
        "acceptance_length": acceptance_length,
        "per_position_acceptance_rates": per_pos_acceptance_rates or None,
    }


def run_once(
    llm: LLM,
    prompts: Sequence[str],
    sampling_params: SamplingParams,
) -> tuple[float, int, list[str], dict[str, float | int | list[float] | None]]:
    metrics_before = llm.get_metrics()
    start = time.perf_counter()
    outputs = llm.generate(list(prompts), sampling_params, use_tqdm=False)
    elapsed = time.perf_counter() - start
    metrics_after = llm.get_metrics()
    return (
        elapsed,
        generated_token_count(outputs),
        generated_texts(outputs),
        spec_decode_metrics(metrics_before, metrics_after),
    )


def main() -> None:
    args = parse_args()
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_tokens,
        ignore_eos=False,
    )
    llm = build_llm(args)

    for _ in range(args.warmup_iters):
        run_once(llm, args.prompts, sampling_params)

    runs = []
    for _ in range(args.iters):
        elapsed, output_tokens, texts, spec_metrics = run_once(
            llm, args.prompts, sampling_params
        )
        runs.append(
            {
                "elapsed_s": elapsed,
                "output_tokens": output_tokens,
                "approx_itl_ms": (elapsed / output_tokens * 1000)
                if output_tokens
                else None,
                "throughput_tok_s": (output_tokens / elapsed) if elapsed else None,
                "spec_decode": spec_metrics,
                "texts": texts,
            }
        )

    steady = runs[-1]
    print(
        json.dumps(
            {
                "case": args.case,
                "model": args.model,
                "draft_model": None if args.case == "vanilla" else args.draft_model,
                "method": None if args.case == "vanilla" else args.method,
                "tree": None if args.case == "vanilla" else args.tree,
                "num_prompts": len(args.prompts),
                "max_tokens": args.max_tokens,
                "warmup_iters": args.warmup_iters,
                "iters": args.iters,
                "steady_state": steady,
                "runs": runs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
