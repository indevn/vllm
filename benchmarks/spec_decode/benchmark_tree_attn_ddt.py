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
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams

DEFAULT_PROMPTS = [
    "The capital of France is",
    "Write three colors:",
    "Explain speculative decoding in one sentence:",
    "List two benefits of GPU batching:",
]


def _full_kary_tree_choices(width: int, depth: int) -> list[tuple[int, ...]]:
    if width <= 0:
        raise ValueError(f"tree width must be positive, got {width}")
    if depth <= 0:
        raise ValueError(f"tree depth must be positive, got {depth}")
    choices: list[tuple[int, ...]] = []
    level: list[tuple[int, ...]] = [()]
    for _ in range(depth):
        next_level: list[tuple[int, ...]] = []
        for prefix in level:
            for child_idx in range(width):
                child = (*prefix, child_idx)
                choices.append(child)
                next_level.append(child)
        level = next_level
    return choices


def parse_tree_arg(tree: str) -> str:
    if tree == "auto":
        return tree
    if tree == "binary30":
        return repr(_full_kary_tree_choices(width=2, depth=4))
    if tree == "binary62":
        return repr(_full_kary_tree_choices(width=2, depth=5))
    ast.literal_eval(tree)
    return tree


def tree_num_nodes(tree: str) -> int | None:
    tree_literal = parse_tree_arg(tree)
    if tree_literal == "auto":
        return None
    return len(ast.literal_eval(tree_literal))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=[
            "vanilla",
            "tree_static",
            "tree_static_repair",
            "ddt_bridge",
            "ddt_target_mask",
            "ddt_dense_mask",
            "ddt_near_tie_repair",
            "ddt_mask_kernel",
            "ddt_cudagraph_probe",
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
        "--compilation-config",
        default=None,
        help=(
            "Optional JSON compilation_config passed to LLM. Useful for "
            "cudagraph probes, e.g. '{\"cudagraph_mode\":\"piecewise\"}'."
        ),
    )
    parser.add_argument(
        "--near-tie-threshold",
        type=float,
        default=0.25,
        help=(
            "Threshold exported for repair/fallback benchmark cases "
            "(ddt_near_tie_repair and tree_static_repair)."
        ),
    )
    parser.add_argument(
        "--serial-repair-scope",
        choices=["all", "nonprefix", "fallback_req", "fallback_or_nonprefix"],
        default="all",
        help=(
            "Accepted-state repair scope for the ddt_near_tie_repair case. "
            "The default preserves the correctness-first repair policy."
        ),
    )
    parser.add_argument(
        "--tree",
        default="[(0,), (1,), (0, 0), (0, 1)]",
        help=(
            "Static speculative token tree for TREE_ATTN cases. Use 'auto' to "
            "omit and let vLLM resolve the default chain. Presets: binary30, "
            "binary62."
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
            "verify subtree. Used by DDT cases."
        ),
    )
    parser.add_argument(
        "--ddt-runtime-mode",
        choices=["root_only", "prefix_only", "branching"],
        default="branching",
        help=(
            "Runtime DDT mode for DDT benchmark cases. The default benchmarks "
            "the full branching + relocation path so acceptance can be measured."
        ),
    )
    parser.add_argument(
        "--disable-ddt-kv-relocation",
        action="store_true",
        help="Disable dynamic tree KV/state relocation for DDT cases.",
    )
    parser.add_argument(
        "--enable-static-kv-relocation",
        action="store_true",
        help=(
            "Enable tree KV/state relocation for the tree_static baseline. "
            "This makes static branching TREE_ATTN a true acceptance-capable "
            "SDT comparison path instead of the correctness-safe suppressed "
            "draft path."
        ),
    )
    parser.add_argument(
        "--trace-path",
        type=Path,
        default=None,
        help=(
            "Optional VLLM_SPEC_VERIFY_STATE_TRACE_PATH. When set, the "
            "benchmark reports acceptance reconstructed from verifier traces."
        ),
    )
    parser.add_argument(
        "--stage-profile",
        action="store_true",
        help=(
            "Enable lightweight DDT runtime stage profiling in the verifier "
            "trace. The summary is reported under trace_stage_profile."
        ),
    )
    parser.add_argument(
        "--stage-profile-sync",
        action="store_true",
        help=(
            "Synchronize CUDA around profiled runtime stages. This improves "
            "attribution but should be used only for profiling, not throughput."
        ),
    )
    parser.add_argument(
        "--draft-stage-profile",
        action="store_true",
        help=(
            "Include drafter-internal substage timings in tree_attn_stage_profile "
            "records. This is diagnostic-only and should not be used for clean "
            "throughput benchmarks."
        ),
    )
    parser.add_argument(
        "--dynamic-metadata-stage-profile",
        action="store_true",
        help=(
            "Include dynamic draft tree metadata construction timings in "
            "tree_attn_stage_profile records. Diagnostic-only."
        ),
    )
    parser.add_argument(
        "--draft-stage-profile-sync",
        action="store_true",
        help=(
            "Synchronize CUDA around drafter substage profiling. Diagnostic-only."
        ),
    )
    parser.add_argument(
        "--sample-stage-profile",
        action="store_true",
        help=(
            "Include tree sampler/verifier substage timings in "
            "tree_attn_stage_profile records. Diagnostic-only."
        ),
    )
    parser.add_argument(
        "--sample-stage-profile-sync",
        action="store_true",
        help=(
            "Synchronize CUDA around tree sampler substage profiling. "
            "Diagnostic-only."
        ),
    )
    parser.add_argument(
        "--tree-verify-kernel",
        action="store_true",
        help=(
            "Use the experimental fused tree verifier sampler kernel. "
            "Correctness-gated diagnostic/perf candidate."
        ),
    )
    parser.add_argument(
        "--dynamic-metadata-select-path",
        choices=["default", "static_topk_kernel", "selected_bool_kernel"],
        default="default",
        help=(
            "DDT dynamic metadata selection implementation. This is a "
            "diagnostic/performance switch; correctness should be compared "
            "against the default/full-CPU oracle before treating throughput "
            "as comparable."
        ),
    )
    parser.add_argument(
        "--device-metadata-handle",
        action="store_true",
        help=(
            "Enable the experimental typed/device DDT metadata handle "
            "handoff. Requires a kernelized dynamic metadata select path."
        ),
    )
    parser.add_argument(
        "--torch-profiler-dir",
        type=Path,
        default=None,
        help=(
            "Enable vLLM torch profiler and write traces/tables under this "
            "directory. This is diagnostic-only and should not be used for "
            "clean throughput comparisons."
        ),
    )
    parser.add_argument(
        "--torch-profiler-prefix",
        default="ddt_profile",
        help="Trace name prefix used with --torch-profiler-dir.",
    )
    parser.add_argument(
        "--torch-profiler-active-iters",
        type=int,
        default=5,
        help="Active iterations for the torch profiler schedule.",
    )
    parser.add_argument(
        "--torch-profiler-warmup-iters",
        type=int,
        default=0,
        help="Warmup iterations for the torch profiler schedule.",
    )
    parser.add_argument(
        "--torch-profiler-record-shapes",
        action="store_true",
        help="Record tensor shapes in the torch profiler trace.",
    )
    parser.add_argument(
        "--torch-profiler-with-memory",
        action="store_true",
        help="Record memory events in the torch profiler trace.",
    )
    parser.add_argument(
        "--torch-profiler-with-stack",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Record stack traces in torch profiler. Disabled by default.",
    )
    parser.add_argument(
        "--custom-profile-scopes",
        action="store_true",
        help=(
            "Enable VLLM_CUSTOM_SCOPES_FOR_PROFILING so vLLM record_function "
            "scopes appear in torch profiler traces."
        ),
    )
    parser.add_argument(
        "--nvtx-profile-scopes",
        action="store_true",
        help=(
            "Enable VLLM_NVTX_SCOPES_FOR_PROFILING so vLLM scopes appear in "
            "nsys/NVTX traces."
        ),
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Optional JSONL prompt file with records containing a 'prompt' field.",
    )
    parser.add_argument(
        "--prompt-repeat",
        type=int,
        default=1,
        help=(
            "Repeat the loaded prompt set N times for clean offline batch-size "
            "expansion. This is not HTTP concurrency; serving concurrency is "
            "covered by tree_correctness_harness.py."
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


def canonical_case(case: str) -> str:
    if case in {"ddt_dense_mask", "ddt_near_tie_repair", "ddt_mask_kernel"}:
        return "ddt_target_mask"
    if case == "ddt_cudagraph_probe":
        return "ddt_target_mask"
    if case == "tree_static_repair":
        return "tree_static"
    return case


def build_llm(args: argparse.Namespace) -> LLM:
    if args.disable_mla:
        os.environ.setdefault("VLLM_MLA_DISABLE", "1")
    if getattr(args, "stage_profile", False) and args.trace_path is None:
        raise ValueError("--stage-profile requires --trace-path")
    if args.trace_path is not None:
        args.trace_path.parent.mkdir(parents=True, exist_ok=True)
        args.trace_path.write_text("", encoding="utf-8")
        os.environ["VLLM_SPEC_VERIFY_STATE_TRACE_PATH"] = str(args.trace_path)
    if getattr(args, "stage_profile", False):
        os.environ["VLLM_TREE_ATTN_STAGE_PROFILE"] = "1"
    if getattr(args, "stage_profile_sync", False):
        os.environ["VLLM_TREE_ATTN_STAGE_PROFILE_SYNC"] = "1"
    if getattr(args, "draft_stage_profile", False):
        os.environ["VLLM_TREE_ATTN_DRAFT_STAGE_PROFILE"] = "1"
    if getattr(args, "dynamic_metadata_stage_profile", False):
        os.environ["VLLM_DYNAMIC_TREE_METADATA_STAGE_PROFILE"] = "1"
    if getattr(args, "draft_stage_profile_sync", False):
        os.environ["VLLM_TREE_ATTN_DRAFT_STAGE_PROFILE_SYNC"] = "1"
    if getattr(args, "sample_stage_profile", False):
        os.environ["VLLM_TREE_ATTN_SAMPLE_STAGE_PROFILE"] = "1"
    if getattr(args, "sample_stage_profile_sync", False):
        os.environ["VLLM_TREE_ATTN_SAMPLE_STAGE_PROFILE_SYNC"] = "1"
    if getattr(args, "tree_verify_kernel", False):
        os.environ["VLLM_TREE_ATTN_VERIFY_KERNEL"] = "1"
    dynamic_metadata_select_path = getattr(
        args, "dynamic_metadata_select_path", "default"
    )
    if dynamic_metadata_select_path == "static_topk_kernel":
        os.environ["VLLM_DYNAMIC_TREE_STATIC_TOPK_METADATA_KERNEL"] = "1"
    elif dynamic_metadata_select_path == "selected_bool_kernel":
        os.environ["VLLM_DYNAMIC_TREE_SELECTED_BOOL_METADATA_KERNEL"] = "1"
    if getattr(args, "device_metadata_handle", False):
        os.environ["VLLM_DYNAMIC_TREE_DEVICE_METADATA_HANDLE"] = "1"
    if getattr(args, "custom_profile_scopes", False):
        os.environ["VLLM_CUSTOM_SCOPES_FOR_PROFILING"] = "1"
    if getattr(args, "nvtx_profile_scopes", False):
        os.environ["VLLM_NVTX_SCOPES_FOR_PROFILING"] = "1"
    if args.case in {"ddt_near_tie_repair", "tree_static_repair"}:
        os.environ.setdefault(
            "VLLM_TREE_ATTN_NEAR_TIE_Q1_FALLBACK_THRESHOLD",
            str(args.near_tie_threshold),
        )
        os.environ.setdefault("VLLM_TREE_ATTN_SERIAL_ACCEPTED_STATE_REPAIR", "1")
        os.environ.setdefault(
            "VLLM_TREE_ATTN_SERIAL_ACCEPTED_STATE_REPAIR_SCOPE",
            args.serial_repair_scope,
        )
    if args.case == "ddt_mask_kernel":
        os.environ.setdefault("VLLM_TREE_ATTN_DYNAMIC_MASK_KERNEL", "1")
    if args.case == "ddt_cudagraph_probe":
        os.environ.setdefault("VLLM_TREE_ATTN_DYNAMIC_MASK_KERNEL", "1")
        os.environ.setdefault("VLLM_TREE_ATTN_CUDAGRAPH_PROBE", "1")

    kwargs = dict(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_chunked_prefill=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_log_stats=False,
    )
    if args.compilation_config is not None:
        kwargs["compilation_config"] = json.loads(args.compilation_config)
    if args.torch_profiler_dir is not None:
        args.torch_profiler_dir.mkdir(parents=True, exist_ok=True)
        kwargs["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": str(args.torch_profiler_dir),
            "torch_profiler_record_shapes": args.torch_profiler_record_shapes,
            "torch_profiler_with_memory": args.torch_profiler_with_memory,
            "torch_profiler_with_stack": args.torch_profiler_with_stack,
            "active_iterations": args.torch_profiler_active_iters,
            "warmup_iterations": args.torch_profiler_warmup_iters,
        }
    target_attn_backend = args.target_attn_backend
    run_case = canonical_case(args.case)
    if target_attn_backend is None and run_case in {
        "ddt_bridge",
        "ddt_target_mask",
    }:
        target_attn_backend = "TREE_ATTN"
    if target_attn_backend:
        kwargs["attention_config"] = {"backend": target_attn_backend}
    if args.async_scheduling is not None:
        kwargs["async_scheduling"] = args.async_scheduling
    if run_case == "vanilla":
        return LLM(**kwargs)

    spec_config = {
        "method": args.method,
        "model": args.draft_model,
        "max_model_len": args.max_model_len,
    }
    tree_literal = parse_tree_arg(args.tree)
    if tree_literal == "auto":
        spec_config["num_speculative_tokens"] = 3
    else:
        spec_config["num_speculative_tokens"] = len(ast.literal_eval(tree_literal))
        spec_config["speculative_token_tree"] = tree_literal
    if args.draft_attn_backend != "auto":
        spec_config["attention_backend"] = args.draft_attn_backend
    if run_case in {"ddt_bridge", "ddt_target_mask"}:
        spec_config["enable_dynamic_draft_tree"] = True
        spec_config["dynamic_draft_tree_runtime_mode"] = args.ddt_runtime_mode
        if args.ddt_max_draft_tokens is not None:
            spec_config["dynamic_draft_tree_max_draft_tokens"] = (
                args.ddt_max_draft_tokens
            )
        if not args.disable_ddt_kv_relocation:
            spec_config["enable_tree_spec_decode_kv_relocation"] = True
    elif run_case == "tree_static" and (
        args.enable_static_kv_relocation or args.case == "tree_static_repair"
    ):
        spec_config["enable_tree_spec_decode_kv_relocation"] = True
    if run_case == "ddt_target_mask":
        spec_config["enable_dynamic_tree_target_mask"] = True

    return LLM(**kwargs, speculative_config=spec_config)


def load_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompt_file is None:
        prompts = list(args.prompts)
    else:
        prompts = []
        with args.prompt_file.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                prompts.append(str(row["prompt"]))
        if not prompts:
            raise ValueError(f"prompt file is empty: {args.prompt_file}")
    if args.prompt_repeat < 1:
        raise ValueError("--prompt-repeat must be >= 1")
    return prompts * args.prompt_repeat



def load_prompt_ids(args: argparse.Namespace, num_prompts: int) -> list[str]:
    if args.prompt_repeat < 1:
        raise ValueError("--prompt-repeat must be >= 1")
    if args.prompt_file is None:
        base_prompt_ids = [f"prompt_{idx}" for idx, _ in enumerate(args.prompts)]
    else:
        base_prompt_ids = []
        with args.prompt_file.open(encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                row = json.loads(line)
                base_prompt_ids.append(str(row.get("id", f"prompt_{idx}")))
    prompt_ids = [
        prompt_id if repeat_idx == 0 else f"{prompt_id}#r{repeat_idx}"
        for repeat_idx in range(args.prompt_repeat)
        for prompt_id in base_prompt_ids
    ]
    if len(prompt_ids) != num_prompts:
        raise ValueError(
            f"prompt id count {len(prompt_ids)} does not match prompt count "
            f"{num_prompts}"
        )
    return prompt_ids


def generated_token_count(outputs) -> int:
    total = 0
    for output in outputs:
        total += len(output.outputs[0].token_ids)
    return total


def generated_texts(outputs) -> list[str]:
    return [output.outputs[0].text for output in outputs]


def generated_token_ids(outputs) -> list[list[int]]:
    return [
        [int(token_id) for token_id in output.outputs[0].token_ids]
        for output in outputs
    ]


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


def _trace_acceptance_from_records(
    records: list[dict[str, Any]],
) -> dict[str, int | float | None]:
    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    non_prefix_accepts = 0
    relocation_pairs = 0
    compact_kernel_records = 0
    dense_mask_records = 0
    compact_kernel_requested_records = 0

    for record in records:
        if record.get("trace_kind") == "tree_attn_stage_profile":
            continue
        scheduled = record.get("scheduled_spec_decode_tokens") or []
        if not scheduled:
            continue
        num_drafts += 1
        num_draft_tokens += len(scheduled)
        accept_indices = record.get("accept_indices") or []
        valid_accept_indices = [
            int(local_idx)
            for local_idx in accept_indices
            if isinstance(local_idx, int) and local_idx >= 0
        ]
        accepted = max(0, len(valid_accept_indices) - 1)
        num_accepted_tokens += accepted
        for output_idx, local_idx in enumerate(valid_accept_indices[1:], start=1):
            if local_idx != output_idx:
                non_prefix_accepts += 1
        relocation_pairs += len(record.get("tree_relocation_pairs") or [])
        if record.get("tree_compact_bias_kernel_requested"):
            compact_kernel_requested_records += 1
        if record.get("tree_compact_bias_kernel_used"):
            compact_kernel_records += 1
        elif record.get("tree_parent") is not None and record.get(
            "tree_attn_bias_mask"
        ):
            dense_mask_records += 1

    return {
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted_tokens,
        "acceptance_rate": num_accepted_tokens / num_draft_tokens
        if num_draft_tokens
        else None,
        "acceptance_length": 1 + (num_accepted_tokens / num_drafts)
        if num_drafts
        else None,
        "non_prefix_accepts": non_prefix_accepts,
        "relocation_pairs": relocation_pairs,
        "dense_mask_records": dense_mask_records,
        "compact_kernel_records": compact_kernel_records,
        "compact_kernel_requested_records": compact_kernel_requested_records,
    }


def _trace_stage_profile_from_records(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    stage_records = [
        record
        for record in records
        if record.get("trace_kind") == "tree_attn_stage_profile"
    ]
    stage_totals_ms: dict[str, float] = {}
    draft_stage_totals_ms: dict[str, float] = {}
    dynamic_metadata_stage_totals_ms: dict[str, float] = {}
    sample_stage_totals_ms: dict[str, float] = {}
    total_stage_ms = 0.0
    output_tokens = 0
    accepted_tokens = 0
    scheduled_spec_decode_tokens = 0
    relocation_pairs = 0
    near_tie_fallback_rows = 0
    serial_repair_rows = 0
    serial_repair_prefix_rows = 0
    serial_repair_nonprefix_rows = 0
    serial_repair_near_tie_rows = 0
    compact_requested = 0
    compact_used = 0
    cudagraph_runtime_modes: dict[str, int] = {}
    cudagraph_fallback_reasons: dict[str, int] = {}
    cudagraph_metadata_buffered = 0
    cudagraph_dispatch_hits = 0
    cudagraph_eager_fallbacks = 0
    cudagraph_capture_count_delta = 0
    cudagraph_replay_count_delta = 0
    metadata_host_staged = 0
    metadata_device_buffered = 0
    metadata_typed_view = 0
    dynamic_select_vectorized = 0
    metadata_device_buffer_reasons: dict[str, int] = {}
    serial_repair_scopes: dict[str, int] = {}
    serial_repair_batch_by_depth_records = 0

    for record in stage_records:
        stage_ms = record.get("stage_ms") or {}
        for name, value in stage_ms.items():
            stage_totals_ms[name] = stage_totals_ms.get(name, 0.0) + float(value)
        draft_stage_ms = record.get("draft_stage_ms") or {}
        if isinstance(draft_stage_ms, dict):
            for name, value in draft_stage_ms.items():
                draft_stage_totals_ms[name] = (
                    draft_stage_totals_ms.get(name, 0.0) + float(value)
                )
        dynamic_metadata_stage_ms = (
            record.get("dynamic_metadata_stage_ms") or {}
        )
        if isinstance(dynamic_metadata_stage_ms, dict):
            for name, value in dynamic_metadata_stage_ms.items():
                dynamic_metadata_stage_totals_ms[name] = (
                    dynamic_metadata_stage_totals_ms.get(name, 0.0)
                    + float(value)
                )
        sample_stage_ms = record.get("sample_stage_ms") or {}
        if isinstance(sample_stage_ms, dict):
            for name, value in sample_stage_ms.items():
                sample_stage_totals_ms[name] = (
                    sample_stage_totals_ms.get(name, 0.0) + float(value)
                )
        total_stage_ms += float(record.get("stage_total_ms") or 0.0)
        output_tokens += int(record.get("output_tokens") or 0)
        accepted_tokens += int(record.get("accepted_tokens") or 0)
        scheduled_spec_decode_tokens += int(
            record.get("scheduled_spec_decode_tokens") or 0
        )
        relocation_pairs += int(record.get("relocation_pairs") or 0)
        near_tie_fallback_rows += int(record.get("near_tie_fallback_rows") or 0)
        serial_repair_rows += int(record.get("serial_repair_rows") or 0)
        serial_repair_prefix_rows += int(
            record.get("serial_repair_prefix_rows") or 0
        )
        serial_repair_nonprefix_rows += int(
            record.get("serial_repair_nonprefix_rows") or 0
        )
        serial_repair_near_tie_rows += int(
            record.get("serial_repair_near_tie_rows") or 0
        )
        if record.get("tree_compact_bias_kernel_requested"):
            compact_requested += 1
        if record.get("tree_compact_bias_kernel_used"):
            compact_used += 1
        runtime = record.get("tree_cudagraph_runtime") or {}
        if isinstance(runtime, dict):
            mode = str(runtime.get("mode"))
            cudagraph_runtime_modes[mode] = cudagraph_runtime_modes.get(mode, 0) + 1
            reason = runtime.get("fallback_reason")
            if reason:
                reason = str(reason)
                cudagraph_fallback_reasons[reason] = (
                    cudagraph_fallback_reasons.get(reason, 0) + 1
                )
            if runtime.get("metadata_buffered"):
                cudagraph_metadata_buffered += 1
            if runtime.get("dispatch_hit"):
                cudagraph_dispatch_hits += 1
            if runtime.get("eager_fallback"):
                cudagraph_eager_fallbacks += 1
            cudagraph_capture_count_delta += int(
                runtime.get("capture_count_delta") or 0
            )
            cudagraph_replay_count_delta += int(runtime.get("replay_count_delta") or 0)
        if record.get("tree_metadata_host_staged"):
            metadata_host_staged += 1
        if record.get("tree_metadata_device_buffered"):
            metadata_device_buffered += 1
        if record.get("tree_metadata_typed_view"):
            metadata_typed_view += 1
        if record.get("tree_dynamic_select_vectorized"):
            dynamic_select_vectorized += 1
        reason = record.get("tree_metadata_device_buffer_reason")
        if reason:
            reason = str(reason)
            metadata_device_buffer_reasons[reason] = (
                metadata_device_buffer_reasons.get(reason, 0) + 1
            )
        scope = record.get("serial_repair_scope")
        if scope:
            scope = str(scope)
            serial_repair_scopes[scope] = serial_repair_scopes.get(scope, 0) + 1
        if record.get("serial_repair_batch_by_depth"):
            serial_repair_batch_by_depth_records += 1

    count = len(stage_records)
    avg_stage_ms = {
        name: value / count for name, value in stage_totals_ms.items()
    } if count else {}
    pct_stage_ms = {
        name: value / total_stage_ms for name, value in stage_totals_ms.items()
    } if total_stage_ms else {}
    draft_stage_avg_ms = {
        name: value / count for name, value in draft_stage_totals_ms.items()
    } if count else {}
    draft_stage_total = sum(draft_stage_totals_ms.values())
    draft_stage_pct = {
        name: value / draft_stage_total
        for name, value in draft_stage_totals_ms.items()
    } if draft_stage_total else {}
    dynamic_metadata_stage_avg_ms = {
        name: value / count
        for name, value in dynamic_metadata_stage_totals_ms.items()
    } if count else {}
    dynamic_metadata_stage_total = sum(
        dynamic_metadata_stage_totals_ms.values()
    )
    dynamic_metadata_stage_pct = {
        name: value / dynamic_metadata_stage_total
        for name, value in dynamic_metadata_stage_totals_ms.items()
    } if dynamic_metadata_stage_total else {}
    sample_stage_avg_ms = {
        name: value / count for name, value in sample_stage_totals_ms.items()
    } if count else {}
    sample_stage_total = sum(sample_stage_totals_ms.values())
    sample_stage_pct = {
        name: value / sample_stage_total
        for name, value in sample_stage_totals_ms.items()
    } if sample_stage_total else {}
    if cudagraph_capture_count_delta == 0:
        cudagraph_capture_count_delta = _derive_cudagraph_counter_delta(
            stage_records, "capture"
        )
    if cudagraph_replay_count_delta == 0:
        cudagraph_replay_count_delta = _derive_cudagraph_counter_delta(
            stage_records, "replay"
        )

    return {
        "num_records": count,
        "stage_total_ms": total_stage_ms,
        "stage_total_ms_per_record": total_stage_ms / count if count else None,
        "stage_total_ms_per_output_token": (
            total_stage_ms / output_tokens if output_tokens else None
        ),
        "stage_totals_ms": stage_totals_ms,
        "stage_avg_ms": avg_stage_ms,
        "stage_pct": pct_stage_ms,
        "draft_stage_totals_ms": draft_stage_totals_ms,
        "draft_stage_avg_ms": draft_stage_avg_ms,
        "draft_stage_pct": draft_stage_pct,
        "dynamic_metadata_stage_totals_ms": (
            dynamic_metadata_stage_totals_ms
        ),
        "dynamic_metadata_stage_avg_ms": dynamic_metadata_stage_avg_ms,
        "dynamic_metadata_stage_pct": dynamic_metadata_stage_pct,
        "sample_stage_totals_ms": sample_stage_totals_ms,
        "sample_stage_avg_ms": sample_stage_avg_ms,
        "sample_stage_pct": sample_stage_pct,
        "output_tokens": output_tokens,
        "accepted_tokens": accepted_tokens,
        "scheduled_spec_decode_tokens": scheduled_spec_decode_tokens,
        "relocation_pairs": relocation_pairs,
        "near_tie_fallback_rows": near_tie_fallback_rows,
        "serial_repair_rows": serial_repair_rows,
        "serial_repair_prefix_rows": serial_repair_prefix_rows,
        "serial_repair_nonprefix_rows": serial_repair_nonprefix_rows,
        "serial_repair_near_tie_rows": serial_repair_near_tie_rows,
        "serial_repair_scopes": serial_repair_scopes,
        "serial_repair_batch_by_depth_records": (
            serial_repair_batch_by_depth_records
        ),
        "compact_kernel_records": compact_used,
        "compact_kernel_requested_records": compact_requested,
        "cudagraph_runtime_modes": cudagraph_runtime_modes,
        "cudagraph_fallback_reasons": cudagraph_fallback_reasons,
        "cudagraph_metadata_buffered_records": cudagraph_metadata_buffered,
        "cudagraph_dispatch_hit_records": cudagraph_dispatch_hits,
        "cudagraph_eager_fallback_records": cudagraph_eager_fallbacks,
        "cudagraph_capture_count_delta": cudagraph_capture_count_delta,
        "cudagraph_replay_count_delta": cudagraph_replay_count_delta,
        "metadata_host_staged_records": metadata_host_staged,
        "metadata_device_buffered_records": metadata_device_buffered,
        "metadata_typed_view_records": metadata_typed_view,
        "dynamic_select_vectorized_records": dynamic_select_vectorized,
        "metadata_device_buffer_reasons": metadata_device_buffer_reasons,
    }


def _graph_bucket_from_record(record: dict[str, Any]) -> str | None:
    key = record.get("tree_cudagraph_key") or {}
    runtime = record.get("tree_cudagraph_runtime") or {}
    if not isinstance(key, dict) or not isinstance(runtime, dict):
        return None
    mode = runtime.get("mode")
    if mode is None:
        return None
    batch_descriptor = runtime.get("batch_descriptor") or {}
    return "|".join(
        [
            f"mode={mode}",
            f"tree_width={key.get('tree_width')}",
            f"max_q={key.get('max_query_len')}",
            f"num_reqs={key.get('num_reqs')}",
            f"tokens={runtime.get('num_tokens_unpadded')}",
            f"padded={runtime.get('num_tokens_padded')}",
            f"bd_tokens={batch_descriptor.get('num_tokens')}",
            f"repair={key.get('serial_repair_scope')}",
            f"mask={key.get('dynamic_mask_kernel')}",
            f"fallback={runtime.get('fallback_reason')}",
        ]
    )


def _derive_cudagraph_counter_delta(
    records: list[dict[str, Any]],
    counter_name: str,
) -> int:
    total_values: list[int] = []
    for record in records:
        runtime = record.get("tree_cudagraph_runtime") or {}
        if not isinstance(runtime, dict):
            continue
        for suffix in ("dispatch", "record"):
            key = f"{counter_name}_count_total_at_{suffix}"
            value = runtime.get(key)
            if value is not None:
                total_values.append(int(value))
    if len(total_values) < 2:
        return 0
    return max(0, max(total_values) - min(total_values))


def _trace_graph_summary_from_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    graph_records = [
        record
        for record in records
        if isinstance(record.get("tree_cudagraph_runtime"), dict)
    ]
    bucket_counts: Counter[str] = Counter()
    runtime_modes: Counter[str] = Counter()
    fallback_reasons: Counter[str] = Counter()
    dispatch_hit_records = 0
    eager_fallback_records = 0
    metadata_buffered_records = 0
    capture_count_delta = 0
    replay_count_delta = 0
    compact_requested_records = 0
    compact_used_records = 0
    metadata_host_staged_records = 0
    metadata_device_buffered_records = 0
    metadata_typed_view_records = 0
    dynamic_select_vectorized_records = 0
    metadata_device_buffer_reasons: Counter[str] = Counter()

    for record in graph_records:
        bucket = _graph_bucket_from_record(record)
        if bucket is not None:
            bucket_counts[bucket] += 1
        runtime = record.get("tree_cudagraph_runtime") or {}
        runtime_modes[str(runtime.get("mode"))] += 1
        reason = runtime.get("fallback_reason")
        if reason:
            fallback_reasons[str(reason)] += 1
        if runtime.get("dispatch_hit"):
            dispatch_hit_records += 1
        if runtime.get("eager_fallback"):
            eager_fallback_records += 1
        if runtime.get("metadata_buffered"):
            metadata_buffered_records += 1
        capture_count_delta += int(runtime.get("capture_count_delta") or 0)
        replay_count_delta += int(runtime.get("replay_count_delta") or 0)
        if record.get("tree_compact_bias_kernel_requested"):
            compact_requested_records += 1
        if record.get("tree_compact_bias_kernel_used"):
            compact_used_records += 1
        if record.get("tree_metadata_host_staged"):
            metadata_host_staged_records += 1
        if record.get("tree_metadata_device_buffered"):
            metadata_device_buffered_records += 1
        if record.get("tree_metadata_typed_view"):
            metadata_typed_view_records += 1
        if record.get("tree_dynamic_select_vectorized"):
            dynamic_select_vectorized_records += 1
        reason = record.get("tree_metadata_device_buffer_reason")
        if reason:
            metadata_device_buffer_reasons[str(reason)] += 1

    if capture_count_delta == 0:
        capture_count_delta = _derive_cudagraph_counter_delta(
            graph_records, "capture"
        )
    if replay_count_delta == 0:
        replay_count_delta = _derive_cudagraph_counter_delta(
            graph_records, "replay"
        )

    return {
        "graph_records": len(graph_records),
        "graph_key_buckets": dict(sorted(bucket_counts.items())),
        "graph_key_bucket_count": len(bucket_counts),
        "cudagraph_runtime_modes": dict(sorted(runtime_modes.items())),
        "cudagraph_fallback_reasons": dict(sorted(fallback_reasons.items())),
        "cudagraph_dispatch_hit_records": dispatch_hit_records,
        "cudagraph_eager_fallback_records": eager_fallback_records,
        "cudagraph_metadata_buffered_records": metadata_buffered_records,
        "cudagraph_capture_count_delta": capture_count_delta,
        "cudagraph_replay_count_delta": replay_count_delta,
        "compact_kernel_records": compact_used_records,
        "compact_kernel_requested_records": compact_requested_records,
        "metadata_host_staged_records": metadata_host_staged_records,
        "metadata_device_buffered_records": metadata_device_buffered_records,
        "metadata_typed_view_records": metadata_typed_view_records,
        "dynamic_select_vectorized_records": dynamic_select_vectorized_records,
        "metadata_device_buffer_reasons": dict(
            sorted(metadata_device_buffer_reasons.items())
        ),
    }


def read_trace_records(
    trace_path: Path | None,
    start_offset: int,
) -> tuple[int, list[dict[str, Any]]]:
    if trace_path is None or not trace_path.exists():
        return start_offset, []
    with trace_path.open(encoding="utf-8") as f:
        f.seek(start_offset)
        records = []
        while line := f.readline():
            if line.strip():
                records.append(json.loads(line))
        end_offset = f.tell()
    return end_offset, records


def run_once(
    llm: LLM,
    prompts: Sequence[str],
    sampling_params: SamplingParams,
    trace_path: Path | None,
    trace_offset: int,
) -> tuple[
    float,
    int,
    list[str],
    list[list[int]],
    dict[str, float | int | list[float] | None],
    dict[str, int | float | None],
    dict[str, Any],
    dict[str, Any],
    int,
]:
    metrics_before = llm.get_metrics()
    start = time.perf_counter()
    outputs = llm.generate(list(prompts), sampling_params, use_tqdm=False)
    elapsed = time.perf_counter() - start
    metrics_after = llm.get_metrics()
    next_trace_offset, trace_records = read_trace_records(trace_path, trace_offset)
    return (
        elapsed,
        generated_token_count(outputs),
        generated_texts(outputs),
        generated_token_ids(outputs),
        spec_decode_metrics(metrics_before, metrics_after),
        _trace_acceptance_from_records(trace_records),
        _trace_stage_profile_from_records(trace_records),
        _trace_graph_summary_from_records(trace_records),
        next_trace_offset,
    )


def main() -> None:
    args = parse_args()
    run_case = canonical_case(args.case)
    prompts = load_prompts(args)
    prompt_ids = load_prompt_ids(args, len(prompts))
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_tokens,
        ignore_eos=False,
    )
    llm = build_llm(args)

    trace_offset = 0
    for _ in range(args.warmup_iters):
        *_, trace_offset = run_once(
            llm, prompts, sampling_params, args.trace_path, trace_offset
        )

    runs = []
    profile_started = False
    try:
        if args.torch_profiler_dir is not None:
            llm.start_profile(args.torch_profiler_prefix)
            profile_started = True
        for _ in range(args.iters):
            (
                elapsed,
                output_tokens,
                texts,
                token_ids,
                spec_metrics,
                trace_metrics,
                trace_stage_profile,
                trace_graph_summary,
                trace_offset,
            ) = run_once(
                llm, prompts, sampling_params, args.trace_path, trace_offset
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
                    "trace_acceptance": trace_metrics,
                    "trace_stage_profile": trace_stage_profile,
                    "trace_graph_summary": trace_graph_summary,
                    "texts": texts,
                    "token_ids": token_ids,
                }
            )
    finally:
        if profile_started:
            llm.stop_profile()

    steady = runs[-1]
    tree_literal = None if run_case == "vanilla" else parse_tree_arg(args.tree)
    tree_literal_for_output = (
        tree_literal
        if tree_literal is not None
        and args.tree not in {"binary30", "binary62"}
        else None
    )
    print(
        json.dumps(
            {
                "case": args.case,
                "canonical_case": run_case,
                "model": args.model,
                "draft_model": None
                if run_case == "vanilla"
                else args.draft_model,
                "method": None if run_case == "vanilla" else args.method,
                "tree": None if run_case == "vanilla" else args.tree,
                "tree_num_nodes": None
                if run_case == "vanilla"
                else tree_num_nodes(args.tree),
                "tree_literal": tree_literal_for_output,
                "ddt_runtime_mode": args.ddt_runtime_mode
                if run_case in {"ddt_bridge", "ddt_target_mask"}
                else None,
                "ddt_kv_relocation": (
                    not args.disable_ddt_kv_relocation
                    if run_case in {"ddt_bridge", "ddt_target_mask"}
                    else None
                ),
                "static_kv_relocation": (
                    args.enable_static_kv_relocation
                    or args.case == "tree_static_repair"
                    if run_case == "tree_static"
                    else None
                ),
                "ddt_max_draft_tokens": args.ddt_max_draft_tokens,
                "compilation_config": args.compilation_config,
                "near_tie_threshold": args.near_tie_threshold
                if args.case in {"ddt_near_tie_repair", "tree_static_repair"}
                else None,
                "serial_repair_scope": args.serial_repair_scope
                if args.case in {"ddt_near_tie_repair", "tree_static_repair"}
                else None,
                "prompt_file": str(args.prompt_file)
                if args.prompt_file is not None
                else None,
                "prompt_repeat": args.prompt_repeat,
                "trace_path": str(args.trace_path)
                if args.trace_path is not None
                else None,
                "stage_profile": args.stage_profile,
                "stage_profile_sync": args.stage_profile_sync,
                "dynamic_metadata_stage_profile": (
                    args.dynamic_metadata_stage_profile
                ),
                "torch_profiler_dir": str(args.torch_profiler_dir)
                if args.torch_profiler_dir is not None
                else None,
                "torch_profiler_prefix": args.torch_profiler_prefix
                if args.torch_profiler_dir is not None
                else None,
                "custom_profile_scopes": args.custom_profile_scopes,
                "nvtx_profile_scopes": args.nvtx_profile_scopes,
                "dynamic_metadata_select_path": args.dynamic_metadata_select_path,
                "device_metadata_handle": args.device_metadata_handle,
                "num_prompts": len(prompts),
                "prompt_ids": prompt_ids,
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
