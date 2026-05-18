# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run TREE_ATTN / DDT benchmark cases as isolated subprocesses.

Each case launches ``benchmark_tree_attn_ddt.py`` in a fresh process so model
state and GPU memory are released between dense-mask, compact-mask, repair, and
cudagraph probe runs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_CASES = [
    "ddt_dense_mask",
    "ddt_mask_kernel",
    "ddt_near_tie_repair",
]

DEFAULT_NEAR_TIE_REPAIR_SCOPE = "fallback_or_nonprefix"
GRAPH_COVERAGE_CASES = {
    "ddt_cudagraph_probe",
    "ddt_tree_verify_kernel_cudagraph_probe",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        nargs="+",
        default=DEFAULT_CASES,
        choices=[
            "vanilla",
            "tree_static",
            "tree_static_repair",
            "ddt_bridge",
            "ddt_target_mask",
            "ddt_dense_mask",
            "ddt_near_tie_repair",
            "ddt_mask_kernel",
            "ddt_tree_verify_kernel",
            "ddt_tree_verify_kernel_cudagraph_probe",
            "ddt_cudagraph_probe",
        ],
        help="Benchmark cases to run in isolated subprocesses.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("benchmarks/spec_decode/ddt_benchmark_matrix.jsonl"),
    )
    parser.add_argument(
        "--benchmark-arg",
        action="append",
        default=[],
        help=(
            "Extra argument forwarded to benchmark_tree_attn_ddt.py. Repeat "
            "for flag/value pairs, e.g. --benchmark-arg=--max-tokens "
            "--benchmark-arg=64."
        ),
    )
    parser.add_argument(
        "--cudagraph-compilation-config",
        default='{"cudagraph_mode":"piecewise"}',
        help=(
            "Compilation config appended only for ddt_cudagraph_probe when no "
            "--compilation-config was already forwarded."
        ),
    )
    parser.add_argument(
        "--near-tie-repair-scope",
        default=DEFAULT_NEAR_TIE_REPAIR_SCOPE,
        choices=["all", "nonprefix", "fallback_req", "fallback_or_nonprefix"],
        help=(
            "Default --serial-repair-scope appended only for "
            "repair/fallback cases when no explicit scope was forwarded."
        ),
    )
    parser.add_argument(
        "--correctness-summary",
        type=Path,
        default=None,
        help=(
            "Optional regression_summary.json or regression_repeat_summary.json "
            "that must pass before any benchmark case runs."
        ),
    )
    parser.add_argument(
        "--max-hard-fails",
        type=int,
        default=0,
        help="Maximum allowed hard_fail_count in --correctness-summary.",
    )
    parser.add_argument(
        "--compare-case",
        action="append",
        default=[],
        metavar="REF=CASE",
        help=(
            "Compare completed benchmark outputs without enabling trace. "
            "The comparator checks steady-state generated texts and spec "
            "decode metrics. Repeat for multiple pairs, for example "
            "--compare-case=ddt_mask_kernel=ddt_tree_verify_kernel."
        ),
    )
    parser.add_argument(
        "--compare-text-case",
        action="append",
        default=[],
        metavar="REF=CASE",
        help=(
            "Compare only steady-state generated texts. Use this for "
            "target-only vs speculative cases where spec decode counters are "
            "expected to differ but greedy output should stay identical."
        ),
    )
    parser.add_argument(
        "--speedup-case",
        action="append",
        default=[],
        metavar="REF=CASE",
        help=(
            "Require CASE steady-state throughput to be at least "
            "--min-speedup times REF. This is intended for clean no-trace "
            "benchmark comparisons, for example "
            "--speedup-case=tree_static=ddt_tree_verify_kernel."
        ),
    )
    parser.add_argument(
        "--min-speedup",
        type=float,
        default=1.0,
        help="Minimum throughput ratio for every --speedup-case pair.",
    )
    parser.add_argument(
        "--require-acceptance",
        action="append",
        default=[],
        metavar="CASE",
        help=(
            "Fail CASE unless steady-state spec decode produced at least "
            "--min-accepted-tokens accepted draft tokens. Use this for "
            "clean speed gates that must prove the speculative/DDT path was "
            "acceptance-producing."
        ),
    )
    parser.add_argument(
        "--min-accepted-tokens",
        type=int,
        default=1,
        help="Minimum accepted draft tokens for every --require-acceptance case.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "Run the complete case list repeatedly. Comparisons are evaluated "
            "within each repeat so kernel/reference stability can be used as "
            "a no-trace regression target."
        ),
    )
    parser.add_argument(
        "--require-graph-coverage",
        action="store_true",
        help=(
            "Fail cudagraph probe cases unless trace graph summary shows "
            "dispatch coverage, no eager fallback, and enough graph replay."
        ),
    )
    parser.add_argument(
        "--min-graph-replay-delta",
        type=int,
        default=1,
        help=(
            "Minimum merged cudagraph replay_count_delta required when "
            "--require-graph-coverage is enabled."
        ),
    )
    parser.add_argument(
        "--stability-case",
        action="append",
        default=[],
        metavar="CASE",
        help=(
            "Check token/text stability for CASE across repeats by comparing "
            "repeat 1 against every later repeat. Repeat this flag for "
            "multiple cases, e.g. --stability-case=vanilla."
        ),
    )
    parser.add_argument(
        "--require-stability",
        action="store_true",
        help=(
            "Fail the matrix if any --stability-case differs across repeats."
        ),
    )
    return parser.parse_args()


def command_for_case(args: argparse.Namespace, case: str) -> list[str]:
    if case == "ddt_tree_verify_kernel":
        benchmark_case = "ddt_mask_kernel"
    elif case == "ddt_tree_verify_kernel_cudagraph_probe":
        benchmark_case = "ddt_cudagraph_probe"
    else:
        benchmark_case = case
    cmd = [
        sys.executable,
        "benchmarks/spec_decode/benchmark_tree_attn_ddt.py",
        "--case",
        benchmark_case,
        *args.benchmark_arg,
    ]
    if (
        case in {"ddt_cudagraph_probe", "ddt_tree_verify_kernel_cudagraph_probe"}
        and "--compilation-config" not in args.benchmark_arg
    ):
        cmd += ["--compilation-config", args.cudagraph_compilation_config]
    if (
        case in {"ddt_near_tie_repair", "tree_static_repair"}
        and "--serial-repair-scope" not in args.benchmark_arg
    ):
        cmd += ["--serial-repair-scope", args.near_tie_repair_scope]
    if (
        case == "tree_static"
        and "--enable-static-kv-relocation" not in args.benchmark_arg
    ):
        cmd += ["--enable-static-kv-relocation"]
    if (
        case in {"ddt_tree_verify_kernel", "ddt_tree_verify_kernel_cudagraph_probe"}
        and "--tree-verify-kernel" not in cmd
    ):
        cmd += ["--tree-verify-kernel"]
    return cmd


def parse_last_json(stdout: str) -> dict:
    decoder = json.JSONDecoder()
    idx = 0
    last_obj = None
    while idx < len(stdout):
        start = stdout.find("{", idx)
        if start == -1:
            break
        try:
            obj, end = decoder.raw_decode(stdout[start:])
        except json.JSONDecodeError:
            idx = start + 1
            continue
        last_obj = obj
        idx = start + end
    if last_obj is None:
        raise ValueError("benchmark output did not contain a JSON object")
    return last_obj


def load_correctness_gate(path: Path | None, max_hard_fails: int) -> dict | None:
    if path is None:
        return None
    with path.open(encoding="utf-8") as f:
        summary = json.load(f)
    hard_fail_count = int(summary.get("hard_fail_count") or 0)
    if not summary.get("pass") or hard_fail_count > max_hard_fails:
        raise SystemExit(
            "correctness gate failed: "
            f"path={path} pass={summary.get('pass')} "
            f"hard_fail_count={hard_fail_count} max_hard_fails={max_hard_fails}"
        )
    return {
        "path": str(path),
        "pass": bool(summary.get("pass")),
        "hard_fail_count": hard_fail_count,
        "outcome_summary": summary.get("outcome_summary"),
        "repeat": summary.get("repeat"),
    }


def _spec_decode_metric_subset(result: dict) -> dict:
    spec = (result.get("steady_state") or {}).get("spec_decode") or {}
    return {
        "num_drafts": spec.get("num_drafts"),
        "num_draft_tokens": spec.get("num_draft_tokens"),
        "num_accepted_tokens": spec.get("num_accepted_tokens"),
        "acceptance_rate": spec.get("acceptance_rate"),
        "acceptance_length": spec.get("acceptance_length"),
        "per_position_acceptance_rates": spec.get("per_position_acceptance_rates"),
    }


def _steady_texts(result: dict) -> list[str]:
    texts = (result.get("steady_state") or {}).get("texts") or []
    return [str(text) for text in texts]


def _prompt_ids(result: dict) -> list[str]:
    prompt_ids = result.get("prompt_ids") or []
    return [str(prompt_id) for prompt_id in prompt_ids]


def _steady_token_ids(result: dict) -> list[list[int]]:
    token_ids = (result.get("steady_state") or {}).get("token_ids") or []
    return [[int(token_id) for token_id in row] for row in token_ids]


def _steady_throughput(result: dict) -> float | None:
    value = (result.get("steady_state") or {}).get("throughput_tok_s")
    if value is None:
        return None
    return float(value)


def evaluate_sdt_baseline_gate(case: str, result: dict) -> dict[str, Any]:
    required = case == "tree_static"
    spec = _spec_decode_metric_subset(result)
    num_drafts = int(spec.get("num_drafts") or 0)
    num_draft_tokens = int(spec.get("num_draft_tokens") or 0)
    static_kv_relocation = result.get("static_kv_relocation")
    canonical_case = result.get("canonical_case")
    reasons: list[str] = []
    if required:
        if canonical_case != "tree_static":
            reasons.append("canonical_case_not_tree_static")
        if static_kv_relocation is not True:
            reasons.append("static_kv_relocation_disabled")
        if num_drafts <= 0:
            reasons.append("num_drafts_zero")
        if num_draft_tokens <= 0:
            reasons.append("num_draft_tokens_zero")
    return {
        "required": required,
        "pass": not required or not reasons,
        "canonical_case": canonical_case,
        "static_kv_relocation": static_kv_relocation,
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "reasons": reasons,
    }


def evaluate_acceptance_gate(
    *,
    case: str,
    result: dict,
    required_cases: set[str],
    min_accepted_tokens: int,
) -> dict[str, Any]:
    required = case in required_cases
    spec = _spec_decode_metric_subset(result)
    num_drafts = int(spec.get("num_drafts") or 0)
    num_draft_tokens = int(spec.get("num_draft_tokens") or 0)
    num_accepted_tokens = int(spec.get("num_accepted_tokens") or 0)
    reasons: list[str] = []
    if required:
        if num_drafts <= 0:
            reasons.append("num_drafts_zero")
        if num_draft_tokens <= 0:
            reasons.append("num_draft_tokens_zero")
        if num_accepted_tokens < min_accepted_tokens:
            reasons.append("accepted_tokens_below_threshold")
    return {
        "required": required,
        "pass": not required or not reasons,
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted_tokens,
        "min_accepted_tokens": min_accepted_tokens,
        "reasons": reasons,
    }


def first_token_diff(lhs: list[int], rhs: list[int]) -> int | None:
    limit = min(len(lhs), len(rhs))
    for idx in range(limit):
        if lhs[idx] != rhs[idx]:
            return idx
    if len(lhs) != len(rhs):
        return limit
    return None


def _steady_graph_summary(result: dict) -> dict[str, Any]:
    summary = (result.get("steady_state") or {}).get("trace_graph_summary") or {}
    return dict(summary) if isinstance(summary, dict) else {}


def merge_graph_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    counter_fields = [
        "graph_records",
        "cudagraph_dispatch_hit_records",
        "cudagraph_eager_fallback_records",
        "cudagraph_metadata_buffered_records",
        "cudagraph_capture_count_delta",
        "cudagraph_replay_count_delta",
        "compact_kernel_records",
        "compact_kernel_requested_records",
        "metadata_host_staged_records",
        "metadata_device_buffered_records",
        "metadata_typed_view_records",
        "dynamic_select_vectorized_records",
    ]
    merged: dict[str, Any] = {field: 0 for field in counter_fields}
    bucket_counts: dict[str, int] = {}
    runtime_modes: dict[str, int] = {}
    fallback_reasons: dict[str, int] = {}
    metadata_buffer_reasons: dict[str, int] = {}
    for summary in summaries:
        for field in counter_fields:
            merged[field] += int(summary.get(field) or 0)
        for key, value in (summary.get("graph_key_buckets") or {}).items():
            bucket_counts[key] = bucket_counts.get(key, 0) + int(value)
        for key, value in (summary.get("cudagraph_runtime_modes") or {}).items():
            runtime_modes[key] = runtime_modes.get(key, 0) + int(value)
        for key, value in (summary.get("cudagraph_fallback_reasons") or {}).items():
            fallback_reasons[key] = fallback_reasons.get(key, 0) + int(value)
        for key, value in (
            summary.get("metadata_device_buffer_reasons") or {}
        ).items():
            metadata_buffer_reasons[key] = (
                metadata_buffer_reasons.get(key, 0) + int(value)
            )
    merged["graph_key_buckets"] = dict(sorted(bucket_counts.items()))
    merged["graph_key_bucket_count"] = len(bucket_counts)
    merged["cudagraph_runtime_modes"] = dict(sorted(runtime_modes.items()))
    merged["cudagraph_fallback_reasons"] = dict(sorted(fallback_reasons.items()))
    merged["metadata_device_buffer_reasons"] = dict(
        sorted(metadata_buffer_reasons.items())
    )
    return merged


def evaluate_graph_coverage_gate(
    *,
    case: str,
    graph_summary: dict[str, Any],
    min_graph_replay_delta: int,
) -> dict[str, Any]:
    required = case in GRAPH_COVERAGE_CASES
    graph_records = int(graph_summary.get("graph_records") or 0)
    dispatch_hits = int(graph_summary.get("cudagraph_dispatch_hit_records") or 0)
    eager_fallbacks = int(graph_summary.get("cudagraph_eager_fallback_records") or 0)
    metadata_buffered = int(
        graph_summary.get("cudagraph_metadata_buffered_records") or 0
    )
    metadata_device_buffered = int(
        graph_summary.get("metadata_device_buffered_records") or 0
    )
    metadata_typed_view = int(
        graph_summary.get("metadata_typed_view_records") or 0
    )
    dynamic_select_vectorized = int(
        graph_summary.get("dynamic_select_vectorized_records") or 0
    )
    replay_delta = int(graph_summary.get("cudagraph_replay_count_delta") or 0)
    fallback_reasons = graph_summary.get("cudagraph_fallback_reasons") or {}
    metadata_buffer_reasons = (
        graph_summary.get("metadata_device_buffer_reasons") or {}
    )
    graph_records_present = graph_records > 0
    passed = (
        not required
        or (
            graph_records_present
            and dispatch_hits == graph_records
            and eager_fallbacks == 0
            and not fallback_reasons
            and metadata_buffered == graph_records
            and metadata_device_buffered == graph_records
            and metadata_typed_view == graph_records
            and dynamic_select_vectorized == graph_records
            and not metadata_buffer_reasons
            and replay_delta >= min_graph_replay_delta
        )
    )
    return {
        "required": required,
        "pass": passed,
        "graph_records": graph_records,
        "dispatch_hit_records": dispatch_hits,
        "eager_fallback_records": eager_fallbacks,
        "metadata_buffered_records": metadata_buffered,
        "metadata_device_buffered_records": metadata_device_buffered,
        "metadata_typed_view_records": metadata_typed_view,
        "dynamic_select_vectorized_records": dynamic_select_vectorized,
        "replay_count_delta": replay_delta,
        "min_replay_count_delta": min_graph_replay_delta,
        "fallback_reasons": fallback_reasons,
        "metadata_device_buffer_reasons": metadata_buffer_reasons,
    }


def compare_case_results(
    ref_case: str,
    case: str,
    rows_by_case: dict[str, dict],
    *,
    compare_spec_decode: bool = True,
) -> dict:
    ref_row = rows_by_case.get(ref_case)
    row = rows_by_case.get(case)
    if ref_row is None or row is None:
        return {
            "ref_case": ref_case,
            "case": case,
            "available": False,
            "pass": False,
            "reason": "missing_case",
        }
    if ref_row.get("returncode") != 0 or row.get("returncode") != 0:
        return {
            "ref_case": ref_case,
            "case": case,
            "available": True,
            "pass": False,
            "reason": "nonzero_returncode",
            "ref_returncode": ref_row.get("returncode"),
            "returncode": row.get("returncode"),
        }
    ref_result = ref_row.get("result") or {}
    result = row.get("result") or {}
    ref_texts = _steady_texts(ref_result)
    texts = _steady_texts(result)
    ref_token_ids = _steady_token_ids(ref_result)
    token_ids = _steady_token_ids(result)
    text_match = ref_texts == texts
    first_text_diff = None
    if not text_match:
        compare_len = min(len(ref_texts), len(texts))
        first_text_diff = compare_len
        for idx in range(compare_len):
            if ref_texts[idx] != texts[idx]:
                first_text_diff = idx
                break
    token_match = ref_token_ids == token_ids
    first_token_diff_prompt = None
    first_token_diff_index = None
    first_token_diff_prompt_id = None
    first_token_diff_tokens = None
    if not token_match:
        compare_len = min(len(ref_token_ids), len(token_ids))
        first_token_diff_prompt = compare_len
        for idx in range(compare_len):
            diff_idx = first_token_diff(ref_token_ids[idx], token_ids[idx])
            if diff_idx is not None:
                first_token_diff_prompt = idx
                first_token_diff_index = diff_idx
                first_token_diff_tokens = {
                    "ref": ref_token_ids[idx][diff_idx]
                    if diff_idx < len(ref_token_ids[idx])
                    else None,
                    "case": token_ids[idx][diff_idx]
                    if diff_idx < len(token_ids[idx])
                    else None,
                }
                break
        prompt_ids = _prompt_ids(ref_result)
        if (
            first_token_diff_prompt is not None
            and first_token_diff_prompt < len(prompt_ids)
        ):
            first_token_diff_prompt_id = prompt_ids[first_token_diff_prompt]
    ref_spec = _spec_decode_metric_subset(ref_result)
    spec = _spec_decode_metric_subset(result)
    spec_match = ref_spec == spec
    ref_graph_summary = _steady_graph_summary(ref_result)
    graph_summary = _steady_graph_summary(result)
    passed = text_match and token_match and (
        spec_match if compare_spec_decode else True
    )
    return {
        "ref_case": ref_case,
        "case": case,
        "available": True,
        "pass": passed,
        "compare_spec_decode": compare_spec_decode,
        "text_match": text_match,
        "token_match": token_match,
        "spec_decode_match": spec_match,
        "graph_summary_match": ref_graph_summary == graph_summary,
        "first_text_diff": first_text_diff,
        "first_token_diff_prompt": first_token_diff_prompt,
        "first_token_diff_prompt_id": first_token_diff_prompt_id,
        "first_token_diff_index": first_token_diff_index,
        "first_token_diff_tokens": first_token_diff_tokens,
        "ref_num_texts": len(ref_texts),
        "num_texts": len(texts),
        "ref_num_token_rows": len(ref_token_ids),
        "num_token_rows": len(token_ids),
        "ref_spec_decode": ref_spec,
        "spec_decode": spec,
        "ref_graph_summary": ref_graph_summary,
        "graph_summary": graph_summary,
    }


def compare_speedup_results(
    ref_case: str,
    case: str,
    rows_by_case: dict[str, dict],
    *,
    min_speedup: float,
) -> dict[str, Any]:
    ref_row = rows_by_case.get(ref_case)
    row = rows_by_case.get(case)
    if ref_row is None or row is None:
        return {
            "ref_case": ref_case,
            "case": case,
            "available": False,
            "pass": False,
            "reason": "missing_case",
            "min_speedup": min_speedup,
        }
    if ref_row.get("returncode") != 0 or row.get("returncode") != 0:
        return {
            "ref_case": ref_case,
            "case": case,
            "available": True,
            "pass": False,
            "reason": "nonzero_returncode",
            "ref_returncode": ref_row.get("returncode"),
            "returncode": row.get("returncode"),
            "min_speedup": min_speedup,
        }
    ref_gate = ref_row.get("sdt_baseline_gate") or {}
    if ref_gate.get("required") and not ref_gate.get("pass"):
        return {
            "ref_case": ref_case,
            "case": case,
            "available": True,
            "pass": False,
            "reason": "invalid_ref_sdt_baseline",
            "ref_sdt_baseline_gate": ref_gate,
            "min_speedup": min_speedup,
        }
    ref_result = ref_row.get("result") or {}
    result = row.get("result") or {}
    ref_throughput = _steady_throughput(ref_result)
    throughput = _steady_throughput(result)
    if ref_throughput is None or throughput is None or ref_throughput <= 0:
        return {
            "ref_case": ref_case,
            "case": case,
            "available": True,
            "pass": False,
            "reason": "missing_or_invalid_throughput",
            "ref_throughput_tok_s": ref_throughput,
            "throughput_tok_s": throughput,
            "min_speedup": min_speedup,
        }
    speedup = throughput / ref_throughput
    return {
        "ref_case": ref_case,
        "case": case,
        "available": True,
        "pass": speedup >= min_speedup,
        "reason": None if speedup >= min_speedup else "speedup_below_threshold",
        "ref_throughput_tok_s": ref_throughput,
        "throughput_tok_s": throughput,
        "speedup": speedup,
        "min_speedup": min_speedup,
    }


def compare_repeat_stability(
    *,
    case: str,
    rows: list[dict],
) -> dict[str, Any]:
    case_rows = [
        row
        for row in rows
        if row.get("case") == case
        and row.get("kind") != "comparison"
        and row.get("returncode") == 0
    ]
    case_rows = sorted(case_rows, key=lambda row: int(row.get("repeat_index") or 0))
    if not case_rows:
        return {
            "case": case,
            "available": False,
            "pass": False,
            "reason": "missing_case",
            "repeat_count": 0,
            "comparisons": [],
        }
    ref_row = case_rows[0]
    comparisons = []
    for row in case_rows[1:]:
        repeat_index = int(row.get("repeat_index") or 0)
        comparison = compare_case_results(
            f"{case}@repeat1",
            f"{case}@repeat{repeat_index}",
            {
                f"{case}@repeat1": ref_row,
                f"{case}@repeat{repeat_index}": row,
            },
            compare_spec_decode=False,
        )
        comparison["ref_repeat_index"] = int(ref_row.get("repeat_index") or 0)
        comparison["repeat_index"] = repeat_index
        comparisons.append(comparison)
    return {
        "case": case,
        "available": True,
        "pass": all(comparison["pass"] for comparison in comparisons),
        "repeat_count": len(case_rows),
        "comparisons": comparisons,
    }


def parse_compare_cases(compare_cases: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in compare_cases:
        if "=" not in item:
            raise SystemExit(f"--compare-case must be REF=CASE, got {item!r}")
        ref_case, case = item.split("=", 1)
        if not ref_case or not case:
            raise SystemExit(f"--compare-case must be REF=CASE, got {item!r}")
        pairs.append((ref_case, case))
    return pairs


def _comparison_label(
    ref_case: str,
    case: str,
    *,
    compare_spec_decode: bool,
) -> str:
    kind = "compare" if compare_spec_decode else "compare_text"
    return f"{kind}:{ref_case}={case}"


def summarize_repeat_rows(
    *,
    repeat: int,
    rows: list[dict],
    compare_pairs: list[tuple[str, str, bool]],
) -> dict:
    case_rows = [
        row
        for row in rows
        if row.get("kind")
        not in {"comparison", "performance_comparison", "repeat_stability"}
    ]
    comparison_rows = [row for row in rows if row.get("kind") == "comparison"]
    performance_rows = [
        row for row in rows if row.get("kind") == "performance_comparison"
    ]
    comparison_summaries = []
    for ref_case, case, compare_spec_decode in compare_pairs:
        matching = [
            row
            for row in comparison_rows
            if (row.get("comparison") or {}).get("ref_case") == ref_case
            and (row.get("comparison") or {}).get("case") == case
            and (row.get("comparison") or {}).get("compare_spec_decode")
            == compare_spec_decode
        ]
        comparison_summaries.append(
            {
                "ref_case": ref_case,
                "case": case,
                "compare_spec_decode": compare_spec_decode,
                "pass_count": sum(1 for row in matching if row.get("returncode") == 0),
                "fail_count": sum(1 for row in matching if row.get("returncode") != 0),
                "total": len(matching),
            }
        )
    case_returncodes: dict[str, list[int | None]] = {}
    for row in case_rows:
        case_returncodes.setdefault(str(row.get("case")), []).append(
            row.get("returncode")
        )
    graph_summary = merge_graph_summaries(
        [
            row.get("graph_summary") or {}
            for row in case_rows
            if isinstance(row.get("graph_summary"), dict)
        ]
    )
    graph_coverage_gates = [
        {
            "case": row.get("case"),
            "repeat_index": row.get("repeat_index"),
            **(row.get("graph_coverage_gate") or {}),
        }
        for row in case_rows
        if isinstance(row.get("graph_coverage_gate"), dict)
    ]
    acceptance_gates = [
        {
            "case": row.get("case"),
            "repeat_index": row.get("repeat_index"),
            **(row.get("acceptance_gate") or {}),
        }
        for row in case_rows
        if isinstance(row.get("acceptance_gate"), dict)
    ]
    stability_rows = [
        row
        for row in rows
        if row.get("kind") == "repeat_stability"
    ]
    return {
        "kind": "repeat_summary",
        "repeat": repeat,
        "pass": all(row.get("returncode") == 0 for row in rows),
        "case_returncodes": case_returncodes,
        "comparisons": comparison_summaries,
        "performance_comparisons": [
            row.get("performance") for row in performance_rows
        ],
        "repeat_stability": [
            row.get("stability") for row in stability_rows
        ],
        "graph_summary": graph_summary,
        "graph_coverage_gates": graph_coverage_gates,
        "acceptance_gates": acceptance_gates,
    }


def main() -> None:
    args = parse_args()
    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")
    correctness_gate = load_correctness_gate(
        args.correctness_summary,
        args.max_hard_fails,
    )
    acceptance_cases = set(args.require_acceptance)
    compare_pairs = [
        (ref_case, case, True)
        for ref_case, case in parse_compare_cases(args.compare_case)
    ]
    compare_pairs.extend(
        (ref_case, case, False)
        for ref_case, case in parse_compare_cases(args.compare_text_case)
    )
    speedup_pairs = parse_compare_cases(args.speedup_case)
    all_rows: list[dict] = []
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("a", encoding="utf-8") as f:
        for repeat_index in range(1, args.repeat + 1):
            rows_by_case: dict[str, dict] = {}
            for case in args.cases:
                cmd = command_for_case(args, case)
                started = time.time()
                proc = subprocess.run(
                    cmd,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                row = {
                    "case": case,
                    "repeat_index": repeat_index,
                    "repeat": args.repeat,
                    "cmd": cmd,
                    "started_at": started,
                    "returncode": proc.returncode,
                    "correctness_gate": correctness_gate,
                    "stderr_tail": proc.stderr[-4000:],
                }
                if proc.returncode == 0:
                    row["result"] = parse_last_json(proc.stdout)
                    row["sdt_baseline_gate"] = evaluate_sdt_baseline_gate(
                        case,
                        row["result"],
                    )
                    row["acceptance_gate"] = evaluate_acceptance_gate(
                        case=case,
                        result=row["result"],
                        required_cases=acceptance_cases,
                        min_accepted_tokens=args.min_accepted_tokens,
                    )
                    row["graph_summary"] = (
                        row["result"]
                        .get("steady_state", {})
                        .get("trace_graph_summary", {})
                    )
                    row["graph_coverage_gate"] = evaluate_graph_coverage_gate(
                        case=case,
                        graph_summary=row["graph_summary"],
                        min_graph_replay_delta=args.min_graph_replay_delta,
                    )
                    if (
                        args.require_graph_coverage
                        and not row["graph_coverage_gate"]["pass"]
                    ):
                        row["returncode"] = 1
                        row["graph_coverage_required"] = True
                    if not row["sdt_baseline_gate"]["pass"]:
                        row["returncode"] = 1
                        row["sdt_baseline_required"] = True
                    if not row["acceptance_gate"]["pass"]:
                        row["returncode"] = 1
                        row["acceptance_required"] = True
                else:
                    row["stdout_tail"] = proc.stdout[-4000:]
                rows_by_case[case] = row
                all_rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                print(json.dumps(row, ensure_ascii=False), flush=True)
                if row["returncode"] != 0:
                    raise SystemExit(row["returncode"])
            for ref_case, case, compare_spec_decode in compare_pairs:
                comparison = compare_case_results(
                    ref_case,
                    case,
                    rows_by_case,
                    compare_spec_decode=compare_spec_decode,
                )
                row = {
                    "case": _comparison_label(
                        ref_case,
                        case,
                        compare_spec_decode=compare_spec_decode,
                    ),
                    "kind": "comparison",
                    "repeat_index": repeat_index,
                    "repeat": args.repeat,
                    "comparison": comparison,
                    "returncode": 0 if comparison["pass"] else 1,
                }
                all_rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                print(json.dumps(row, ensure_ascii=False), flush=True)
                if not comparison["pass"]:
                    raise SystemExit(1)
            for ref_case, case in speedup_pairs:
                performance = compare_speedup_results(
                    ref_case,
                    case,
                    rows_by_case,
                    min_speedup=args.min_speedup,
                )
                row = {
                    "case": f"speedup:{ref_case}={case}",
                    "kind": "performance_comparison",
                    "repeat_index": repeat_index,
                    "repeat": args.repeat,
                    "performance": performance,
                    "returncode": 0 if performance["pass"] else 1,
                }
                all_rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                print(json.dumps(row, ensure_ascii=False), flush=True)
                if not performance["pass"]:
                    raise SystemExit(1)
        for case in args.stability_case:
            stability = compare_repeat_stability(case=case, rows=all_rows)
            row = {
                "case": f"stability:{case}",
                "kind": "repeat_stability",
                "repeat": args.repeat,
                "stability": stability,
                "returncode": 0
                if (not args.require_stability or stability["pass"])
                else 1,
            }
            all_rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if row["returncode"] != 0:
                raise SystemExit(1)
        if args.repeat > 1:
            row = {
                "case": "summary:repeat",
                **summarize_repeat_rows(
                    repeat=args.repeat,
                    rows=all_rows,
                    compare_pairs=compare_pairs,
                ),
                "returncode": 0,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
