# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run the DDT correctness regression matrix against live vLLM servers.

This script intentionally does not start or stop vLLM.  The target-only and
DDT servers should already be running with trace paths that match
``--target-trace`` and ``--ddt-trace``.  The runner stitches together:

* tree_correctness_harness.py
* export_replay_bundles.py
* classify_replay_bundles.py
* summarize_replay_outcomes.py

The pass/fail criterion is outcome-level: raw token mismatches are acceptable
only when they are explained as low-margin/tie or target-only oracle-unstable.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.spec_decode.benchmark_tree_attn_ddt import (  # noqa: E402
    _trace_graph_summary_from_records,
)

SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class RegressionCell:
    name: str
    prompts: str
    max_tokens: int
    concurrency: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=("quick", "core", "full"),
        default="core",
        help=(
            "quick=fixed32 only; core=fixed32/fixed32_conc2/expanded64; "
            "full=core plus expanded64_conc2/fixed32_conc4/"
            "expanded128/expanded128_conc2."
        ),
    )
    parser.add_argument(
        "--cell",
        action="append",
        default=[],
        help=(
            "Run only the named matrix cell. Repeat to run multiple cells. "
            "Names are fixed32, fixed32_conc2, expanded64, "
            "expanded64_conc2, fixed32_conc4, expanded128, "
            "expanded128_conc2, fixed128, expanded256."
        ),
    )
    parser.add_argument("--target-url", default="http://127.0.0.1:8024")
    parser.add_argument("--ddt-url", default="http://127.0.0.1:8028")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--prompts",
        default="benchmarks/spec_decode/tree_correctness_prompts.jsonl",
    )
    parser.add_argument(
        "--expanded-prompts",
        default="benchmarks/spec_decode/tree_correctness_prompts_expanded.jsonl",
    )
    parser.add_argument("--target-trace", required=True)
    parser.add_argument("--ddt-trace", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--case-name", default="ddt")
    parser.add_argument("--target-case-name", default="target")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "Run the selected suite repeatedly against the same live servers. "
            "Use this for CUDA graph/serving stability checks."
        ),
    )
    parser.add_argument(
        "--skip-selfcheck",
        action="store_true",
        help="Skip target-only same-service self-checks for concurrent cells.",
    )
    parser.add_argument(
        "--max-hard-fails",
        type=int,
        default=0,
        help="Exit non-zero if outcome hard_fail count exceeds this value.",
    )
    parser.add_argument(
        "--low-margin-threshold",
        type=float,
        default=0.25,
        help=(
            "Forwarded to classify_replay_bundles.py. Use 0.5 for the "
            "wider near-tie policy validated by threshold sweeps."
        ),
    )
    parser.add_argument(
        "--fail-on-hard-fail",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def suite_cells(args: argparse.Namespace) -> list[RegressionCell]:
    fixed = args.prompts
    expanded = args.expanded_prompts
    cells = [
        RegressionCell("fixed32", fixed, 32, 1),
    ]
    if args.suite in {"core", "full"}:
        cells.extend(
            [
                RegressionCell("fixed32_conc2", fixed, 32, 2),
                RegressionCell("expanded64", expanded, 64, 1),
            ]
        )
    if args.suite == "full":
        cells.extend(
            [
                RegressionCell("expanded64_conc2", expanded, 64, 2),
                RegressionCell("fixed32_conc4", fixed, 32, 4),
                RegressionCell("expanded128", expanded, 128, 1),
                RegressionCell("expanded128_conc2", expanded, 128, 2),
            ]
        )
    if args.cell:
        optional_cells = {
            "fixed128": RegressionCell("fixed128", fixed, 128, 1),
            "expanded256": RegressionCell("expanded256", expanded, 256, 1),
        }
        for name in args.cell:
            if (
                name in optional_cells
                and all(cell.name != name for cell in cells)
            ):
                cells.append(optional_cells[name])
        selected = set(args.cell)
        unknown = selected.difference(cell.name for cell in cells)
        if unknown:
            raise ValueError(
                f"unknown regression cell(s) for suite {args.suite}: "
                f"{sorted(unknown)}"
            )
        cells = [cell for cell in cells if cell.name in selected]
    return cells


def run_command(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def iteration_output_dir(base_output_dir: Path, repeat: int, repeat_index: int) -> Path:
    if repeat == 1:
        return base_output_dir
    return base_output_dir / f"repeat_{repeat_index:02d}"


def truncate(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def copy_trace(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copyfile(src, dst)
    else:
        dst.write_text("", encoding="utf-8")


def read_trace_jsonl(path: str | Path) -> list[dict[str, Any]]:
    trace_path = Path(path)
    if not trace_path.exists():
        return []
    records: list[dict[str, Any]] = []
    with trace_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def harness_path(cell: RegressionCell, output_dir: Path) -> Path:
    return output_dir / "harness" / f"{cell.name}.json"


def trace_snapshot_path(
    cell: RegressionCell,
    output_dir: Path,
    trace_name: str,
) -> Path:
    return output_dir / "traces" / f"{cell.name}_{trace_name}.jsonl"


def run_cell(
    *,
    args: argparse.Namespace,
    cell: RegressionCell,
    output_dir: Path,
) -> dict[str, Any]:
    target_trace = Path(args.target_trace)
    ddt_trace = Path(args.ddt_trace)
    truncate(target_trace)
    truncate(ddt_trace)

    output = harness_path(cell, output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            sys.executable,
            str(SCRIPT_DIR / "tree_correctness_harness.py"),
            "--prompts",
            cell.prompts,
            "--case",
            f"{args.target_case_name}={args.target_url}",
            "--case",
            f"{args.case_name}={args.ddt_url}",
            "--model",
            args.model,
            "--max-tokens",
            str(cell.max_tokens),
            "--temperature",
            str(args.temperature),
            "--concurrency",
            str(cell.concurrency),
            "--timeout",
            str(args.timeout),
            "--trace",
            f"{args.target_case_name}={target_trace}",
            "--trace",
            f"{args.case_name}={ddt_trace}",
            "--output",
            str(output),
        ]
    )

    target_snapshot = trace_snapshot_path(cell, output_dir, "target")
    ddt_snapshot = trace_snapshot_path(cell, output_dir, "ddt")
    copy_trace(target_trace, target_snapshot)
    copy_trace(ddt_trace, ddt_snapshot)
    return {
        **asdict(cell),
        "harness": str(output),
        "target_trace": str(target_snapshot),
        "ddt_trace": str(ddt_snapshot),
    }


def run_selfcheck(
    *,
    args: argparse.Namespace,
    cell: RegressionCell,
    output_dir: Path,
) -> str:
    output = (
        output_dir
        / "selfcheck"
        / f"{cell.name}_target_only_conc{cell.concurrency}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            sys.executable,
            str(SCRIPT_DIR / "tree_correctness_harness.py"),
            "--prompts",
            cell.prompts,
            "--case",
            f"{args.target_case_name}_a={args.target_url}",
            "--case",
            f"{args.target_case_name}_b={args.target_url}",
            "--model",
            args.model,
            "--max-tokens",
            str(cell.max_tokens),
            "--temperature",
            str(args.temperature),
            "--concurrency",
            str(cell.concurrency),
            "--timeout",
            str(args.timeout),
            "--output",
            str(output),
        ]
    )
    return str(output)


def export_cell_bundles(
    *,
    args: argparse.Namespace,
    cell: dict[str, Any],
    output_dir: Path,
) -> None:
    cell_name = cell["name"]
    run_command(
        [
            sys.executable,
            str(SCRIPT_DIR / "export_replay_bundles.py"),
            "--harness-output",
            cell["harness"],
            "--case",
            args.case_name,
            "--trace",
            cell["ddt_trace"],
            "--output-dir",
            str(output_dir / "bundles" / cell_name),
            "--mode",
            "ddt_regression",
            "--case-prefix",
            cell_name,
        ]
    )
    run_command(
        [
            sys.executable,
            str(SCRIPT_DIR / "export_replay_bundles.py"),
            "--harness-output",
            cell["harness"],
            "--case",
            args.case_name,
            "--trace",
            cell["target_trace"],
            "--output-dir",
            str(output_dir / "target_bundles" / cell_name),
            "--mode",
            "target",
            "--case-prefix",
            f"target_{cell_name}",
            "--token-source",
            "baseline",
        ]
    )


def classify_bundles(output_dir: Path, low_margin_threshold: float) -> str:
    output = output_dir / "classification.json"
    run_command(
        [
            sys.executable,
            str(SCRIPT_DIR / "classify_replay_bundles.py"),
            "--case-bundle-glob",
            str(output_dir / "bundles" / "*" / "*.json"),
            "--target-bundle-glob",
            str(output_dir / "target_bundles" / "*" / "*.json"),
            "--output",
            str(output),
            "--low-margin-threshold",
            str(low_margin_threshold),
        ]
    )
    return str(output)


def summarize_outcomes(
    *,
    classification: str,
    selfchecks: list[str],
    output_dir: Path,
) -> str:
    output = output_dir / "outcomes.json"
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "summarize_replay_outcomes.py"),
        "--classification",
        classification,
        "--output",
        str(output),
    ]
    for selfcheck in selfchecks:
        cmd.extend(["--target-selfcheck-harness", selfcheck])
    run_command(cmd)
    return str(output)


def comparison_summary(harness_file: str, case_name: str) -> dict[str, Any]:
    with open(harness_file, encoding="utf-8") as f:
        harness = json.load(f)
    comparison = harness["comparisons"][case_name]
    diffs = [
        {
            "id": diff["id"],
            "first_token_diff": diff["first_token_diff"],
        }
        for diff in comparison["diffs"]
        if not diff["token_match"]
    ]
    return {
        "all_token_match": comparison["all_token_match"],
        "all_text_match": comparison["all_text_match"],
        "num_diffs": len(diffs),
        "diffs": diffs,
    }


def graph_summary_for_cell(cell: dict[str, Any]) -> dict[str, Any]:
    return _trace_graph_summary_from_records(read_trace_jsonl(cell["ddt_trace"]))


def merge_graph_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    counter_fields = [
        "graph_records",
        "graph_key_bucket_count",
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
            if field == "graph_key_bucket_count":
                continue
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


def graph_coverage_gate(
    graph_summary: dict[str, Any],
    *,
    require_graph_records: bool,
    min_graph_replay_delta: int = 1,
) -> dict[str, Any]:
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
    required = bool(require_graph_records)
    passed = (
        not required
        or (
            graph_records > 0
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


def write_summary(
    *,
    args: argparse.Namespace,
    cells: list[dict[str, Any]],
    selfchecks: list[str],
    classification: str,
    outcomes: str,
    output_dir: Path,
) -> dict[str, Any]:
    with open(outcomes, encoding="utf-8") as f:
        outcome_data = json.load(f)
    hard_fails = [
        row for row in outcome_data.get("rows", []) if row["outcome"] == "hard_fail"
    ]
    cell_graph_summaries = {
        cell["name"]: graph_summary_for_cell(cell) for cell in cells
    }
    merged_graph_summary = merge_graph_summaries(
        [cell_graph_summaries[cell["name"]] for cell in cells]
    )
    summary = {
        "suite": args.suite,
        "case_name": args.case_name,
        "target_url": args.target_url,
        "ddt_url": args.ddt_url,
        "low_margin_threshold": args.low_margin_threshold,
        "cells": [
            {
                **cell,
                "comparison": comparison_summary(cell["harness"], args.case_name),
                "graph_summary": cell_graph_summaries[cell["name"]],
            }
            for cell in cells
        ],
        "graph_summary": merged_graph_summary,
        "graph_coverage_gate": graph_coverage_gate(
            merged_graph_summary,
            require_graph_records=merged_graph_summary.get("graph_records", 0) > 0,
        ),
        "selfchecks": selfchecks,
        "classification": classification,
        "outcomes": outcomes,
        "outcome_summary": outcome_data.get("summary", {}),
        "hard_fail_count": len(hard_fails),
        "hard_fails": hard_fails,
        "pass": len(hard_fails) <= args.max_hard_fails,
    }
    output = output_dir / "regression_summary.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return summary


def run_once(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    cells = []
    for cell in suite_cells(args):
        cells.append(run_cell(args=args, cell=cell, output_dir=output_dir))

    selfchecks = []
    if not args.skip_selfcheck:
        seen_selfchecks: set[tuple[str, int, int]] = set()
        for cell in suite_cells(args):
            key = (cell.prompts, cell.max_tokens, cell.concurrency)
            if cell.concurrency <= 1 or key in seen_selfchecks:
                continue
            seen_selfchecks.add(key)
            selfchecks.append(
                run_selfcheck(args=args, cell=cell, output_dir=output_dir)
            )

    for cell in cells:
        export_cell_bundles(args=args, cell=cell, output_dir=output_dir)

    classification = classify_bundles(output_dir, args.low_margin_threshold)
    outcomes = summarize_outcomes(
        classification=classification,
        selfchecks=selfchecks,
        output_dir=output_dir,
    )
    summary = write_summary(
        args=args,
        cells=cells,
        selfchecks=selfchecks,
        classification=classification,
        outcomes=outcomes,
        output_dir=output_dir,
    )
    if args.fail_on_hard_fail and not summary["pass"]:
        raise SystemExit(1)
    return summary


def write_repeat_summary(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    hard_fail_count = sum(summary["hard_fail_count"] for summary in summaries)
    graph_summary = merge_graph_summaries(
        [summary.get("graph_summary", {}) for summary in summaries]
    )
    graph_gate = graph_coverage_gate(
        graph_summary,
        require_graph_records=graph_summary.get("graph_records", 0) > 0,
    )
    repeat_summary = {
        "suite": args.suite,
        "repeat": args.repeat,
        "target_url": args.target_url,
        "ddt_url": args.ddt_url,
        "low_margin_threshold": args.low_margin_threshold,
        "hard_fail_count": hard_fail_count,
        "graph_summary": graph_summary,
        "graph_coverage_gate": graph_gate,
        "pass": all(summary["pass"] for summary in summaries),
        "runs": [
            {
                "repeat_index": idx,
                "output_dir": str(iteration_output_dir(output_dir, args.repeat, idx)),
                "summary": str(
                    iteration_output_dir(output_dir, args.repeat, idx)
                    / "regression_summary.json"
                ),
                "pass": summary["pass"],
                "hard_fail_count": summary["hard_fail_count"],
                "graph_summary": summary.get("graph_summary", {}),
                "outcome_summary": summary["outcome_summary"],
            }
            for idx, summary in enumerate(summaries, start=1)
        ],
    }
    output = output_dir / "regression_repeat_summary.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(repeat_summary,
                  f,
                  indent=2,
                  ensure_ascii=False,
                  sort_keys=True)
    print(json.dumps(repeat_summary,
                     indent=2,
                     ensure_ascii=False,
                     sort_keys=True))
    return repeat_summary


def main() -> None:
    args = parse_args()
    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")

    output_dir = Path(args.output_dir)
    summaries = []
    for repeat_index in range(1, args.repeat + 1):
        iteration_dir = iteration_output_dir(output_dir, args.repeat, repeat_index)
        if args.repeat > 1 and iteration_dir.exists():
            shutil.rmtree(iteration_dir)
        summaries.append(run_once(args, iteration_dir))

    if args.repeat > 1:
        repeat_summary = write_repeat_summary(
            args=args,
            output_dir=output_dir,
            summaries=summaries,
        )
        if args.fail_on_hard_fail and not repeat_summary["pass"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
