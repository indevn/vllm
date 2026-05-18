# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run standalone large-topology DDT correctness classification.

This runner closes the evidence gap between local benchmark experiments and
the controlled-replay correctness policy.  It runs or reuses standalone
``benchmark_tree_attn_ddt.py`` matrix outputs, converts them into correctness
harness JSON, exports replay bundles, classifies first-diff rows, and writes an
outcome summary.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.spec_decode.benchmark_matrix_to_harness import (  # noqa: E402
    find_case_row,
    load_matrix_rows,
)
from benchmarks.spec_decode.run_tree_attn_ddt_benchmark_matrix import (  # noqa: E402
    evaluate_graph_coverage_gate,
    merge_graph_summaries,
)

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tree", default="binary62")
    parser.add_argument(
        "--ref-case",
        default="vanilla",
        help="Reference benchmark case, normally target-only vanilla.",
    )
    parser.add_argument(
        "--case",
        default="ddt_tree_verify_kernel",
        help="DDT benchmark matrix case to classify against --ref-case.",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--draft-model",
        default="RedHatAI/Qwen3-8B-speculator.eagle3",
    )
    parser.add_argument("--method", default="eagle3")
    parser.add_argument("--target-attn-backend", default="TREE_ATTN")
    parser.add_argument(
        "--prompt-file",
        default="benchmarks/spec_decode/tree_correctness_prompts_expanded.jsonl",
    )
    parser.add_argument(
        "--prompt-repeat",
        type=int,
        default=1,
        help=(
            "Repeat the prompt set for offline batch expansion. "
            "This is forwarded to benchmark_tree_attn_ddt.py."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--warmup-iters", type=int, default=0)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--ddt-max-draft-tokens", type=int, default=3)
    parser.add_argument(
        "--dynamic-metadata-select-path",
        choices=["default", "static_topk_kernel", "selected_bool_kernel"],
        default="default",
        help=(
            "Forwarded to benchmark_tree_attn_ddt.py for DDT metadata "
            "handoff correctness runs."
        ),
    )
    parser.add_argument(
        "--device-metadata-handle",
        action="store_true",
        help=(
            "Forward --device-metadata-handle to benchmark_tree_attn_ddt.py "
            "so large-topology correctness can gate the typed/device "
            "metadata handoff path."
        ),
    )
    parser.add_argument(
        "--correctness-summary",
        type=Path,
        default=None,
        help="Optional existing correctness summary gate for benchmark runs.",
    )
    parser.add_argument("--max-hard-fails", type=int, default=0)
    parser.add_argument("--low-margin-threshold", type=float, default=0.5)
    parser.add_argument(
        "--require-graph-coverage",
        action="store_true",
        help=(
            "Require the DDT case graph summary to show full graph dispatch, "
            "metadata buffer, typed view, dynamic select, and compact-kernel "
            "coverage."
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
        "--fail-on-hard-fail",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help=(
            "Reuse --matrix-jsonl, --ref-trace, and --case-trace instead of "
            "running benchmark subprocesses. Useful for reclassifying an "
            "existing trace bundle after classifier changes."
        ),
    )
    parser.add_argument("--matrix-jsonl", type=Path, default=None)
    parser.add_argument("--ref-trace", type=Path, default=None)
    parser.add_argument("--case-trace", type=Path, default=None)
    return parser.parse_args()


def run_command(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def default_matrix_path(output_dir: Path) -> Path:
    return output_dir / "trace_matrix.jsonl"


def default_trace_path(output_dir: Path, case: str) -> Path:
    return output_dir / "traces" / f"{case}.jsonl"


def benchmark_args(args: argparse.Namespace, trace_path: Path) -> list[str]:
    prompt_repeat = getattr(args, "prompt_repeat", 1)
    forwarded = [
        "--benchmark-arg=--model",
        f"--benchmark-arg={args.model}",
        "--benchmark-arg=--draft-model",
        f"--benchmark-arg={args.draft_model}",
        "--benchmark-arg=--method",
        f"--benchmark-arg={args.method}",
        "--benchmark-arg=--target-attn-backend",
        f"--benchmark-arg={args.target_attn_backend}",
        "--benchmark-arg=--tree",
        f"--benchmark-arg={args.tree}",
        "--benchmark-arg=--prompt-file",
        f"--benchmark-arg={args.prompt_file}",
        "--benchmark-arg=--prompt-repeat",
        f"--benchmark-arg={prompt_repeat}",
        "--benchmark-arg=--max-tokens",
        f"--benchmark-arg={args.max_tokens}",
        "--benchmark-arg=--warmup-iters",
        f"--benchmark-arg={args.warmup_iters}",
        "--benchmark-arg=--iters",
        f"--benchmark-arg={args.iters}",
        "--benchmark-arg=--gpu-memory-utilization",
        f"--benchmark-arg={args.gpu_memory_utilization}",
        "--benchmark-arg=--max-num-batched-tokens",
        f"--benchmark-arg={args.max_num_batched_tokens}",
        "--benchmark-arg=--ddt-max-draft-tokens",
        f"--benchmark-arg={args.ddt_max_draft_tokens}",
        "--benchmark-arg=--dynamic-metadata-select-path",
        f"--benchmark-arg={args.dynamic_metadata_select_path}",
        "--benchmark-arg=--trace-path",
        f"--benchmark-arg={trace_path}",
    ]
    if args.device_metadata_handle:
        forwarded.append("--benchmark-arg=--device-metadata-handle")
    return forwarded


def run_benchmark_case(
    *,
    args: argparse.Namespace,
    matrix_path: Path,
    case: str,
    trace_path: Path,
) -> None:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_tree_attn_ddt_benchmark_matrix.py"),
        "--cases",
        case,
        "--output-jsonl",
        str(matrix_path),
        *benchmark_args(args, trace_path),
    ]
    if args.correctness_summary is not None:
        cmd.extend(
            [
                "--correctness-summary",
                str(args.correctness_summary),
                "--max-hard-fails",
                str(args.max_hard_fails),
            ]
        )
    run_command(cmd)


def export_bundles(
    *,
    args: argparse.Namespace,
    harness_path: Path,
    ref_trace: Path,
    case_trace: Path,
    output_dir: Path,
) -> None:
    run_command(
        [
            sys.executable,
            str(SCRIPT_DIR / "export_replay_bundles.py"),
            "--harness-output",
            str(harness_path),
            "--case",
            args.case,
            "--trace",
            str(case_trace),
            "--output-dir",
            str(output_dir / "bundles" / args.tree),
            "--mode",
            f"{args.case}_{args.tree}",
            "--case-prefix",
            args.tree,
        ]
    )
    run_command(
        [
            sys.executable,
            str(SCRIPT_DIR / "export_replay_bundles.py"),
            "--harness-output",
            str(harness_path),
            "--case",
            args.case,
            "--trace",
            str(ref_trace),
            "--output-dir",
            str(output_dir / "target_bundles" / args.tree),
            "--mode",
            f"{args.ref_case}_{args.tree}",
            "--case-prefix",
            f"{args.ref_case}_{args.tree}",
            "--token-source",
            "baseline",
        ]
    )


def classify_and_summarize(
    *,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[Path, Path]:
    classification = output_dir / "classification.json"
    run_command(
        [
            sys.executable,
            str(SCRIPT_DIR / "classify_replay_bundles.py"),
            "--case-bundle-glob",
            str(output_dir / "bundles" / "*" / "*.json"),
            "--target-bundle-glob",
            str(output_dir / "target_bundles" / "*" / "*.json"),
            "--output",
            str(classification),
            "--low-margin-threshold",
            str(args.low_margin_threshold),
        ]
    )
    outcomes = output_dir / "outcomes.json"
    run_command(
        [
            sys.executable,
            str(SCRIPT_DIR / "summarize_replay_outcomes.py"),
            "--classification",
            str(classification),
            "--output",
            str(outcomes),
        ]
    )
    return classification, outcomes


def write_harness(
    *,
    args: argparse.Namespace,
    matrix_path: Path,
    output_dir: Path,
) -> Path:
    harness = output_dir / "harness.json"
    run_command(
        [
            sys.executable,
            str(SCRIPT_DIR / "benchmark_matrix_to_harness.py"),
            "--matrix-jsonl",
            str(matrix_path),
            "--ref-case",
            args.ref_case,
            "--case",
            args.case,
            "--output",
            str(harness),
            "--repeat-index",
            "1",
        ]
    )
    return harness


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def graph_summary_for_cases(
    matrix_path: Path,
    *,
    ref_case: str,
    case: str,
) -> dict[str, Any]:
    rows = load_matrix_rows(matrix_path)
    summaries = []
    for name in (ref_case, case):
        row = find_case_row(rows, case=name, repeat_index=1)
        summary = row.get("graph_summary")
        if isinstance(summary, dict):
            summaries.append(summary)
    return merge_graph_summaries(summaries)


def write_summary(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    matrix_path: Path,
    ref_trace: Path,
    case_trace: Path,
    harness: Path,
    classification: Path,
    outcomes: Path,
) -> dict[str, Any]:
    outcome_data = load_json(outcomes)
    hard_fails = [
        row for row in outcome_data.get("rows", []) if row.get("outcome") == "hard_fail"
    ]
    graph_summary = graph_summary_for_cases(
        matrix_path,
        ref_case=args.ref_case,
        case=args.case,
    )
    graph_gate = evaluate_graph_coverage_gate(
        case="ddt_cudagraph_probe" if args.require_graph_coverage else args.case,
        graph_summary=graph_summary,
        min_graph_replay_delta=args.min_graph_replay_delta,
    )
    summary = {
        "tree": args.tree,
        "ref_case": args.ref_case,
        "case": args.case,
        "model": args.model,
        "draft_model": args.draft_model,
        "prompt_file": args.prompt_file,
        "prompt_repeat": getattr(args, "prompt_repeat", 1),
        "max_tokens": args.max_tokens,
        "dynamic_metadata_select_path": args.dynamic_metadata_select_path,
        "device_metadata_handle": args.device_metadata_handle,
        "low_margin_threshold": args.low_margin_threshold,
        "matrix_jsonl": str(matrix_path),
        "ref_trace": str(ref_trace),
        "case_trace": str(case_trace),
        "harness": str(harness),
        "classification": str(classification),
        "outcomes": str(outcomes),
        "outcome_summary": outcome_data.get("summary", {}),
        "hard_fail_count": len(hard_fails),
        "hard_fails": hard_fails,
        "graph_summary": graph_summary,
        "graph_coverage_gate": graph_gate,
        "pass": len(hard_fails) <= args.max_hard_fails and graph_gate["pass"],
    }
    output = output_dir / "large_topology_summary.json"
    with output.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return summary


def prepare_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    output_dir = Path(args.output_dir)
    matrix_path = args.matrix_jsonl or default_matrix_path(output_dir)
    ref_trace = args.ref_trace or default_trace_path(output_dir, args.ref_case)
    case_trace = args.case_trace or default_trace_path(output_dir, args.case)
    return matrix_path, ref_trace, case_trace


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path, ref_trace, case_trace = prepare_paths(args)

    if not args.skip_benchmark:
        if matrix_path.exists():
            matrix_path.unlink()
        ref_trace.parent.mkdir(parents=True, exist_ok=True)
        case_trace.parent.mkdir(parents=True, exist_ok=True)
        run_benchmark_case(
            args=args,
            matrix_path=matrix_path,
            case=args.ref_case,
            trace_path=ref_trace,
        )
        run_benchmark_case(
            args=args,
            matrix_path=matrix_path,
            case=args.case,
            trace_path=case_trace,
        )
    else:
        missing = [
            path
            for path in (matrix_path, ref_trace, case_trace)
            if not path.exists()
        ]
        if missing:
            raise SystemExit(f"--skip-benchmark missing files: {missing}")

    replay_dir = output_dir / "replay"
    if replay_dir.exists():
        shutil.rmtree(replay_dir)
    replay_dir.mkdir(parents=True, exist_ok=True)
    harness = write_harness(args=args, matrix_path=matrix_path, output_dir=replay_dir)
    export_bundles(
        args=args,
        harness_path=harness,
        ref_trace=ref_trace,
        case_trace=case_trace,
        output_dir=replay_dir,
    )
    classification, outcomes = classify_and_summarize(
        args=args,
        output_dir=replay_dir,
    )
    summary = write_summary(
        args=args,
        output_dir=output_dir,
        matrix_path=matrix_path,
        ref_trace=ref_trace,
        case_trace=case_trace,
        harness=harness,
        classification=classification,
        outcomes=outcomes,
    )
    if args.fail_on_hard_fail and not summary["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
