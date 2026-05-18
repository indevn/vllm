# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Export controlled replay bundles from correctness harness and state traces."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.spec_decode.analyze_verify_state_trace import (  # noqa: E402
    best_trace_offset,
    group_traces_by_request,
    match_trace_groups_to_prompts,
    token_trace_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness-output", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--drafter-model", default="RedHatAI/Qwen3-8B-speculator.eagle3"
    )
    parser.add_argument("--attention-backend", default="TREE_ATTN")
    parser.add_argument("--token-source", choices=("baseline", "case"), default="case")
    parser.add_argument("--case-prefix", default="")
    return parser.parse_args()


def load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def emitted_spans(
    trace_tokens: list[int],
    traces: list[dict[str, Any]],
) -> list[tuple[int, int, dict[str, Any]]]:
    traces = token_trace_records(traces)
    trace_outputs: list[int] = []
    for record in traces:
        trace_outputs.extend(int(token_id) for token_id in record["output_token_ids"])
    trace_offset = best_trace_offset(trace_tokens, trace_outputs)
    spans = []
    emitted = 0
    for record in traces:
        output = [int(token_id) for token_id in record["output_token_ids"]]
        start = trace_offset + emitted
        end = start + len(output)
        emitted += len(output)
        spans.append((start, end, record))
    return spans


def find_first_diff_record(
    first_token_diff: int | None,
    trace_tokens: list[int],
    traces: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if first_token_diff is None:
        return None
    for start, end, record in emitted_spans(trace_tokens, traces):
        if start <= first_token_diff < end:
            bundle_record = dict(record)
            bundle_record["row_span"] = [start, end]
            bundle_record["local_output_index"] = first_token_diff - start
            return bundle_record
    return None


def first_non_empty(records: list[dict[str, Any]], key: str) -> Any:
    for record in records:
        value = record.get(key)
        if value is not None:
            return value
    return None


def relocation_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    total_rows = 0
    by_kind: dict[str, int] = defaultdict(int)
    examples = []
    for record in records:
        pairs = record.get("tree_relocation_pairs") or []
        if not pairs:
            continue
        total_rows += 1
        for pair in pairs:
            key = f"{pair.get('src_local')}->{pair.get('dst_local')}"
            by_kind[key] += 1
        if len(examples) < 5:
            examples.append(
                {
                    "token_ids_cpu_tail": record.get("token_ids_cpu_tail"),
                    "input_ids": record.get("input_ids"),
                    "positions": record.get("positions"),
                    "slot_mapping": record.get("slot_mapping"),
                    "accept_indices": record.get("accept_indices"),
                    "output_token_ids": record.get("output_token_ids"),
                    "tree_relocation_pairs": pairs,
                }
            )
    return {
        "rows_with_relocation": total_rows,
        "relocation_kinds": dict(sorted(by_kind.items())),
        "examples": examples,
    }


def case_id(prefix: str, mode: str, prompt_id: str) -> str:
    raw = f"{prefix}-{mode}-{prompt_id}" if prefix else f"{mode}-{prompt_id}"
    return raw.replace("_", "-").replace("/", "-")


def build_bundle(
    *,
    args: argparse.Namespace,
    harness: dict[str, Any],
    diff: dict[str, Any],
    request_id: str | None,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    token_source_key = f"{args.token_source}_token_ids"
    trace_tokens = [int(token_id) for token_id in diff[token_source_key]]
    first_diff_record = find_first_diff_record(
        diff["first_token_diff"],
        trace_tokens,
        records,
    )
    first_diff = diff["first_token_diff"]
    window_start = max(0, first_diff - 8) if first_diff is not None else 0
    window_end = first_diff + 8 if first_diff is not None else 0
    baseline_tokens = [int(token_id) for token_id in diff["baseline_token_ids"]]
    case_tokens = [int(token_id) for token_id in diff["case_token_ids"]]

    return {
        "case_id": case_id(args.case_prefix, args.mode, diff["id"]),
        "mode": args.mode,
        "prompt_id": diff["id"],
        "request_id": request_id,
        "model": args.model,
        "drafter_model": args.drafter_model,
        "attention_backend": args.attention_backend,
        "harness_output": args.harness_output,
        "trace": args.trace,
        "harness_case": args.case,
        "max_tokens": harness.get("max_tokens"),
        "temperature": harness.get("temperature"),
        "concurrency": harness.get("concurrency"),
        "first_token_diff": first_diff,
        "baseline_token": None if first_diff is None else baseline_tokens[first_diff],
        "case_token": None if first_diff is None else case_tokens[first_diff],
        "baseline_token_window": baseline_tokens[window_start:window_end],
        "case_token_window": case_tokens[window_start:window_end],
        "first_diff_record": first_diff_record,
        "records_before_first_diff": [
            record
            for start, _, record in emitted_spans(trace_tokens, records)
            if first_diff is not None and start <= first_diff
        ],
        "relocation_summary": relocation_summary(records),
        "tree_attn_bias_mask": None
        if first_diff_record is None
        else first_diff_record.get("tree_attn_bias_mask"),
        "tree_parent": None
        if first_diff_record is None
        else first_diff_record.get("tree_parent"),
        "tree_target_mask": None
        if first_diff_record is None
        else first_diff_record.get("tree_target_mask"),
        "tree_target_logits_indices": None
        if first_diff_record is None
        else first_diff_record.get("tree_target_logits_indices"),
        "tree_logits_top_token_ids": None
        if first_diff_record is None
        else first_diff_record.get("tree_logits_top_token_ids"),
        "tree_logits_top_values": None
        if first_diff_record is None
        else first_diff_record.get("tree_logits_top_values"),
        "tree_logits_top_margins": None
        if first_diff_record is None
        else first_diff_record.get("tree_logits_top_margins"),
        "query_start_loc": None
        if first_diff_record is None
        else first_diff_record.get("query_start_loc"),
        "logits_indices": None
        if first_diff_record is None
        else first_diff_record.get("logits_indices"),
        "block_table": None
        if first_diff_record is None
        else first_diff_record.get("block_table"),
        "all_trace_request_ids": sorted(
            str(record.get("request_id")) for record in records
        ),
        "first_available_tree_runtime_mode": first_non_empty(
            records, "tree_runtime_mode"
        ),
    }


def main() -> None:
    args = parse_args()
    harness = load_json(args.harness_output)
    traces = load_jsonl(args.trace)
    comparison = harness["comparisons"][args.case]
    trace_groups = group_traces_by_request(traces)
    trace_by_prompt = match_trace_groups_to_prompts(
        comparison["diffs"],
        trace_groups,
        args.token_source,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported = []
    for diff in comparison["diffs"]:
        if diff["first_token_diff"] is None:
            continue
        request_id, records = trace_by_prompt.get(diff["id"], (None, []))
        bundle = build_bundle(
            args=args,
            harness=harness,
            diff=diff,
            request_id=request_id,
            records=records,
        )
        output_path = output_dir / f"{bundle['case_id']}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2, ensure_ascii=False, sort_keys=True)
        exported.append(output_path)

    print(f"exported {len(exported)} bundles")
    for path in exported:
        print(path)


if __name__ == "__main__":
    main()
