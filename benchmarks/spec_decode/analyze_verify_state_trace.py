# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Analyze speculative verify state traces against harness outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--harness-output", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_jsonl(path: str) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def best_trace_offset(case_tokens: list[int], trace_outputs: list[int]) -> int:
    best_len = -1
    best_offset = 0
    for offset in range(len(case_tokens) + 1):
        matched = 0
        while (
            offset + matched < len(case_tokens)
            and matched < len(trace_outputs)
            and case_tokens[offset + matched] == trace_outputs[matched]
        ):
            matched += 1
        if matched > best_len:
            best_len = matched
            best_offset = offset
    return best_offset


def summarize_prompt(
    *,
    prompt_id: str,
    baseline_tokens: list[int],
    case_tokens: list[int],
    first_token_diff: int | None,
    traces: list[dict[str, Any]],
) -> dict[str, Any]:
    trace_outputs: list[int] = []
    for record in traces:
        trace_outputs.extend(int(token_id) for token_id in record["output_token_ids"])
    trace_offset = best_trace_offset(case_tokens, trace_outputs)

    rows = []
    emitted = 0
    first_diff_row = None
    for row_idx, record in enumerate(traces):
        row_output = [int(token_id) for token_id in record["output_token_ids"]]
        row_start = trace_offset + emitted
        row_end = row_start + len(row_output)
        emitted += len(row_output)
        row_summary = {
            "row": row_idx,
            "row_span": [row_start, row_end],
            "output_token_ids": row_output,
            "input_ids": record.get("input_ids"),
            "positions": record.get("positions"),
            "slot_mapping": record.get("slot_mapping"),
            "seq_lens": record.get("seq_lens"),
            "attn_num_actual_tokens": record.get("attn_num_actual_tokens"),
            "attn_max_query_len": record.get("attn_max_query_len"),
            "attn_max_seq_len": record.get("attn_max_seq_len"),
            "query_start_loc": record.get("query_start_loc"),
            "query_start": record.get("query_start"),
            "query_end": record.get("query_end"),
            "block_table": record.get("block_table"),
            "logits_indices": record.get("logits_indices"),
            "logits_argmax_token_ids": record.get("logits_argmax_token_ids"),
            "logits_top_token_ids": record.get("logits_top_token_ids"),
            "logits_top_values": record.get("logits_top_values"),
            "logits_top_margins": record.get("logits_top_margins"),
            "scheduled_spec_decode_tokens": record.get(
                "scheduled_spec_decode_tokens"
            ),
            "num_draft_tokens": record.get("num_draft_tokens"),
            "accepted_prefix_len": record.get("accepted_prefix_len"),
            "first_reject_index": record.get("first_reject_index"),
            "draft_token_ids": record.get("draft_token_ids"),
            "draft_token_ids_flat": record.get("draft_token_ids_flat"),
            "target_logits_indices": record.get("target_logits_indices"),
            "bonus_logits_indices": record.get("bonus_logits_indices"),
            "tree_target_mask": record.get("tree_target_mask"),
            "tree_position_offsets": record.get("tree_position_offsets"),
            "tree_attn_bias_mask": record.get("tree_attn_bias_mask"),
            "target_argmax_token_ids": record.get("target_argmax_token_ids"),
            "num_computed_tokens_cpu": record.get("num_computed_tokens_cpu"),
            "num_prompt_tokens": record.get("num_prompt_tokens"),
            "num_tokens_no_spec": record.get("num_tokens_no_spec"),
            "token_ids_cpu_tail": record.get("token_ids_cpu_tail"),
        }
        if (
            first_token_diff is not None
            and row_start <= first_token_diff < row_end
            and first_diff_row is None
        ):
            row_summary["contains_first_diff"] = True
            row_summary["local_output_index"] = first_token_diff - row_start
            first_diff_row = row_summary
        rows.append(row_summary)

    return {
        "id": prompt_id,
        "first_token_diff": first_token_diff,
        "baseline_token": None
        if first_token_diff is None
        else baseline_tokens[first_token_diff],
        "case_token": None
        if first_token_diff is None
        else case_tokens[first_token_diff],
        "trace_offset": trace_offset,
        "first_diff_row": first_diff_row,
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    with open(args.harness_output, encoding="utf-8") as f:
        harness = json.load(f)
    traces = load_jsonl(args.trace)
    comparison = harness["comparisons"][args.case]

    trace_groups: list[list[dict[str, Any]]] = []
    for record in traces:
        if not trace_groups or record.get("request_id") != trace_groups[-1][-1].get(
            "request_id"
        ):
            trace_groups.append([record])
        else:
            trace_groups[-1].append(record)
    trace_by_prompt: dict[str, list[dict[str, Any]]] = {}
    for diff in comparison["diffs"]:
        prompt_id = diff["id"]
        trace_by_prompt[prompt_id] = (
            trace_groups[len(trace_by_prompt)]
            if len(trace_by_prompt) < len(trace_groups)
            else []
        )

    summaries = []
    for diff in comparison["diffs"]:
        summaries.append(
            summarize_prompt(
                prompt_id=diff["id"],
                baseline_tokens=diff["baseline_token_ids"],
                case_tokens=diff["case_token_ids"],
                first_token_diff=diff["first_token_diff"],
                traces=trace_by_prompt.get(diff["id"], []),
            )
        )

    output = {
        "case": args.case,
        "baseline": comparison["baseline"],
        "all_token_match": comparison["all_token_match"],
        "summaries": summaries,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, sort_keys=True)

    print(f"{args.case}: token_match={comparison['all_token_match']}")
    for summary in summaries:
        if summary["first_token_diff"] is not None:
            row = summary["first_diff_row"]
            row_idx = None if row is None else row["row"]
            print(
                f"  {summary['id']}: first_token_diff="
                f"{summary['first_token_diff']} row={row_idx}"
            )


if __name__ == "__main__":
    main()
