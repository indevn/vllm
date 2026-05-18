# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Convert benchmark-matrix outputs into correctness-harness JSON.

The standalone ``benchmark_tree_attn_ddt.py`` path is useful for local
topology/performance experiments, while the controlled-replay tooling expects
``tree_correctness_harness.py`` output.  This adapter keeps those paths joined
without teaching the replay classifier about benchmark JSONL internals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-jsonl", required=True)
    parser.add_argument("--ref-case", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat-index", type=int, default=1)
    return parser.parse_args()


def load_matrix_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def find_case_row(
    rows: list[dict[str, Any]],
    *,
    case: str,
    repeat_index: int,
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row.get("case") == case
        and row.get("repeat_index") == repeat_index
        and row.get("kind") != "comparison"
    ]
    if not matches:
        raise ValueError(
            f"matrix row not found: case={case!r} repeat_index={repeat_index}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"multiple matrix rows found: case={case!r} "
            f"repeat_index={repeat_index}"
        )
    row = matches[0]
    if row.get("returncode") != 0:
        raise ValueError(
            f"matrix row failed: case={case!r} returncode={row.get('returncode')}"
        )
    return row


def first_token_diff(lhs: list[int], rhs: list[int]) -> int | None:
    limit = min(len(lhs), len(rhs))
    for idx in range(limit):
        if lhs[idx] != rhs[idx]:
            return idx
    if len(lhs) != len(rhs):
        return limit
    return None


def steady_state(row: dict[str, Any]) -> dict[str, Any]:
    return (row.get("result") or {}).get("steady_state") or {}


def case_outputs(row: dict[str, Any]) -> list[dict[str, Any]]:
    result = row.get("result") or {}
    prompt_ids = [str(prompt_id) for prompt_id in result.get("prompt_ids") or []]
    steady = steady_state(row)
    texts = [str(text) for text in steady.get("texts") or []]
    token_rows = [
        [int(token_id) for token_id in token_row]
        for token_row in steady.get("token_ids") or []
    ]
    if not prompt_ids:
        prompt_ids = [f"prompt_{idx}" for idx in range(len(texts))]
    if len(prompt_ids) != len(texts) or len(texts) != len(token_rows):
        raise ValueError(
            "steady-state prompt/text/token row counts do not match: "
            f"prompts={len(prompt_ids)} texts={len(texts)} tokens={len(token_rows)}"
        )
    return [
        {
            "id": prompt_id,
            "text": text,
            "token_ids": token_ids,
        }
        for prompt_id, text, token_ids in zip(prompt_ids, texts, token_rows)
    ]


def build_harness(
    *,
    ref_case: str,
    case: str,
    ref_row: dict[str, Any],
    case_row: dict[str, Any],
) -> dict[str, Any]:
    ref_outputs = case_outputs(ref_row)
    outputs = case_outputs(case_row)
    if len(ref_outputs) != len(outputs):
        raise ValueError(
            f"output row count differs: ref={len(ref_outputs)} case={len(outputs)}"
        )

    diffs = []
    for ref_output, output in zip(ref_outputs, outputs):
        if ref_output["id"] != output["id"]:
            raise ValueError(
                f"prompt id differs: ref={ref_output['id']} case={output['id']}"
            )
        diff_idx = first_token_diff(
            ref_output["token_ids"], output["token_ids"]
        )
        diffs.append(
            {
                "id": ref_output["id"],
                "text_match": ref_output["text"] == output["text"],
                "token_match": diff_idx is None,
                "first_token_diff": diff_idx,
                "baseline_text": ref_output["text"],
                "case_text": output["text"],
                "baseline_token_ids": ref_output["token_ids"],
                "case_token_ids": output["token_ids"],
            }
        )

    result = case_row.get("result") or {}
    ref_result = ref_row.get("result") or {}
    return {
        "model": result.get("model") or ref_result.get("model"),
        "max_tokens": result.get("max_tokens") or ref_result.get("max_tokens"),
        "temperature": 0.0,
        "concurrency": 1,
        "prompts": [
            {"id": output["id"], "prompt": None}
            for output in ref_outputs
        ],
        "cases": {
            ref_case: {
                "base_url": None,
                "outputs": ref_outputs,
                "trace": [],
            },
            case: {
                "base_url": None,
                "outputs": outputs,
                "trace": [],
            },
        },
        "comparisons": {
            case: {
                "baseline": ref_case,
                "all_text_match": all(diff["text_match"] for diff in diffs),
                "all_token_match": all(diff["token_match"] for diff in diffs),
                "diffs": diffs,
            },
        },
    }


def main() -> None:
    args = parse_args()
    rows = load_matrix_rows(args.matrix_jsonl)
    ref_row = find_case_row(
        rows,
        case=args.ref_case,
        repeat_index=args.repeat_index,
    )
    case_row = find_case_row(
        rows,
        case=args.case,
        repeat_index=args.repeat_index,
    )
    harness = build_harness(
        ref_case=args.ref_case,
        case=args.case,
        ref_row=ref_row,
        case_row=case_row,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(harness, f, indent=2, ensure_ascii=False, sort_keys=True)
    comparison = harness["comparisons"][args.case]
    print(
        f"{args.case}: text_match={comparison['all_text_match']} "
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
