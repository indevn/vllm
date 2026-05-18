# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Classify DDT controlled replay bundles against target-only bundles."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-bundle-glob",
        action="append",
        required=True,
        help="Glob for speculative/DDT replay bundle JSON files.",
    )
    parser.add_argument(
        "--target-bundle-glob",
        action="append",
        required=True,
        help="Glob for target-only replay bundle JSON files.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--low-margin-threshold",
        type=float,
        default=0.25,
        help=(
            "Top-1/top-2 margin threshold used to classify near-tie rows. "
            "Both DDT and target margins must be at or below this value."
        ),
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def expand_globs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(path) for path in glob.glob(pattern))
    return sorted(set(paths))


def cell_id(path: Path, bundle: dict[str, Any]) -> str:
    # Matrix bundle directories are named fixed64, expanded32, etc.  Fall back to
    # max_tokens when analyzing ad-hoc bundles outside the matrix tree.
    if path.parent.name:
        return path.parent.name
    return f"max{bundle.get('max_tokens', 'unknown')}"


def top_at(
    record: dict[str, Any],
    *,
    local_output_index: int | None,
    accepted_local_index: int | None,
) -> tuple[str | None, list[int] | None, list[float] | None, float | None]:
    """Return the logits row that emitted the first-diff token.

    Tree verify rows emit accepted tokens by accepted local index; non-tree rows
    emit from the normal sampled logits row.
    """

    tree_top_ids = record.get("tree_logits_top_token_ids")
    tree_top_values = record.get("tree_logits_top_values")
    tree_margins = record.get("tree_logits_top_margins")
    if (
        isinstance(accepted_local_index, int)
        and accepted_local_index >= 0
        and tree_top_ids
        and accepted_local_index < len(tree_top_ids)
    ):
        return (
            "tree",
            tree_top_ids[accepted_local_index],
            tree_top_values[accepted_local_index],
            tree_margins[accepted_local_index],
        )

    top_ids = record.get("logits_top_token_ids")
    top_values = record.get("logits_top_values")
    margins = record.get("logits_top_margins")
    row_index = record.get("query_start")
    if not isinstance(row_index, int):
        row_index = 0
    if (
        isinstance(local_output_index, int)
        and local_output_index > 0
        and top_ids
        and row_index + local_output_index < len(top_ids)
    ):
        row_index += local_output_index
    if top_ids and row_index < len(top_ids):
        return (
            "direct",
            top_ids[row_index],
            top_values[row_index],
            margins[row_index],
        )
    return None, None, None, None


def ancestor_visibility_sanity(
    mask: list[list[int]] | None,
    *,
    valid_width: int | None = None,
) -> bool | None:
    if mask is None:
        return None
    padded_width = len(mask)
    if any(len(row) != padded_width for row in mask):
        return False
    width = padded_width if valid_width is None else min(valid_width, padded_width)
    if width <= 0:
        return None
    for row_idx, row in enumerate(mask[:width]):
        if row[0] != 1:
            return False
        if row_idx == 0:
            if any(row[col_idx] for col_idx in range(1, width)):
                return False
            continue
        if row[row_idx] != 1:
            return False
        if any(row[col_idx] for col_idx in range(row_idx + 1, width)):
            return False
    return True


def relocation_counts(records: list[dict[str, Any]]) -> tuple[int, int, int]:
    pairs = 0
    nonprefix_pairs = 0
    rows = 0
    for record in records:
        row_pairs = record.get("tree_relocation_pairs") or []
        if not row_pairs:
            continue
        rows += 1
        pairs += len(row_pairs)
        nonprefix_pairs += sum(
            1
            for pair in row_pairs
            if pair.get("src_local") != pair.get("dst_local")
        )
    return rows, pairs, nonprefix_pairs


def contains_both(tokens: list[int] | None, lhs: int, rhs: int) -> bool:
    return tokens is not None and lhs in tokens[:5] and rhs in tokens[:5]


def comparable_tail(record: dict[str, Any]) -> list[int] | None:
    tail = record.get("token_ids_cpu_tail")
    if tail is None:
        return None
    # Standalone target-only traces can expose placeholder-filled tails before
    # the sampled output IDs have been committed back to the CPU token table.
    # Treat these as unavailable evidence instead of a real history mismatch.
    if any(int(token_id) < 0 for token_id in tail):
        return None
    local_output_index = record.get("local_output_index")
    output_token_ids = record.get("output_token_ids") or []
    if isinstance(local_output_index, int) and local_output_index > 0:
        tail = [*tail, *output_token_ids[:local_output_index]]
    return tail[-32:]


def tails_match(
    case_record: dict[str, Any],
    target_record: dict[str, Any],
) -> bool | None:
    case_tail = comparable_tail(case_record)
    target_tail = comparable_tail(target_record)
    if case_tail is None or target_tail is None:
        return None
    compare_width = min(len(case_tail), len(target_tail))
    return case_tail[-compare_width:] == target_tail[-compare_width:]


def is_low_margin(value: float | None, threshold: float) -> bool:
    return value is not None and value <= threshold


def classify(row: dict[str, Any], low_margin_threshold: float = 0.25) -> str:
    if row["mask_ok"] is False:
        return "hard_fail_mask_visibility"
    if row["tails_equal"] is False:
        return "hard_fail_history_mismatch"
    if not row["ddt_top_contains_tokens"] or not row["target_top_contains_tokens"]:
        return "needs_investigation_non_top2"
    if is_low_margin(row["ddt_margin"], low_margin_threshold) and is_low_margin(
        row["target_margin"], low_margin_threshold
    ):
        if (
            row["has_tree_metadata"]
            and row["logits_source"] == "tree"
            and (row["num_draft_tokens"] or 0) > 0
        ):
            return "low_margin_tree_verify_candidate"
        if row["nonprefix_relocation_pairs_before"] > 0:
            return "post_relocation_q1_low_margin_candidate"
        return "q1_low_margin_candidate"
    return "needs_investigation_margin"


def classify_bundle(
    path: Path,
    bundle: dict[str, Any],
    target_bundle: dict[str, Any] | None,
    low_margin_threshold: float = 0.25,
) -> dict[str, Any]:
    record = bundle.get("first_diff_record") or {}
    target_record = (
        None if target_bundle is None else target_bundle.get("first_diff_record")
    ) or {}
    local_output_index = record.get("local_output_index")
    accept_indices = record.get("accept_indices") or []
    accepted_local_index = None
    if isinstance(local_output_index, int) and local_output_index < len(accept_indices):
        accepted_local_index = accept_indices[local_output_index]
    valid_width = None
    if isinstance(record.get("num_draft_tokens"), int):
        valid_width = int(record["num_draft_tokens"]) + 1
        if valid_width <= 1:
            accepted_local_index = None

    ddt_source, ddt_top_ids, ddt_top_values, ddt_margin = top_at(
        record,
        local_output_index=local_output_index,
        accepted_local_index=accepted_local_index,
    )
    target_source, target_top_ids, target_top_values, target_margin = top_at(
        target_record,
        local_output_index=target_record.get("local_output_index"),
        accepted_local_index=None,
    )
    before = bundle.get("records_before_first_diff") or []
    relocation_rows, relocation_pairs, nonprefix_pairs = relocation_counts(before)
    diff_row_pairs = len(record.get("tree_relocation_pairs") or [])
    baseline_token = bundle.get("baseline_token")
    case_token = bundle.get("case_token")
    row = {
        "cell": cell_id(path, bundle),
        "bundle": str(path),
        "target_bundle": None
        if target_bundle is None
        else target_bundle.get("case_id"),
        "prompt_id": bundle.get("prompt_id"),
        "first_token_diff": bundle.get("first_token_diff"),
        "baseline_token": baseline_token,
        "case_token": case_token,
        "has_tree_metadata": record.get("has_tree_metadata"),
        "num_draft_tokens": record.get("num_draft_tokens"),
        "logits_source": ddt_source,
        "local_output_index": local_output_index,
        "accepted_local_index": accepted_local_index,
        "ddt_top_token_ids": ddt_top_ids,
        "ddt_top_values": ddt_top_values,
        "ddt_margin": ddt_margin,
        "target_logits_source": target_source,
        "target_top_token_ids": target_top_ids,
        "target_top_values": target_top_values,
        "target_margin": target_margin,
        "ddt_top_contains_tokens": contains_both(
            ddt_top_ids, baseline_token, case_token
        )
        if baseline_token is not None and case_token is not None
        else False,
        "target_top_contains_tokens": contains_both(
            target_top_ids, baseline_token, case_token
        )
        if baseline_token is not None and case_token is not None
        else False,
        "mask_ok": ancestor_visibility_sanity(
            record.get("tree_attn_bias_mask"),
            valid_width=valid_width,
        ),
        "target_mask": record.get("tree_target_mask"),
        "relocation_rows_before": relocation_rows,
        "relocation_pairs_before": relocation_pairs,
        "nonprefix_relocation_pairs_before": nonprefix_pairs,
        "diff_row_relocation_pairs": diff_row_pairs,
        "tails_equal": tails_match(record, target_record)
        if record and target_record
        else None,
        "input_ids": record.get("input_ids"),
        "positions": record.get("positions"),
        "slot_mapping": record.get("slot_mapping"),
        "target_input_ids": target_record.get("input_ids"),
        "target_positions": target_record.get("positions"),
        "target_slot_mapping": target_record.get("slot_mapping"),
    }
    row["classification"] = classify(row, low_margin_threshold)
    return row


def main() -> None:
    args = parse_args()
    case_paths = expand_globs(args.case_bundle_glob)
    target_paths = expand_globs(args.target_bundle_glob)
    target_by_cell_prompt: dict[tuple[str, str], dict[str, Any]] = {}
    for path in target_paths:
        bundle = load_json(path)
        target_by_cell_prompt[(cell_id(path, bundle), bundle["prompt_id"])] = bundle

    rows = []
    for path in case_paths:
        bundle = load_json(path)
        target = target_by_cell_prompt.get((cell_id(path, bundle), bundle["prompt_id"]))
        rows.append(
            classify_bundle(
                path,
                bundle,
                target,
                low_margin_threshold=args.low_margin_threshold,
            )
        )

    output = {"low_margin_threshold": args.low_margin_threshold, "rows": rows}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, sort_keys=True)

    print(
        "cell\tprompt\tdiff\tbase->case\tsource\tmargin\t"
        "target_margin\trelocation\tclass"
    )
    for row in rows:
        relocation = (
            f"{row['relocation_pairs_before']}/"
            f"{row['nonprefix_relocation_pairs_before']}/"
            f"{row['diff_row_relocation_pairs']}"
        )
        print(
            f"{row['cell']}\t{row['prompt_id']}\t{row['first_token_diff']}\t"
            f"{row['baseline_token']}->{row['case_token']}\t"
            f"{row['logits_source']}\t{row['ddt_margin']}\t"
            f"{row['target_margin']}\t{relocation}\t"
            f"{row['classification']}"
        )


if __name__ == "__main__":
    main()
