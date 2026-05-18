# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Summarize DDT replay classifications with target-only self-checks.

The controlled-replay classifier explains a first-diff row by inspecting DDT
and target-only logits.  Concurrent serving adds one more oracle question: did
the target-only service itself produce unstable output for the same prompt?
This helper keeps that policy explicit and repeatable.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

LOW_MARGIN_CLASSES = {
    "low_margin_tree_verify_candidate",
    "post_relocation_q1_low_margin_candidate",
    "q1_low_margin_candidate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification", required=True)
    parser.add_argument("--target-selfcheck-harness", action="append", default=[])
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def selfcheck_unstable_prompts(paths: list[str]) -> dict[str, list[dict[str, Any]]]:
    unstable: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        harness = load_json(path)
        for case_name, comparison in harness.get("comparisons", {}).items():
            for diff in comparison.get("diffs", []):
                if diff.get("token_match", True):
                    continue
                prompt_id = diff["id"]
                unstable.setdefault(prompt_id, []).append(
                    {
                        "harness": path,
                        "case": case_name,
                        "first_token_diff": diff.get("first_token_diff"),
                    }
                )
    return unstable


def outcome_for(row: dict[str, Any], unstable_prompts: dict[str, Any]) -> str:
    if row.get("prompt_id") in unstable_prompts:
        return "oracle_unstable"
    if row.get("classification") in LOW_MARGIN_CLASSES:
        return "explained_low_margin"
    return "hard_fail"


def main() -> None:
    args = parse_args()
    classification = load_json(args.classification)
    unstable_prompts = selfcheck_unstable_prompts(args.target_selfcheck_harness)
    rows = []
    for row in classification.get("rows", []):
        outcome = outcome_for(row, unstable_prompts)
        rows.append(
            {
                "cell": row.get("cell"),
                "prompt_id": row.get("prompt_id"),
                "first_token_diff": row.get("first_token_diff"),
                "baseline_token": row.get("baseline_token"),
                "case_token": row.get("case_token"),
                "classification": row.get("classification"),
                "outcome": outcome,
                "ddt_margin": row.get("ddt_margin"),
                "target_margin": row.get("target_margin"),
                "oracle_selfcheck": unstable_prompts.get(row.get("prompt_id"), []),
            }
        )

    summary = Counter(row["outcome"] for row in rows)
    output = {
        "classification": args.classification,
        "target_selfcheck_harnesses": args.target_selfcheck_harness,
        "oracle_unstable_prompts": sorted(unstable_prompts),
        "summary": dict(sorted(summary.items())),
        "rows": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, sort_keys=True)

    print("outcome\tcell\tprompt\tdiff\tclass")
    for row in rows:
        print(
            f"{row['outcome']}\t{row['cell']}\t{row['prompt_id']}\t"
            f"{row['first_token_diff']}\t{row['classification']}"
        )
    print("summary", dict(sorted(summary.items())))


if __name__ == "__main__":
    main()
