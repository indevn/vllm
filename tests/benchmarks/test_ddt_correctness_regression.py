# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
from argparse import Namespace

from benchmarks.spec_decode.analyze_verify_state_trace import (
    group_traces_by_request,
    match_trace_groups_to_prompts,
    trace_match_score,
)
from benchmarks.spec_decode.benchmark_matrix_to_harness import (
    build_harness as build_benchmark_harness,
)
from benchmarks.spec_decode.benchmark_tree_attn_ddt import (
    _trace_acceptance_from_records,
    _trace_graph_summary_from_records,
    _trace_stage_profile_from_records,
    build_llm,
    generated_token_ids,
    load_prompt_ids,
    load_prompts,
    parse_tree_arg,
    tree_num_nodes,
)
from benchmarks.spec_decode.classify_replay_bundles import classify, tails_match
from benchmarks.spec_decode.run_ddt_correctness_regression import (
    graph_coverage_gate as regression_graph_coverage_gate,
)
from benchmarks.spec_decode.run_ddt_correctness_regression import (
    iteration_output_dir,
    merge_graph_summaries,
    suite_cells,
    write_repeat_summary,
)
from benchmarks.spec_decode.run_ddt_large_topology_correctness import (
    benchmark_args as large_topology_benchmark_args,
)
from benchmarks.spec_decode.run_ddt_large_topology_correctness import (
    default_trace_path as large_topology_default_trace_path,
)
from benchmarks.spec_decode.run_ddt_large_topology_correctness import (
    graph_summary_for_cases as large_topology_graph_summary_for_cases,
)
from benchmarks.spec_decode.run_ddt_large_topology_correctness import (
    prepare_paths as large_topology_prepare_paths,
)
from benchmarks.spec_decode.run_ddt_large_topology_correctness import (
    write_summary as write_large_topology_summary,
)
from benchmarks.spec_decode.run_tree_attn_ddt_benchmark_matrix import (
    command_for_case,
    compare_case_results,
    compare_repeat_stability,
    compare_speedup_results,
    evaluate_acceptance_gate,
    evaluate_graph_coverage_gate,
    evaluate_sdt_baseline_gate,
    load_correctness_gate,
    parse_compare_cases,
    summarize_repeat_rows,
)
from benchmarks.spec_decode.run_tree_attn_ddt_benchmark_matrix import (
    merge_graph_summaries as merge_benchmark_graph_summaries,
)
from benchmarks.spec_decode.summarize_replay_outcomes import outcome_for


def test_ddt_regression_suite_cells():
    args = Namespace(
        suite="full",
        prompts="fixed.jsonl",
        expanded_prompts="expanded.jsonl",
        cell=[],
    )

    cells = suite_cells(args)

    assert [(cell.name, cell.prompts, cell.max_tokens, cell.concurrency)
            for cell in cells] == [
                ("fixed32", "fixed.jsonl", 32, 1),
                ("fixed32_conc2", "fixed.jsonl", 32, 2),
                ("expanded64", "expanded.jsonl", 64, 1),
                ("expanded64_conc2", "expanded.jsonl", 64, 2),
                ("fixed32_conc4", "fixed.jsonl", 32, 4),
                ("expanded128", "expanded.jsonl", 128, 1),
                ("expanded128_conc2", "expanded.jsonl", 128, 2),
            ]


def test_benchmark_tree_presets_expand_to_expected_width():
    assert tree_num_nodes("binary30") == 30
    assert tree_num_nodes("binary62") == 62
    assert parse_tree_arg("binary30").startswith("[(0,), (1,),")


def test_ddt_regression_suite_filters_cells():
    args = Namespace(
        suite="full",
        prompts="fixed.jsonl",
        expanded_prompts="expanded.jsonl",
        cell=["expanded128", "fixed32_conc4"],
    )

    cells = suite_cells(args)

    assert [cell.name for cell in cells] == ["fixed32_conc4", "expanded128"]


def test_ddt_regression_suite_accepts_optional_long_cells():
    args = Namespace(
        suite="full",
        prompts="fixed.jsonl",
        expanded_prompts="expanded.jsonl",
        cell=["fixed128", "expanded256"],
    )

    cells = suite_cells(args)

    assert [(cell.name, cell.max_tokens, cell.concurrency) for cell in cells] == [
        ("fixed128", 128, 1),
        ("expanded256", 256, 1),
    ]


def test_ddt_regression_suite_full_includes_expanded128_conc2():
    args = Namespace(
        suite="full",
        prompts="fixed.jsonl",
        expanded_prompts="expanded.jsonl",
        cell=[],
    )

    cells = suite_cells(args)

    assert [cell.name for cell in cells] == [
        "fixed32",
        "fixed32_conc2",
        "expanded64",
        "expanded64_conc2",
        "fixed32_conc4",
        "expanded128",
        "expanded128_conc2",
    ]


def test_ddt_regression_repeat_uses_subdirectories(tmp_path):
    base = tmp_path / "out"

    assert iteration_output_dir(base, repeat=1, repeat_index=1) == base
    assert iteration_output_dir(base, repeat=2, repeat_index=1) == base / "repeat_01"
    assert iteration_output_dir(base, repeat=2, repeat_index=2) == base / "repeat_02"


def test_ddt_regression_repeat_summary_aggregates_runs(tmp_path):
    args = Namespace(
        suite="full",
        repeat=2,
        target_url="http://target",
        ddt_url="http://ddt",
        low_margin_threshold=0.5,
    )

    summary = write_repeat_summary(
        args=args,
        output_dir=tmp_path,
        summaries=[
            {
                "pass": True,
                "hard_fail_count": 0,
                "outcome_summary": {"explained_low_margin": 31},
            },
            {
                "pass": True,
                "hard_fail_count": 0,
                "outcome_summary": {"explained_low_margin": 28},
            },
        ],
    )

    assert summary["pass"]
    assert summary["hard_fail_count"] == 0
    assert summary["graph_summary"]["graph_records"] == 0
    assert [run["output_dir"] for run in summary["runs"]] == [
        str(tmp_path / "repeat_01"),
        str(tmp_path / "repeat_02"),
    ]


def test_outcome_policy_marks_oracle_unstable_before_low_margin():
    row = {
        "prompt_id": "software_short",
        "classification": "low_margin_tree_verify_candidate",
    }

    outcome = outcome_for(row, {"software_short": [{"first_token_diff": 22}]})

    assert outcome == "oracle_unstable"


def test_outcome_policy_marks_unexplained_rows_as_hard_fail():
    row = {
        "prompt_id": "prompt",
        "classification": "needs_investigation_margin",
    }

    outcome = outcome_for(row, {})

    assert outcome == "hard_fail"


def test_replay_classifier_low_margin_threshold_is_explicit():
    row = {
        "mask_ok": True,
        "tails_equal": True,
        "ddt_top_contains_tokens": True,
        "target_top_contains_tokens": True,
        "ddt_margin": 0.375,
        "target_margin": 0.25,
        "has_tree_metadata": True,
        "logits_source": "direct",
        "num_draft_tokens": 0,
        "nonprefix_relocation_pairs_before": 2,
    }

    assert classify(row, low_margin_threshold=0.25) == "needs_investigation_margin"
    assert (
        classify(row, low_margin_threshold=0.5)
        == "post_relocation_q1_low_margin_candidate"
    )


def test_replay_classifier_no_draft_direct_rows_are_not_tree_verify():
    row = {
        "mask_ok": True,
        "tails_equal": True,
        "ddt_top_contains_tokens": True,
        "target_top_contains_tokens": True,
        "ddt_margin": 0.0,
        "target_margin": 0.0,
        "has_tree_metadata": True,
        "logits_source": "direct",
        "num_draft_tokens": 0,
        "nonprefix_relocation_pairs_before": 0,
    }

    assert classify(row, low_margin_threshold=0.25) == "q1_low_margin_candidate"


def test_replay_classifier_ignores_placeholder_target_tails():
    assert tails_match(
        {
            "token_ids_cpu_tail": [11, 12, 13],
            "local_output_index": 0,
            "output_token_ids": [14],
        },
        {
            "token_ids_cpu_tail": [-1, -1, -1],
            "local_output_index": 0,
            "output_token_ids": [14],
        },
    ) is None


def test_verify_state_trace_matching_ignores_stage_profile_records():
    records = [
        {
            "trace_kind": "tree_attn_stage_profile",
            "request_id": "req-a",
            "tree_cudagraph_runtime": {"mode": "PIECEWISE"},
        },
        {
            "trace_kind": "spec_verify_state",
            "request_id": "req-a",
            "output_token_ids": [11, 12],
        },
        {
            "trace_kind": "spec_verify_state",
            "request_id": "req-a",
            "output_token_ids": [13],
        },
    ]

    assert trace_match_score([11, 12, 13], records) == 3
    groups = group_traces_by_request(records)
    assert len(groups) == 1
    assert len(groups[0][1]) == 2

    matched = match_trace_groups_to_prompts(
        [
            {
                "id": "prompt",
                "case_token_ids": [11, 12, 13],
            }
        ],
        groups,
        "case",
    )

    assert matched["prompt"][0] == "req-a"


def test_ddt_benchmark_matrix_adds_cudagraph_probe_config():
    args = Namespace(
        benchmark_arg=["--max-tokens", "8"],
        cudagraph_compilation_config='{"cudagraph_mode":"piecewise"}',
        near_tie_repair_scope="fallback_or_nonprefix",
    )

    cmd = command_for_case(args, "ddt_cudagraph_probe")

    assert "--compilation-config" in cmd
    assert '{"cudagraph_mode":"piecewise"}' in cmd


def test_ddt_benchmark_matrix_defaults_near_tie_scope():
    args = Namespace(
        benchmark_arg=["--max-tokens", "8"],
        cudagraph_compilation_config='{"cudagraph_mode":"piecewise"}',
        near_tie_repair_scope="fallback_or_nonprefix",
    )

    cmd = command_for_case(args, "ddt_near_tie_repair")

    assert cmd[-2:] == ["--serial-repair-scope", "fallback_or_nonprefix"]


def test_ddt_benchmark_matrix_defaults_static_repair_scope():
    args = Namespace(
        benchmark_arg=["--max-tokens", "8"],
        cudagraph_compilation_config='{"cudagraph_mode":"piecewise"}',
        near_tie_repair_scope="fallback_or_nonprefix",
    )

    cmd = command_for_case(args, "tree_static_repair")

    assert cmd[:4] == [
        os.sys.executable,
        "benchmarks/spec_decode/benchmark_tree_attn_ddt.py",
        "--case",
        "tree_static_repair",
    ]
    assert cmd[-2:] == ["--serial-repair-scope", "fallback_or_nonprefix"]


def test_ddt_benchmark_matrix_static_baseline_enables_relocation():
    args = Namespace(
        benchmark_arg=["--max-tokens", "8"],
        cudagraph_compilation_config='{"cudagraph_mode":"piecewise"}',
        near_tie_repair_scope="fallback_or_nonprefix",
    )

    cmd = command_for_case(args, "tree_static")

    assert "--enable-static-kv-relocation" in cmd


def test_ddt_benchmark_matrix_respects_explicit_static_relocation():
    args = Namespace(
        benchmark_arg=["--enable-static-kv-relocation"],
        cudagraph_compilation_config='{"cudagraph_mode":"piecewise"}',
        near_tie_repair_scope="fallback_or_nonprefix",
    )

    cmd = command_for_case(args, "tree_static")

    assert cmd.count("--enable-static-kv-relocation") == 1


def test_ddt_benchmark_matrix_respects_explicit_near_tie_scope():
    args = Namespace(
        benchmark_arg=[
            "--serial-repair-scope",
            "all",
        ],
        cudagraph_compilation_config='{"cudagraph_mode":"piecewise"}',
        near_tie_repair_scope="fallback_or_nonprefix",
    )

    cmd = command_for_case(args, "ddt_near_tie_repair")

    assert cmd.count("--serial-repair-scope") == 1
    assert cmd[-2:] == ["--serial-repair-scope", "all"]


def test_ddt_benchmark_matrix_correctness_gate(tmp_path):
    summary_file = tmp_path / "regression_repeat_summary.json"
    summary_file.write_text(
        '{"pass": true, "hard_fail_count": 0, "repeat": 2}\n',
        encoding="utf-8",
    )

    gate = load_correctness_gate(summary_file, max_hard_fails=0)

    assert gate == {
        "path": str(summary_file),
        "pass": True,
        "hard_fail_count": 0,
        "outcome_summary": None,
        "repeat": 2,
    }


def test_ddt_benchmark_matrix_correctness_gate_rejects_fail(tmp_path):
    summary_file = tmp_path / "regression_summary.json"
    summary_file.write_text(
        '{"pass": false, "hard_fail_count": 1}\n',
        encoding="utf-8",
    )

    try:
        load_correctness_gate(summary_file, max_hard_fails=0)
    except SystemExit as exc:
        assert "correctness gate failed" in str(exc)
    else:
        raise AssertionError("expected correctness gate failure")


def _benchmark_row(
    texts: list[str],
    accepted: int = 3,
    graph_summary: dict | None = None,
    token_ids: list[list[int]] | None = None,
    prompt_ids: list[str] | None = None,
    *,
    case: str = "ddt_mask_kernel",
    canonical_case: str | None = None,
    static_kv_relocation: bool | None = None,
    throughput_tok_s: float = 10.0,
    num_drafts: int = 2,
    num_draft_tokens: int = 6,
) -> dict:
    token_ids = token_ids or [[idx + 1] for idx, _ in enumerate(texts)]
    prompt_ids = prompt_ids or [f"prompt_{idx}" for idx, _ in enumerate(texts)]
    return {
        "returncode": 0,
        "result": {
            "case": case,
            "canonical_case": canonical_case or case,
            "static_kv_relocation": static_kv_relocation,
            "prompt_ids": prompt_ids,
            "steady_state": {
                "throughput_tok_s": throughput_tok_s,
                "texts": texts,
                "token_ids": token_ids,
                "spec_decode": {
                    "num_drafts": num_drafts,
                    "num_draft_tokens": num_draft_tokens,
                    "num_accepted_tokens": accepted,
                    "acceptance_rate": accepted / num_draft_tokens
                    if num_draft_tokens
                    else None,
                    "acceptance_length": 1 + accepted / num_drafts
                    if num_drafts
                    else None,
                    "per_position_acceptance_rates": [1.0, 0.5, 0.0],
                },
                "trace_graph_summary": graph_summary or {},
            },
        },
    }


def test_ddt_benchmark_matrix_compare_case_passes_on_text_and_spec_match():
    rows = {
        "ddt_mask_kernel": _benchmark_row(["a", "b"]),
        "ddt_tree_verify_kernel": _benchmark_row(["a", "b"]),
    }

    comparison = compare_case_results(
        "ddt_mask_kernel",
        "ddt_tree_verify_kernel",
        rows,
    )

    assert comparison["pass"]
    assert comparison["text_match"]
    assert comparison["spec_decode_match"]
    assert comparison["graph_summary_match"]


def test_ddt_benchmark_matrix_compare_case_fails_on_text_diff():
    rows = {
        "ddt_mask_kernel": _benchmark_row(["a", "b"]),
        "ddt_tree_verify_kernel": _benchmark_row(["a", "c"]),
    }

    comparison = compare_case_results(
        "ddt_mask_kernel",
        "ddt_tree_verify_kernel",
        rows,
    )

    assert not comparison["pass"]
    assert not comparison["text_match"]
    assert comparison["first_text_diff"] == 1


def test_ddt_benchmark_matrix_compare_case_reports_token_diff():
    rows = {
        "vanilla": _benchmark_row(
            ["same"],
            token_ids=[[10, 20]],
            prompt_ids=["p0"],
        ),
        "ddt_tree_verify_kernel": _benchmark_row(
            ["same"],
            token_ids=[[10, 21]],
            prompt_ids=["p0"],
        ),
    }

    comparison = compare_case_results(
        "vanilla",
        "ddt_tree_verify_kernel",
        rows,
        compare_spec_decode=False,
    )

    assert not comparison["pass"]
    assert comparison["text_match"]
    assert not comparison["token_match"]
    assert comparison["first_token_diff_prompt"] == 0
    assert comparison["first_token_diff_prompt_id"] == "p0"
    assert comparison["first_token_diff_index"] == 1
    assert comparison["first_token_diff_tokens"] == {"ref": 20, "case": 21}


def test_ddt_benchmark_matrix_compare_case_fails_on_spec_diff():
    rows = {
        "ddt_mask_kernel": _benchmark_row(["a", "b"], accepted=3),
        "ddt_tree_verify_kernel": _benchmark_row(["a", "b"], accepted=2),
    }

    comparison = compare_case_results(
        "ddt_mask_kernel",
        "ddt_tree_verify_kernel",
        rows,
    )

    assert not comparison["pass"]
    assert comparison["text_match"]
    assert not comparison["spec_decode_match"]


def test_ddt_benchmark_matrix_compare_text_case_ignores_spec_diff():
    rows = {
        "vanilla": _benchmark_row(["a", "b"], accepted=0),
        "ddt_tree_verify_kernel": _benchmark_row(["a", "b"], accepted=2),
    }

    comparison = compare_case_results(
        "vanilla",
        "ddt_tree_verify_kernel",
        rows,
        compare_spec_decode=False,
    )

    assert comparison["pass"]
    assert comparison["text_match"]
    assert not comparison["spec_decode_match"]
    assert not comparison["compare_spec_decode"]


def test_ddt_benchmark_matrix_repeat_stability_passes():
    rows = [
        {
            "case": "vanilla",
            "repeat_index": 1,
            **_benchmark_row(["same"], token_ids=[[10, 20]], prompt_ids=["p0"]),
        },
        {
            "case": "vanilla",
            "repeat_index": 2,
            **_benchmark_row(["same"], token_ids=[[10, 20]], prompt_ids=["p0"]),
        },
    ]

    stability = compare_repeat_stability(case="vanilla", rows=rows)

    assert stability["pass"]
    assert stability["repeat_count"] == 2
    assert stability["comparisons"][0]["token_match"]


def test_ddt_benchmark_matrix_repeat_stability_reports_first_diff():
    rows = [
        {
            "case": "vanilla",
            "repeat_index": 1,
            **_benchmark_row(["same"], token_ids=[[10, 20]], prompt_ids=["p0"]),
        },
        {
            "case": "vanilla",
            "repeat_index": 2,
            **_benchmark_row(["same"], token_ids=[[10, 21]], prompt_ids=["p0"]),
        },
    ]

    stability = compare_repeat_stability(case="vanilla", rows=rows)

    assert not stability["pass"]
    comparison = stability["comparisons"][0]
    assert comparison["first_token_diff_prompt_id"] == "p0"
    assert comparison["first_token_diff_index"] == 1
    assert comparison["first_token_diff_tokens"] == {"ref": 20, "case": 21}


def test_ddt_benchmark_matrix_compare_case_reports_graph_diff():
    rows = {
        "ddt_cudagraph_probe": _benchmark_row(
            ["a", "b"],
            graph_summary={"graph_records": 2, "cudagraph_replay_count_delta": 2},
        ),
        "ddt_tree_verify_kernel_cudagraph_probe": _benchmark_row(
            ["a", "b"],
            graph_summary={"graph_records": 2, "cudagraph_replay_count_delta": 3},
        ),
    }

    comparison = compare_case_results(
        "ddt_cudagraph_probe",
        "ddt_tree_verify_kernel_cudagraph_probe",
        rows,
    )

    assert comparison["pass"]
    assert not comparison["graph_summary_match"]
    assert comparison["ref_graph_summary"]["cudagraph_replay_count_delta"] == 2
    assert comparison["graph_summary"]["cudagraph_replay_count_delta"] == 3


def test_ddt_benchmark_matrix_sdt_baseline_gate_requires_real_static_decode():
    passing = evaluate_sdt_baseline_gate(
        "tree_static",
        _benchmark_row(
            ["a"],
            case="tree_static",
            canonical_case="tree_static",
            static_kv_relocation=True,
        )["result"],
    )
    failing = evaluate_sdt_baseline_gate(
        "tree_static",
        _benchmark_row(
            ["a"],
            case="tree_static",
            canonical_case="tree_static",
            static_kv_relocation=False,
            num_drafts=0,
            num_draft_tokens=0,
        )["result"],
    )
    not_required = evaluate_sdt_baseline_gate(
        "ddt_tree_verify_kernel",
        _benchmark_row(["a"])["result"],
    )

    assert passing["required"]
    assert passing["pass"]
    assert not failing["pass"]
    assert failing["reasons"] == [
        "static_kv_relocation_disabled",
        "num_drafts_zero",
        "num_draft_tokens_zero",
    ]
    assert not_required["pass"]
    assert not not_required["required"]


def test_ddt_benchmark_matrix_acceptance_gate_requires_real_acceptance():
    passing = evaluate_acceptance_gate(
        case="ddt_tree_verify_kernel",
        result=_benchmark_row(
            ["a"],
            case="ddt_tree_verify_kernel",
            accepted=3,
            num_drafts=2,
            num_draft_tokens=6,
        )["result"],
        required_cases={"ddt_tree_verify_kernel"},
        min_accepted_tokens=1,
    )
    failing = evaluate_acceptance_gate(
        case="ddt_tree_verify_kernel",
        result=_benchmark_row(
            ["a"],
            case="ddt_tree_verify_kernel",
            accepted=0,
            num_drafts=2,
            num_draft_tokens=6,
        )["result"],
        required_cases={"ddt_tree_verify_kernel"},
        min_accepted_tokens=1,
    )
    not_required = evaluate_acceptance_gate(
        case="tree_static",
        result=_benchmark_row(["a"], case="tree_static", accepted=0)["result"],
        required_cases={"ddt_tree_verify_kernel"},
        min_accepted_tokens=1,
    )

    assert passing["required"]
    assert passing["pass"]
    assert passing["num_accepted_tokens"] == 3
    assert failing["required"]
    assert not failing["pass"]
    assert failing["reasons"] == ["accepted_tokens_below_threshold"]
    assert not_required["pass"]
    assert not not_required["required"]


def test_ddt_benchmark_matrix_speedup_gate_checks_sdt_baseline_and_ratio():
    passing_rows = {
        "tree_static": {
            **_benchmark_row(
                ["a"],
                case="tree_static",
                canonical_case="tree_static",
                static_kv_relocation=True,
                throughput_tok_s=100.0,
            ),
            "sdt_baseline_gate": {
                "required": True,
                "pass": True,
            },
        },
        "ddt_tree_verify_kernel": _benchmark_row(
            ["a"],
            case="ddt_tree_verify_kernel",
            throughput_tok_s=250.0,
        ),
    }
    invalid_rows = {
        **passing_rows,
        "tree_static": {
            **passing_rows["tree_static"],
            "sdt_baseline_gate": {
                "required": True,
                "pass": False,
                "reasons": ["num_drafts_zero"],
            },
        },
    }

    passing = compare_speedup_results(
        "tree_static",
        "ddt_tree_verify_kernel",
        passing_rows,
        min_speedup=1.5,
    )
    invalid = compare_speedup_results(
        "tree_static",
        "ddt_tree_verify_kernel",
        invalid_rows,
        min_speedup=1.5,
    )

    assert passing["pass"]
    assert passing["speedup"] == 2.5
    assert not invalid["pass"]
    assert invalid["reason"] == "invalid_ref_sdt_baseline"


def test_ddt_benchmark_matrix_graph_coverage_gate():
    passing = evaluate_graph_coverage_gate(
        case="ddt_cudagraph_probe",
        graph_summary={
            "graph_records": 4,
            "cudagraph_dispatch_hit_records": 4,
            "cudagraph_eager_fallback_records": 0,
            "cudagraph_metadata_buffered_records": 4,
            "cudagraph_replay_count_delta": 3,
            "metadata_device_buffered_records": 4,
            "metadata_typed_view_records": 4,
            "dynamic_select_vectorized_records": 4,
        },
        min_graph_replay_delta=1,
    )
    failing = evaluate_graph_coverage_gate(
        case="ddt_cudagraph_probe",
        graph_summary={
            "graph_records": 4,
            "cudagraph_dispatch_hit_records": 4,
            "cudagraph_eager_fallback_records": 1,
            "cudagraph_replay_count_delta": 3,
            "cudagraph_fallback_reasons": {"metadata_not_buffered": 1},
            "cudagraph_metadata_buffered_records": 3,
            "metadata_device_buffered_records": 3,
            "metadata_typed_view_records": 3,
            "dynamic_select_vectorized_records": 4,
            "metadata_device_buffer_reasons": {"runtime_buffers_unavailable": 1},
        },
        min_graph_replay_delta=1,
    )
    not_required = evaluate_graph_coverage_gate(
        case="ddt_mask_kernel",
        graph_summary={},
        min_graph_replay_delta=1,
    )

    assert passing["required"]
    assert passing["pass"]
    assert failing["required"]
    assert not failing["pass"]
    assert not_required == {
        "required": False,
        "pass": True,
        "graph_records": 0,
        "dispatch_hit_records": 0,
        "eager_fallback_records": 0,
        "metadata_buffered_records": 0,
        "metadata_device_buffered_records": 0,
        "metadata_typed_view_records": 0,
        "dynamic_select_vectorized_records": 0,
        "replay_count_delta": 0,
        "min_replay_count_delta": 1,
        "fallback_reasons": {},
        "metadata_device_buffer_reasons": {},
    }


def test_ddt_benchmark_matrix_merge_graph_summaries_tracks_metadata():
    merged = merge_benchmark_graph_summaries(
        [
            {
                "graph_records": 1,
                "graph_key_buckets": {"a": 1},
                "cudagraph_runtime_modes": {"PIECEWISE": 1},
                "metadata_host_staged_records": 1,
                "metadata_device_buffered_records": 1,
                "metadata_typed_view_records": 1,
                "dynamic_select_vectorized_records": 1,
                "metadata_device_buffer_reasons": {},
            },
            {
                "graph_records": 2,
                "graph_key_buckets": {"a": 1, "b": 1},
                "cudagraph_runtime_modes": {"NONE": 2},
                "metadata_device_buffer_reasons": {"tree_width>4": 2},
            },
        ]
    )

    assert merged["graph_records"] == 3
    assert merged["graph_key_buckets"] == {"a": 2, "b": 1}
    assert merged["graph_key_bucket_count"] == 2
    assert merged["cudagraph_runtime_modes"] == {"NONE": 2, "PIECEWISE": 1}
    assert merged["metadata_host_staged_records"] == 1
    assert merged["metadata_device_buffered_records"] == 1
    assert merged["metadata_typed_view_records"] == 1
    assert merged["dynamic_select_vectorized_records"] == 1
    assert merged["metadata_device_buffer_reasons"] == {"tree_width>4": 2}


def test_ddt_benchmark_matrix_parse_compare_cases():
    assert parse_compare_cases(["a=b", "c=d"]) == [("a", "b"), ("c", "d")]


def test_ddt_benchmark_matrix_summarizes_repeat_comparisons():
    summary = summarize_repeat_rows(
        repeat=2,
        compare_pairs=[("ref", "kernel", True)],
        rows=[
            {"case": "ref", "repeat_index": 1, "returncode": 0},
            {
                "case": "kernel",
                "repeat_index": 1,
                "returncode": 0,
                "graph_summary": {
                    "graph_records": 1,
                    "cudagraph_replay_count_delta": 2,
                },
                "graph_coverage_gate": {"required": True, "pass": True},
                "acceptance_gate": {
                    "required": True,
                    "pass": True,
                    "num_accepted_tokens": 3,
                },
            },
            {
                "case": "compare:ref=kernel",
                "kind": "comparison",
                "repeat_index": 1,
                "returncode": 0,
                "comparison": {
                    "ref_case": "ref",
                    "case": "kernel",
                    "compare_spec_decode": True,
                },
            },
            {
                "case": "speedup:ref=kernel",
                "kind": "performance_comparison",
                "repeat_index": 1,
                "returncode": 0,
                "performance": {
                    "ref_case": "ref",
                    "case": "kernel",
                    "pass": True,
                    "speedup": 1.5,
                },
            },
            {"case": "ref", "repeat_index": 2, "returncode": 0},
            {
                "case": "kernel",
                "repeat_index": 2,
                "returncode": 0,
                "graph_summary": {
                    "graph_records": 1,
                    "cudagraph_replay_count_delta": 3,
                },
                "graph_coverage_gate": {"required": True, "pass": False},
                "acceptance_gate": {
                    "required": True,
                    "pass": False,
                    "num_accepted_tokens": 0,
                    "reasons": ["accepted_tokens_below_threshold"],
                },
            },
            {
                "case": "compare:ref=kernel",
                "kind": "comparison",
                "repeat_index": 2,
                "returncode": 1,
                "comparison": {
                    "ref_case": "ref",
                    "case": "kernel",
                    "compare_spec_decode": True,
                },
            },
            {
                "case": "speedup:ref=kernel",
                "kind": "performance_comparison",
                "repeat_index": 2,
                "returncode": 0,
                "performance": {
                    "ref_case": "ref",
                    "case": "kernel",
                    "pass": True,
                    "speedup": 1.6,
                },
            },
        ],
    )

    assert not summary["pass"]
    assert summary["case_returncodes"] == {"ref": [0, 0], "kernel": [0, 0]}
    assert summary["comparisons"] == [
        {
            "ref_case": "ref",
            "case": "kernel",
            "compare_spec_decode": True,
            "pass_count": 1,
            "fail_count": 1,
            "total": 2,
        }
    ]
    assert summary["performance_comparisons"] == [
        {
            "ref_case": "ref",
            "case": "kernel",
            "pass": True,
            "speedup": 1.5,
        },
        {
            "ref_case": "ref",
            "case": "kernel",
            "pass": True,
            "speedup": 1.6,
        },
    ]
    assert summary["graph_summary"]["graph_records"] == 2
    assert summary["graph_summary"]["cudagraph_replay_count_delta"] == 5
    assert summary["graph_coverage_gates"] == [
        {
            "case": "kernel",
            "repeat_index": 1,
            "required": True,
            "pass": True,
        },
        {
            "case": "kernel",
            "repeat_index": 2,
            "required": True,
            "pass": False,
        },
    ]
    assert summary["acceptance_gates"] == [
        {
            "case": "kernel",
            "repeat_index": 1,
            "required": True,
            "pass": True,
            "num_accepted_tokens": 3,
        },
        {
            "case": "kernel",
            "repeat_index": 2,
            "required": True,
            "pass": False,
            "num_accepted_tokens": 0,
            "reasons": ["accepted_tokens_below_threshold"],
        },
    ]


def test_ddt_benchmark_enables_branching_relocation_by_default(monkeypatch):
    captured = {}

    def fake_llm(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("benchmarks.spec_decode.benchmark_tree_attn_ddt.LLM", fake_llm)

    args = Namespace(
        disable_mla=False,
        trace_path=None,
        case="ddt_mask_kernel",
        model="target",
        max_model_len=128,
        max_num_batched_tokens=128,
        gpu_memory_utilization=0.5,
        compilation_config=None,
        target_attn_backend=None,
        async_scheduling=None,
        method="eagle3",
        draft_model="draft",
        tree="[(0,), (1,), (0, 0), (0, 1)]",
        draft_attn_backend="TREE_ATTN",
        ddt_runtime_mode="branching",
        ddt_max_draft_tokens=3,
        disable_ddt_kv_relocation=False,
        dynamic_metadata_select_path="default",
        device_metadata_handle=False,
        torch_profiler_dir=None,
        custom_profile_scopes=False,
        nvtx_profile_scopes=False,
    )

    build_llm(args)

    assert captured["attention_config"] == {"backend": "TREE_ATTN"}
    assert captured["speculative_config"][
        "dynamic_draft_tree_runtime_mode"
    ] == "branching"
    assert captured["speculative_config"][
        "enable_tree_spec_decode_kv_relocation"
    ]
    assert captured["speculative_config"]["enable_dynamic_tree_target_mask"]


def test_ddt_benchmark_enables_device_metadata_handle_switches(monkeypatch):
    captured = {}

    def fake_llm(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("benchmarks.spec_decode.benchmark_tree_attn_ddt.LLM", fake_llm)
    monkeypatch.delenv("VLLM_DYNAMIC_TREE_SELECTED_BOOL_METADATA_KERNEL", raising=False)
    monkeypatch.delenv("VLLM_DYNAMIC_TREE_DEVICE_METADATA_HANDLE", raising=False)

    args = Namespace(
        disable_mla=False,
        trace_path=None,
        stage_profile=False,
        stage_profile_sync=False,
        draft_stage_profile=False,
        dynamic_metadata_stage_profile=True,
        draft_stage_profile_sync=False,
        sample_stage_profile=False,
        sample_stage_profile_sync=False,
        tree_verify_kernel=False,
        case="ddt_mask_kernel",
        model="target",
        max_model_len=128,
        max_num_batched_tokens=128,
        gpu_memory_utilization=0.5,
        compilation_config=None,
        target_attn_backend=None,
        async_scheduling=None,
        method="eagle3",
        draft_model="draft",
        tree="[(0,), (1,), (0, 0), (0, 1)]",
        draft_attn_backend="TREE_ATTN",
        ddt_runtime_mode="branching",
        ddt_max_draft_tokens=3,
        disable_ddt_kv_relocation=False,
        enable_static_kv_relocation=False,
        dynamic_metadata_select_path="selected_bool_kernel",
        device_metadata_handle=True,
        torch_profiler_dir=None,
        custom_profile_scopes=False,
        nvtx_profile_scopes=False,
    )

    build_llm(args)

    assert captured["speculative_config"]["enable_dynamic_draft_tree"]
    assert os.environ["VLLM_DYNAMIC_TREE_SELECTED_BOOL_METADATA_KERNEL"] == "1"
    assert os.environ["VLLM_DYNAMIC_TREE_DEVICE_METADATA_HANDLE"] == "1"
    assert os.environ["VLLM_DYNAMIC_TREE_METADATA_STAGE_PROFILE"] == "1"


def test_ddt_benchmark_torch_profiler_config(monkeypatch, tmp_path):
    captured = {}

    def fake_llm(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("benchmarks.spec_decode.benchmark_tree_attn_ddt.LLM", fake_llm)
    monkeypatch.delenv("VLLM_CUSTOM_SCOPES_FOR_PROFILING", raising=False)

    args = Namespace(
        disable_mla=False,
        trace_path=None,
        stage_profile=False,
        stage_profile_sync=False,
        draft_stage_profile=False,
        dynamic_metadata_stage_profile=False,
        draft_stage_profile_sync=False,
        sample_stage_profile=False,
        sample_stage_profile_sync=False,
        tree_verify_kernel=False,
        custom_profile_scopes=True,
        nvtx_profile_scopes=False,
        case="vanilla",
        model="target",
        max_model_len=128,
        max_num_batched_tokens=128,
        gpu_memory_utilization=0.5,
        compilation_config=None,
        torch_profiler_dir=tmp_path / "prof",
        torch_profiler_record_shapes=True,
        torch_profiler_with_memory=True,
        torch_profiler_with_stack=False,
        torch_profiler_active_iters=2,
        torch_profiler_warmup_iters=1,
        target_attn_backend=None,
        async_scheduling=None,
        draft_model="draft",
        method="eagle3",
        tree="auto",
        draft_attn_backend="auto",
        ddt_runtime_mode="branching",
        ddt_max_draft_tokens=None,
        disable_ddt_kv_relocation=False,
        enable_static_kv_relocation=False,
        dynamic_metadata_select_path="default",
        device_metadata_handle=False,
    )

    build_llm(args)

    assert captured["profiler_config"] == {
        "profiler": "torch",
        "torch_profiler_dir": str(tmp_path / "prof"),
        "torch_profiler_record_shapes": True,
        "torch_profiler_with_memory": True,
        "torch_profiler_with_stack": False,
        "active_iterations": 2,
        "warmup_iterations": 1,
    }
    assert (tmp_path / "prof").is_dir()
    assert os.environ["VLLM_CUSTOM_SCOPES_FOR_PROFILING"] == "1"


def test_ddt_benchmark_stage_profile_requires_trace_path():
    args = Namespace(
        disable_mla=False,
        trace_path=None,
        stage_profile=True,
        stage_profile_sync=False,
        case="vanilla",
        serial_repair_scope="all",
        custom_profile_scopes=False,
        nvtx_profile_scopes=False,
    )

    try:
        build_llm(args)
    except ValueError as exc:
        assert "--stage-profile requires --trace-path" in str(exc)
    else:
        raise AssertionError("expected stage profile trace-path guard")


def test_ddt_benchmark_loads_prompt_file(tmp_path):
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text(
        '{"id":"a","prompt":"first"}\n{"id":"b","prompt":"second"}\n',
        encoding="utf-8",
    )

    prompts = load_prompts(
        Namespace(prompt_file=prompt_file, prompts=["fallback"], prompt_repeat=1)
    )

    assert prompts == ["first", "second"]


def test_ddt_benchmark_repeats_prompt_file(tmp_path):
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text(
        '{"id":"a","prompt":"first"}\n{"id":"b","prompt":"second"}\n',
        encoding="utf-8",
    )

    args = Namespace(prompt_file=prompt_file, prompts=["fallback"], prompt_repeat=2)
    prompts = load_prompts(args)
    prompt_ids = load_prompt_ids(args, num_prompts=len(prompts))

    assert prompts == ["first", "second", "first", "second"]
    assert prompt_ids == ["a", "b", "a#r1", "b#r1"]


def test_ddt_benchmark_rejects_invalid_prompt_repeat(tmp_path):
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text('{"id":"a","prompt":"first"}\n', encoding="utf-8")

    try:
        load_prompts(
            Namespace(prompt_file=prompt_file, prompts=[], prompt_repeat=0)
        )
    except ValueError as exc:
        assert "--prompt-repeat must be >= 1" in str(exc)
    else:
        raise AssertionError("expected prompt-repeat validation failure")


def test_ddt_benchmark_loads_prompt_ids(tmp_path):
    prompt_file = tmp_path / "prompts.jsonl"
    prompt_file.write_text(
        '{"id":"a","prompt":"first"}\n{"prompt":"second"}\n',
        encoding="utf-8",
    )

    prompt_ids = load_prompt_ids(
        Namespace(prompt_file=prompt_file, prompts=[], prompt_repeat=1),
        num_prompts=2,
    )

    assert prompt_ids == ["a", "prompt_1"]


def test_ddt_benchmark_generated_token_ids_are_serializable():
    class Output:
        def __init__(self, token_ids):
            self.outputs = [type("Inner", (), {"token_ids": token_ids})()]

    assert generated_token_ids([Output((1, 2)), Output([3])]) == [[1, 2], [3]]


def test_benchmark_matrix_to_harness_builds_replay_shape():
    ref_row = _benchmark_row(
        ["alpha", "beta"],
        token_ids=[[1, 2], [3, 4]],
        prompt_ids=["p0", "p1"],
    )
    case_row = _benchmark_row(
        ["alpha", "beta changed"],
        token_ids=[[1, 2], [3, 5]],
        prompt_ids=["p0", "p1"],
    )

    harness = build_benchmark_harness(
        ref_case="vanilla",
        case="ddt",
        ref_row=ref_row,
        case_row=case_row,
    )

    comparison = harness["comparisons"]["ddt"]
    assert not comparison["all_text_match"]
    assert not comparison["all_token_match"]
    assert comparison["diffs"][0]["token_match"]
    assert comparison["diffs"][1]["first_token_diff"] == 1
    assert comparison["diffs"][1]["baseline_token_ids"] == [3, 4]
    assert comparison["diffs"][1]["case_token_ids"] == [3, 5]


def test_large_topology_prepare_paths_uses_defaults(tmp_path):
    args = Namespace(
        output_dir=str(tmp_path),
        matrix_jsonl=None,
        ref_trace=None,
        case_trace=None,
        ref_case="vanilla",
        case="ddt_tree_verify_kernel",
    )

    matrix, ref_trace, case_trace = large_topology_prepare_paths(args)

    assert matrix == tmp_path / "trace_matrix.jsonl"
    assert ref_trace == large_topology_default_trace_path(tmp_path, "vanilla")
    assert case_trace == large_topology_default_trace_path(
        tmp_path, "ddt_tree_verify_kernel"
    )


def test_large_topology_benchmark_args_forward_device_handle(tmp_path):
    args = Namespace(
        model="target",
        draft_model="draft",
        method="eagle3",
        target_attn_backend="TREE_ATTN",
        tree="binary62",
        prompt_file="prompts.jsonl",
        prompt_repeat=2,
        max_tokens=16,
        warmup_iters=0,
        iters=1,
        gpu_memory_utilization=0.6,
        max_num_batched_tokens=2048,
        ddt_max_draft_tokens=3,
        dynamic_metadata_select_path="default",
        device_metadata_handle=True,
    )

    forwarded = large_topology_benchmark_args(args, tmp_path / "trace.jsonl")

    assert "--benchmark-arg=--dynamic-metadata-select-path" in forwarded
    assert "--benchmark-arg=default" in forwarded
    assert "--benchmark-arg=--prompt-repeat" in forwarded
    assert "--benchmark-arg=2" in forwarded
    assert "--benchmark-arg=--device-metadata-handle" in forwarded


def test_large_topology_graph_summary_for_cases(tmp_path):
    matrix = tmp_path / "matrix.jsonl"
    matrix.write_text(
        "\n".join(
            [
                '{"case":"vanilla","repeat_index":1,"returncode":0,'
                '"graph_summary":{"graph_records":0}}',
                '{"case":"ddt_tree_verify_kernel","repeat_index":1,'
                '"returncode":0,"graph_summary":{"graph_records":2,'
                '"cudagraph_replay_count_delta":4,'
                '"dynamic_select_vectorized_records":2}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = large_topology_graph_summary_for_cases(
        matrix,
        ref_case="vanilla",
        case="ddt_tree_verify_kernel",
    )

    assert summary["graph_records"] == 2
    assert summary["cudagraph_replay_count_delta"] == 4
    assert summary["dynamic_select_vectorized_records"] == 2


def test_large_topology_summary_marks_hard_fail(tmp_path):
    matrix = tmp_path / "matrix.jsonl"
    matrix.write_text(
        "\n".join(
            [
                '{"case":"vanilla","repeat_index":1,"returncode":0,'
                '"graph_summary":{"graph_records":0}}',
                '{"case":"ddt_tree_verify_kernel","repeat_index":1,'
                '"returncode":0,"graph_summary":{"graph_records":1}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    outcomes = tmp_path / "outcomes.json"
    outcomes.write_text(
        '{"summary":{"hard_fail":1},"rows":[{"outcome":"hard_fail",'
        '"prompt_id":"p0"}]}',
        encoding="utf-8",
    )

    summary = write_large_topology_summary(
        args=Namespace(
            tree="binary62",
            ref_case="vanilla",
            case="ddt_tree_verify_kernel",
            model="target",
            draft_model="draft",
            prompt_file="prompts.jsonl",
            prompt_repeat=2,
            max_tokens=16,
            dynamic_metadata_select_path="default",
            device_metadata_handle=True,
            low_margin_threshold=0.5,
            max_hard_fails=0,
            require_graph_coverage=False,
            min_graph_replay_delta=1,
        ),
        output_dir=tmp_path,
        matrix_path=matrix,
        ref_trace=tmp_path / "ref.jsonl",
        case_trace=tmp_path / "case.jsonl",
        harness=tmp_path / "harness.json",
        classification=tmp_path / "classification.json",
        outcomes=outcomes,
    )

    assert not summary["pass"]
    assert summary["hard_fail_count"] == 1
    assert summary["hard_fails"] == [{"outcome": "hard_fail", "prompt_id": "p0"}]
    assert summary["dynamic_metadata_select_path"] == "default"
    assert summary["device_metadata_handle"] is True
    assert summary["prompt_repeat"] == 2


def test_large_topology_summary_can_require_graph_coverage(tmp_path):
    matrix = tmp_path / "matrix.jsonl"
    matrix.write_text(
        "\n".join(
            [
                '{"case":"vanilla","repeat_index":1,"returncode":0,'
                '"graph_summary":{"graph_records":0}}',
                '{"case":"ddt_tree_verify_kernel","repeat_index":1,'
                '"returncode":0,"graph_summary":{'
                '"graph_records":2,'
                '"cudagraph_dispatch_hit_records":2,'
                '"cudagraph_eager_fallback_records":0,'
                '"cudagraph_metadata_buffered_records":2,'
                '"cudagraph_replay_count_delta":4,'
                '"compact_kernel_records":2,'
                '"compact_kernel_requested_records":2,'
                '"metadata_device_buffered_records":2,'
                '"metadata_typed_view_records":2,'
                '"dynamic_select_vectorized_records":2}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    outcomes = tmp_path / "outcomes.json"
    outcomes.write_text(
        '{"summary":{"explained_low_margin":1},"rows":[]}',
        encoding="utf-8",
    )

    summary = write_large_topology_summary(
        args=Namespace(
            tree="binary62",
            ref_case="vanilla",
            case="ddt_tree_verify_kernel",
            model="target",
            draft_model="draft",
            prompt_file="prompts.jsonl",
            prompt_repeat=2,
            max_tokens=16,
            dynamic_metadata_select_path="default",
            device_metadata_handle=True,
            low_margin_threshold=0.5,
            max_hard_fails=0,
            require_graph_coverage=True,
            min_graph_replay_delta=1,
        ),
        output_dir=tmp_path,
        matrix_path=matrix,
        ref_trace=tmp_path / "ref.jsonl",
        case_trace=tmp_path / "case.jsonl",
        harness=tmp_path / "harness.json",
        classification=tmp_path / "classification.json",
        outcomes=outcomes,
    )

    assert summary["pass"]
    assert summary["graph_coverage_gate"]["required"]
    assert summary["graph_coverage_gate"]["pass"]
    assert summary["graph_coverage_gate"]["metadata_typed_view_records"] == 2
    assert summary["prompt_repeat"] == 2


def test_large_topology_summary_fails_missing_graph_coverage(tmp_path):
    matrix = tmp_path / "matrix.jsonl"
    matrix.write_text(
        "\n".join(
            [
                '{"case":"vanilla","repeat_index":1,"returncode":0,'
                '"graph_summary":{"graph_records":0}}',
                '{"case":"ddt_tree_verify_kernel","repeat_index":1,'
                '"returncode":0,"graph_summary":{'
                '"graph_records":2,'
                '"cudagraph_dispatch_hit_records":2,'
                '"cudagraph_eager_fallback_records":0,'
                '"cudagraph_metadata_buffered_records":2,'
                '"cudagraph_replay_count_delta":4,'
                '"metadata_device_buffered_records":2,'
                '"metadata_typed_view_records":1,'
                '"dynamic_select_vectorized_records":2}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    outcomes = tmp_path / "outcomes.json"
    outcomes.write_text('{"summary":{},"rows":[]}', encoding="utf-8")

    summary = write_large_topology_summary(
        args=Namespace(
            tree="binary62",
            ref_case="vanilla",
            case="ddt_tree_verify_kernel",
            model="target",
            draft_model="draft",
            prompt_file="prompts.jsonl",
            prompt_repeat=2,
            max_tokens=16,
            dynamic_metadata_select_path="default",
            device_metadata_handle=False,
            low_margin_threshold=0.5,
            max_hard_fails=0,
            require_graph_coverage=True,
            min_graph_replay_delta=1,
        ),
        output_dir=tmp_path,
        matrix_path=matrix,
        ref_trace=tmp_path / "ref.jsonl",
        case_trace=tmp_path / "case.jsonl",
        harness=tmp_path / "harness.json",
        classification=tmp_path / "classification.json",
        outcomes=outcomes,
    )

    assert not summary["pass"]
    assert summary["hard_fail_count"] == 0
    assert not summary["graph_coverage_gate"]["pass"]
    assert summary["prompt_repeat"] == 2


def test_ddt_benchmark_trace_acceptance_counts_non_prefix_and_relocation():
    metrics = _trace_acceptance_from_records(
        [
            {
                "scheduled_spec_decode_tokens": [1, 2, 3],
                "accept_indices": [0, 1, 3, -1],
                "tree_relocation_pairs": [{"src_local": 3, "dst_local": 2}],
                "tree_parent": [-1, 0, 0, 1],
                "tree_compact_bias_kernel_requested": True,
                "tree_compact_bias_kernel_used": True,
            },
            {
                "scheduled_spec_decode_tokens": [4, 5, 6],
                "accept_indices": [0, -1, -1, -1],
                "tree_parent": [-1, 0, 0, 1],
                "tree_attn_bias_mask": [[1]],
                "tree_compact_bias_kernel_requested": True,
                "tree_compact_bias_kernel_used": True,
            },
            {
                "trace_kind": "tree_attn_stage_profile",
                "scheduled_spec_decode_tokens": 3,
                "accepted_tokens": 2,
                "stage_ms": {"sample_ms": 1.0},
            },
            {"scheduled_spec_decode_tokens": []},
        ]
    )

    assert metrics["num_drafts"] == 2
    assert metrics["num_draft_tokens"] == 6
    assert metrics["num_accepted_tokens"] == 2
    assert metrics["non_prefix_accepts"] == 1
    assert metrics["relocation_pairs"] == 1
    assert metrics["compact_kernel_records"] == 2
    assert metrics["compact_kernel_requested_records"] == 2
    assert metrics["dense_mask_records"] == 0


def test_ddt_benchmark_trace_stage_profile_summary():
    summary = _trace_stage_profile_from_records(
        [
            {
                "trace_kind": "tree_attn_stage_profile",
                "stage_ms": {
                    "sample_ms": 1.0,
                    "relocation_ms": 0.5,
                    "bookkeeping_ms": 1.5,
                },
                "stage_total_ms": 3.0,
                "output_tokens": 4,
                "accepted_tokens": 2,
                "scheduled_spec_decode_tokens": 3,
                "relocation_pairs": 1,
                "near_tie_fallback_rows": 0,
                "serial_repair_rows": 2,
                "serial_repair_prefix_rows": 1,
                "serial_repair_nonprefix_rows": 1,
                "serial_repair_near_tie_rows": 1,
                "serial_repair_scope": "all",
                "serial_repair_batch_by_depth": True,
                "tree_cudagraph_runtime": {
                    "mode": "CUDAGraphMode.PIECEWISE",
                    "metadata_buffered": True,
                    "dispatch_hit": True,
                    "capture_count_delta": 1,
                    "replay_count_delta": 2,
                },
                "tree_compact_bias_kernel_requested": True,
                "tree_compact_bias_kernel_used": True,
            },
            {
                "trace_kind": "tree_attn_stage_profile",
                "stage_ms": {
                    "sample_ms": 2.0,
                    "relocation_ms": 1.0,
                },
                "stage_total_ms": 3.0,
                "output_tokens": 2,
                "accepted_tokens": 1,
                "scheduled_spec_decode_tokens": 3,
                "relocation_pairs": 0,
                "near_tie_fallback_rows": 1,
                "serial_repair_rows": 0,
                "serial_repair_scope": "nonprefix",
                "tree_cudagraph_runtime": {
                    "mode": "CUDAGraphMode.NONE",
                    "metadata_buffered": False,
                    "eager_fallback": True,
                    "fallback_reason": "cudagraph_mode_none",
                },
                "tree_compact_bias_kernel_requested": True,
                "tree_compact_bias_kernel_used": False,
            },
            {"trace_kind": "spec_verify_state"},
        ]
    )

    assert summary["num_records"] == 2
    assert summary["stage_total_ms"] == 6.0
    assert summary["stage_total_ms_per_record"] == 3.0
    assert summary["stage_total_ms_per_output_token"] == 1.0
    assert summary["stage_totals_ms"]["sample_ms"] == 3.0
    assert summary["stage_avg_ms"]["sample_ms"] == 1.5
    assert summary["stage_pct"]["sample_ms"] == 0.5
    assert summary["output_tokens"] == 6
    assert summary["accepted_tokens"] == 3
    assert summary["scheduled_spec_decode_tokens"] == 6
    assert summary["relocation_pairs"] == 1
    assert summary["near_tie_fallback_rows"] == 1
    assert summary["serial_repair_rows"] == 2
    assert summary["serial_repair_prefix_rows"] == 1
    assert summary["serial_repair_nonprefix_rows"] == 1
    assert summary["serial_repair_near_tie_rows"] == 1
    assert summary["serial_repair_scopes"] == {"all": 1, "nonprefix": 1}
    assert summary["serial_repair_batch_by_depth_records"] == 1
    assert summary["compact_kernel_records"] == 1
    assert summary["compact_kernel_requested_records"] == 2
    assert summary["cudagraph_runtime_modes"] == {
        "CUDAGraphMode.PIECEWISE": 1,
        "CUDAGraphMode.NONE": 1,
    }
    assert summary["cudagraph_fallback_reasons"] == {"cudagraph_mode_none": 1}
    assert summary["cudagraph_metadata_buffered_records"] == 1
    assert summary["cudagraph_dispatch_hit_records"] == 1
    assert summary["cudagraph_eager_fallback_records"] == 1
    assert summary["cudagraph_capture_count_delta"] == 1
    assert summary["cudagraph_replay_count_delta"] == 2


def test_ddt_benchmark_trace_graph_summary_buckets_runtime_keys():
    summary = _trace_graph_summary_from_records(
        [
            {
                "trace_kind": "spec_verify_state",
                "tree_cudagraph_key": {
                    "tree_width": 4,
                    "max_query_len": 4,
                    "num_reqs": 2,
                    "serial_repair_scope": "fallback_or_nonprefix",
                    "dynamic_mask_kernel": True,
                },
                "tree_cudagraph_runtime": {
                    "mode": "PIECEWISE",
                    "num_tokens_unpadded": 8,
                    "num_tokens_padded": 16,
                    "batch_descriptor": {"num_tokens": 16},
                    "dispatch_hit": True,
                    "metadata_buffered": True,
                },
                "tree_compact_bias_kernel_requested": True,
                "tree_compact_bias_kernel_used": True,
            },
            {
                "trace_kind": "tree_attn_stage_profile",
                "tree_cudagraph_key": {
                    "tree_width": 4,
                    "max_query_len": 4,
                    "num_reqs": 2,
                    "serial_repair_scope": "fallback_or_nonprefix",
                    "dynamic_mask_kernel": True,
                },
                "tree_cudagraph_runtime": {
                    "mode": "PIECEWISE",
                    "num_tokens_unpadded": 8,
                    "num_tokens_padded": 16,
                    "batch_descriptor": {"num_tokens": 16},
                    "dispatch_hit": True,
                    "metadata_buffered": True,
                    "replay_count_delta": 3,
                },
                "tree_compact_bias_kernel_requested": True,
                "tree_compact_bias_kernel_used": True,
            },
            {
                "trace_kind": "tree_attn_stage_profile",
                "tree_cudagraph_key": {
                    "tree_width": 4,
                    "max_query_len": 8,
                    "num_reqs": 1,
                    "serial_repair_scope": "all",
                    "dynamic_mask_kernel": True,
                },
                "tree_cudagraph_runtime": {
                    "mode": "CUDAGraphMode.NONE",
                    "num_tokens_unpadded": 9,
                    "num_tokens_padded": 9,
                    "batch_descriptor": {"num_tokens": 9},
                    "eager_fallback": True,
                    "fallback_reason": "cudagraph_mode_none",
                },
            },
        ]
    )

    assert summary["graph_records"] == 3
    assert summary["graph_key_bucket_count"] == 2
    assert summary["cudagraph_runtime_modes"] == {
        "CUDAGraphMode.NONE": 1,
        "PIECEWISE": 2,
    }
    assert summary["cudagraph_fallback_reasons"] == {"cudagraph_mode_none": 1}
    assert summary["cudagraph_dispatch_hit_records"] == 2
    assert summary["cudagraph_eager_fallback_records"] == 1
    assert summary["cudagraph_metadata_buffered_records"] == 2
    assert summary["cudagraph_replay_count_delta"] == 3
    assert summary["compact_kernel_records"] == 2
    assert summary["compact_kernel_requested_records"] == 2


def test_ddt_benchmark_trace_graph_summary_derives_replay_delta_from_totals():
    summary = _trace_graph_summary_from_records(
        [
            {
                "trace_kind": "spec_verify_state",
                "tree_cudagraph_runtime": {
                    "mode": "PIECEWISE",
                    "dispatch_hit": True,
                    "capture_count_total_at_dispatch": 7,
                    "replay_count_total_at_dispatch": 20,
                },
            },
            {
                "trace_kind": "spec_verify_state",
                "tree_cudagraph_runtime": {
                    "mode": "PIECEWISE",
                    "dispatch_hit": True,
                    "capture_count_total_at_dispatch": 7,
                    "replay_count_total_at_dispatch": 22,
                },
            },
            {
                "trace_kind": "spec_verify_state",
                "tree_cudagraph_runtime": {
                    "mode": "PIECEWISE",
                    "dispatch_hit": True,
                    "capture_count_total_at_dispatch": 8,
                    "replay_count_total_at_dispatch": 25,
                },
            },
        ]
    )

    assert summary["graph_records"] == 3
    assert summary["cudagraph_capture_count_delta"] == 1
    assert summary["cudagraph_replay_count_delta"] == 5


def test_ddt_regression_merge_graph_summaries():
    merged = merge_graph_summaries(
        [
            {
                "graph_records": 2,
                "graph_key_buckets": {"a": 1},
                "cudagraph_runtime_modes": {"PIECEWISE": 2},
                "cudagraph_fallback_reasons": {},
                "cudagraph_dispatch_hit_records": 2,
                "cudagraph_replay_count_delta": 10,
            },
            {
                "graph_records": 3,
                "graph_key_buckets": {"a": 2, "b": 1},
                "cudagraph_runtime_modes": {"PIECEWISE": 2, "NONE": 1},
                "cudagraph_fallback_reasons": {"metadata_not_buffered": 1},
                "cudagraph_eager_fallback_records": 1,
                "cudagraph_replay_count_delta": 4,
            },
        ]
    )

    assert merged["graph_records"] == 5
    assert merged["graph_key_buckets"] == {"a": 3, "b": 1}
    assert merged["graph_key_bucket_count"] == 2
    assert merged["cudagraph_runtime_modes"] == {"NONE": 1, "PIECEWISE": 4}
    assert merged["cudagraph_fallback_reasons"] == {"metadata_not_buffered": 1}
    assert merged["cudagraph_dispatch_hit_records"] == 2
    assert merged["cudagraph_eager_fallback_records"] == 1
    assert merged["cudagraph_replay_count_delta"] == 14


def test_ddt_regression_graph_coverage_gate_tracks_metadata_path():
    passing = regression_graph_coverage_gate(
        {
            "graph_records": 2,
            "cudagraph_dispatch_hit_records": 2,
            "cudagraph_eager_fallback_records": 0,
            "cudagraph_metadata_buffered_records": 2,
            "cudagraph_replay_count_delta": 4,
            "metadata_device_buffered_records": 2,
            "metadata_typed_view_records": 2,
            "dynamic_select_vectorized_records": 2,
        },
        require_graph_records=True,
    )
    failing = regression_graph_coverage_gate(
        {
            "graph_records": 2,
            "cudagraph_dispatch_hit_records": 2,
            "cudagraph_eager_fallback_records": 0,
            "cudagraph_metadata_buffered_records": 1,
            "cudagraph_replay_count_delta": 4,
            "metadata_device_buffered_records": 2,
            "metadata_typed_view_records": 1,
            "dynamic_select_vectorized_records": 2,
            "metadata_device_buffer_reasons": {"runtime_buffers_unavailable": 1},
        },
        require_graph_records=True,
    )
    not_required = regression_graph_coverage_gate(
        {},
        require_graph_records=False,
    )

    assert passing["pass"]
    assert not failing["pass"]
    assert failing["metadata_buffered_records"] == 1
    assert failing["metadata_device_buffer_reasons"] == {
        "runtime_buffers_unavailable": 1
    }
    assert not_required["pass"]
    assert not not_required["required"]
