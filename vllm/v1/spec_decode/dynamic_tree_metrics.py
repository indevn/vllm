# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.v1.outputs import SamplerOutput
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata


def annotate_tree_cudagraph_runtime(
    *,
    spec_decode_metadata: SpecDecodeMetadata | None,
    cudagraph_mode: CUDAGraphMode,
    batch_desc: BatchDescriptor,
    num_reqs: int,
    num_tokens_unpadded: int,
    num_tokens_padded: int,
    forward_num_scheduled_tokens_np: np.ndarray,
    compact_kernel_requested: bool,
    tree_attn_cudagraph_probe: bool,
    serial_repair_enabled: bool,
    serial_repair_scope: str,
    metadata_buffer_capacity: int,
    cudagraph_capture_count: int,
    cudagraph_replay_count: int,
) -> None:
    if spec_decode_metadata is None or not spec_decode_metadata.has_tree_metadata:
        return

    tree_width = (
        int(spec_decode_metadata.tree_parent.shape[-1])
        if spec_decode_metadata.tree_parent is not None
        else None
    )
    q_lens = [
        int(query_len)
        for query_len in forward_num_scheduled_tokens_np[:num_reqs].tolist()
    ]
    metadata_buffered = bool(spec_decode_metadata.tree_cudagraph_metadata_buffered)
    fallback_reason = spec_decode_metadata.tree_cudagraph_metadata_buffer_reason
    if cudagraph_mode == CUDAGraphMode.NONE:
        fallback_reason = fallback_reason or "cudagraph_mode_none"
    elif not metadata_buffered and tree_attn_cudagraph_probe:
        fallback_reason = fallback_reason or "metadata_not_buffered"

    spec_decode_metadata.tree_cudagraph_key = {
        "runtime_mode": spec_decode_metadata.tree_runtime_mode,
        "tree_width": tree_width,
        "q_lens": q_lens,
        "max_query_len": max(q_lens) if q_lens else 0,
        "num_reqs": int(num_reqs),
        "num_draft_tokens": [
            int(num_tokens)
            for num_tokens in spec_decode_metadata.num_draft_tokens[:num_reqs]
        ],
        "dynamic_mask_kernel": compact_kernel_requested,
        "near_tie_threshold": (
            spec_decode_metadata.tree_near_tie_q1_fallback_threshold
        ),
        "serial_repair_enabled": bool(serial_repair_enabled),
        "serial_repair_scope": serial_repair_scope,
        "metadata_buffer_capacity": int(metadata_buffer_capacity),
    }
    spec_decode_metadata.tree_cudagraph_runtime = {
        "probe_enabled": bool(tree_attn_cudagraph_probe),
        "mode": str(cudagraph_mode),
        "dispatch_hit": cudagraph_mode != CUDAGraphMode.NONE,
        "eager_fallback": cudagraph_mode == CUDAGraphMode.NONE,
        "num_tokens_unpadded": int(num_tokens_unpadded),
        "num_tokens_padded": int(num_tokens_padded),
        "num_paddings": int(num_tokens_padded - num_tokens_unpadded),
        "batch_descriptor": {
            "num_tokens": int(batch_desc.num_tokens),
            "num_reqs": None
            if batch_desc.num_reqs is None
            else int(batch_desc.num_reqs),
            "uniform": bool(batch_desc.uniform),
            "has_lora": bool(batch_desc.has_lora),
            "num_active_loras": int(batch_desc.num_active_loras),
        },
        "metadata_buffered": metadata_buffered,
        "fallback_reason": fallback_reason,
        "capture_count_total_at_dispatch": int(cudagraph_capture_count),
        "replay_count_total_at_dispatch": int(cudagraph_replay_count),
    }


def tree_stage_num_output_tokens(
    sampler_output: SamplerOutput | None,
) -> int:
    if sampler_output is None or sampler_output.sampled_token_ids is None:
        return 0
    return int((sampler_output.sampled_token_ids >= 0).sum().item())


def _tree_compact_kernel_used(
    *,
    spec_decode_metadata: SpecDecodeMetadata,
    compact_requested: bool,
    query_start_loc: Sequence[int] | None,
) -> bool:
    if spec_decode_metadata.tree_parent is None:
        return False
    tree_width = int(spec_decode_metadata.tree_parent.shape[-1])
    num_reqs = len(spec_decode_metadata.num_draft_tokens)
    if not query_start_loc or spec_decode_metadata.force_root_only_forward:
        return False
    query_lens = [
        int(query_start_loc[i + 1]) - int(query_start_loc[i])
        for i in range(num_reqs)
    ]
    decode_query_lens: list[int] = []
    for query_len in query_lens:
        if query_len > tree_width:
            break
        decode_query_lens.append(query_len)
    return (
        compact_requested
        and bool(decode_query_lens)
        and all(0 < query_len <= tree_width for query_len in decode_query_lens)
    )


def _tree_cudagraph_runtime_with_record_delta(
    runtime: dict | None,
    *,
    cudagraph_capture_count: int,
    cudagraph_replay_count: int,
) -> dict | None:
    if runtime is None:
        return None
    runtime = dict(runtime)
    capture_count_at_dispatch = int(
        runtime.get("capture_count_total_at_dispatch") or 0
    )
    replay_count_at_dispatch = int(runtime.get("replay_count_total_at_dispatch") or 0)
    runtime["capture_count_total_at_record"] = int(cudagraph_capture_count)
    runtime["replay_count_total_at_record"] = int(cudagraph_replay_count)
    runtime["capture_count_delta"] = max(
        0,
        runtime["capture_count_total_at_record"] - capture_count_at_dispatch,
    )
    runtime["replay_count_delta"] = max(
        0,
        runtime["replay_count_total_at_record"] - replay_count_at_dispatch,
    )
    return runtime


def make_tree_stage_profile_record(
    *,
    step: int,
    req_ids: Sequence[str],
    scheduled_spec_decode_tokens: Mapping[str, Sequence[int]],
    spec_decode_metadata: SpecDecodeMetadata,
    sampler_output: SamplerOutput | None,
    stage_ms: dict[str, float],
    relocation_pairs: int,
    compact_kernel_requested: bool,
    query_start_loc: Sequence[int] | None,
    cudagraph_capture_count: int,
    cudagraph_replay_count: int,
    serial_repair_scope: str,
    serial_repair_batch_by_depth: bool,
    draft_stage_ms: Mapping[str, float] | None,
    dynamic_metadata_stage_ms: Mapping[str, float] | None,
    sample_stage_ms: Mapping[str, float] | None,
    sync: bool,
) -> dict[str, Any]:
    num_reqs = len(spec_decode_metadata.num_draft_tokens)
    scheduled_spec_tokens = sum(
        len(tokens)
        for req_id, tokens in scheduled_spec_decode_tokens.items()
        if req_id in req_ids[:num_reqs]
    )
    output_tokens = tree_stage_num_output_tokens(sampler_output)
    accepted_tokens = max(0, output_tokens - num_reqs)
    compact_used = _tree_compact_kernel_used(
        spec_decode_metadata=spec_decode_metadata,
        compact_requested=compact_kernel_requested,
        query_start_loc=query_start_loc,
    )
    tree_cudagraph_runtime = _tree_cudagraph_runtime_with_record_delta(
        spec_decode_metadata.tree_cudagraph_runtime,
        cudagraph_capture_count=cudagraph_capture_count,
        cudagraph_replay_count=cudagraph_replay_count,
    )
    serial_repair_records = (
        spec_decode_metadata.tree_serial_accepted_state_repair_applied or []
    )
    serial_repair_nonprefix_rows = sum(
        int(record.get("is_nonprefix") or 0) for record in serial_repair_records
    )
    serial_repair_near_tie_rows = sum(
        int(record.get("has_near_tie_fallback") or 0)
        for record in serial_repair_records
    )

    return {
        "trace_kind": "tree_attn_stage_profile",
        "step": int(step),
        "num_reqs": num_reqs,
        "num_draft_tokens": [
            int(num_tokens) for num_tokens in spec_decode_metadata.num_draft_tokens
        ],
        "scheduled_spec_decode_tokens": scheduled_spec_tokens,
        "output_tokens": output_tokens,
        "accepted_tokens": accepted_tokens,
        "relocation_pairs": int(relocation_pairs),
        "tree_runtime_mode": spec_decode_metadata.tree_runtime_mode,
        "tree_linear_kv_safe": spec_decode_metadata.tree_linear_kv_safe,
        "tree_compact_bias_kernel_requested": compact_kernel_requested,
        "tree_compact_bias_kernel_used": compact_used,
        "tree_cudagraph_key": spec_decode_metadata.tree_cudagraph_key,
        "tree_cudagraph_runtime": tree_cudagraph_runtime,
        "tree_cudagraph_metadata_buffered": (
            spec_decode_metadata.tree_cudagraph_metadata_buffered
        ),
        "tree_cudagraph_metadata_buffer_reason": (
            spec_decode_metadata.tree_cudagraph_metadata_buffer_reason
        ),
        "tree_metadata_host_staged": spec_decode_metadata.tree_metadata_host_staged,
        "tree_metadata_device_buffered": (
            spec_decode_metadata.tree_metadata_device_buffered
        ),
        "tree_metadata_device_buffer_reason": (
            spec_decode_metadata.tree_metadata_device_buffer_reason
        ),
        "tree_metadata_typed_view": spec_decode_metadata.tree_metadata_typed_view,
        "tree_dynamic_select_vectorized": (
            spec_decode_metadata.tree_dynamic_select_vectorized
        ),
        "near_tie_fallback_rows": len(
            spec_decode_metadata.tree_near_tie_q1_fallback_applied or []
        ),
        "serial_repair_rows": len(serial_repair_records),
        "serial_repair_prefix_rows": (
            len(serial_repair_records) - serial_repair_nonprefix_rows
        ),
        "serial_repair_nonprefix_rows": serial_repair_nonprefix_rows,
        "serial_repair_near_tie_rows": serial_repair_near_tie_rows,
        "serial_repair_scope": serial_repair_scope,
        "serial_repair_batch_by_depth": bool(serial_repair_batch_by_depth),
        "draft_stage_ms": dict(draft_stage_ms or {}),
        "dynamic_metadata_stage_ms": dict(dynamic_metadata_stage_ms or {}),
        "sample_stage_ms": dict(sample_stage_ms or {}),
        "stage_ms": stage_ms,
        "stage_total_ms": sum(stage_ms.values()),
        "sync": bool(sync),
    }
