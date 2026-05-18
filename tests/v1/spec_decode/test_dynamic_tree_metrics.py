# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import numpy as np
import torch

from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.v1.outputs import SamplerOutput
from vllm.v1.spec_decode.dynamic_tree_metrics import (
    annotate_tree_cudagraph_runtime,
    make_tree_stage_profile_record,
)
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata


def _tree_metadata() -> SpecDecodeMetadata:
    metadata = SpecDecodeMetadata.make_dummy([[11, 12]], device=torch.device("cpu"))
    metadata.tree_target_logits_indices = torch.arange(
        3, dtype=torch.int32
    ).view(1, 3)
    metadata.tree_retrieve_index = metadata.tree_target_logits_indices.clone()
    metadata.tree_retrieve_next_token = torch.tensor(
        [[1, 2, -1]], dtype=torch.int32
    )
    metadata.tree_retrieve_next_sibling = torch.tensor(
        [[-1, -1, -1]], dtype=torch.int32
    )
    metadata.tree_parent = torch.tensor([[-1, 0, 1]], dtype=torch.int32)
    metadata.tree_num_spec_steps = 3
    metadata.tree_runtime_mode = "branching"
    return metadata


def test_annotate_tree_cudagraph_runtime_records_bucket_and_fallback():
    metadata = _tree_metadata()
    metadata.tree_cudagraph_metadata_buffered = False
    batch_desc = BatchDescriptor(num_tokens=8, num_reqs=1, uniform=True)

    annotate_tree_cudagraph_runtime(
        spec_decode_metadata=metadata,
        cudagraph_mode=CUDAGraphMode.NONE,
        batch_desc=batch_desc,
        num_reqs=1,
        num_tokens_unpadded=3,
        num_tokens_padded=8,
        forward_num_scheduled_tokens_np=np.array([3], dtype=np.int32),
        compact_kernel_requested=True,
        tree_attn_cudagraph_probe=True,
        serial_repair_enabled=True,
        serial_repair_scope="fallback_or_nonprefix",
        metadata_buffer_capacity=4,
        cudagraph_capture_count=5,
        cudagraph_replay_count=7,
    )

    assert metadata.tree_cudagraph_key == {
        "runtime_mode": "branching",
        "tree_width": 3,
        "q_lens": [3],
        "max_query_len": 3,
        "num_reqs": 1,
        "num_draft_tokens": [2],
        "dynamic_mask_kernel": True,
        "near_tie_threshold": None,
        "serial_repair_enabled": True,
        "serial_repair_scope": "fallback_or_nonprefix",
        "metadata_buffer_capacity": 4,
    }
    assert metadata.tree_cudagraph_runtime is not None
    assert metadata.tree_cudagraph_runtime["mode"] == "NONE"
    assert metadata.tree_cudagraph_runtime["eager_fallback"]
    assert (
        metadata.tree_cudagraph_runtime["fallback_reason"]
        == "cudagraph_mode_none"
    )
    assert metadata.tree_cudagraph_runtime["num_paddings"] == 5
    assert metadata.tree_cudagraph_runtime["capture_count_total_at_dispatch"] == 5
    assert metadata.tree_cudagraph_runtime["replay_count_total_at_dispatch"] == 7


def test_make_tree_stage_profile_record_centralizes_runtime_metrics():
    metadata = _tree_metadata()
    metadata.tree_cudagraph_key = {"tree_width": 3}
    metadata.tree_cudagraph_runtime = {
        "mode": "PIECEWISE",
        "capture_count_total_at_dispatch": 2,
        "replay_count_total_at_dispatch": 3,
    }
    metadata.tree_cudagraph_metadata_buffered = True
    metadata.tree_metadata_device_buffered = True
    metadata.tree_metadata_typed_view = True
    metadata.tree_dynamic_select_vectorized = True
    metadata.tree_serial_accepted_state_repair_applied = [
        {"is_nonprefix": 1, "has_near_tie_fallback": 1},
        {"is_nonprefix": 0, "has_near_tie_fallback": 0},
    ]
    metadata.tree_near_tie_q1_fallback_applied = [{"req_idx": 0}]
    sampler_output = SamplerOutput(
        sampled_token_ids=torch.tensor([[101, 102, -1]], dtype=torch.int32),
        logprobs_tensors=None,
        spec_decode_accept_indices=torch.tensor([[0, 2, -1]], dtype=torch.int32),
    )

    record = make_tree_stage_profile_record(
        step=9,
        req_ids=["req0"],
        scheduled_spec_decode_tokens={"req0": [11, 12]},
        spec_decode_metadata=metadata,
        sampler_output=sampler_output,
        stage_ms={"sample_ms": 1.0, "bookkeeping_ms": 2.0},
        relocation_pairs=1,
        compact_kernel_requested=True,
        query_start_loc=[0, 3],
        cudagraph_capture_count=4,
        cudagraph_replay_count=6,
        serial_repair_scope="fallback_or_nonprefix",
        serial_repair_batch_by_depth=True,
        draft_stage_ms={"draft_ms": 0.5},
        dynamic_metadata_stage_ms={"select_ms": 0.25},
        sample_stage_ms={"verify_ms": 0.75},
        sync=False,
    )

    assert record["trace_kind"] == "tree_attn_stage_profile"
    assert record["step"] == 9
    assert record["scheduled_spec_decode_tokens"] == 2
    assert record["output_tokens"] == 2
    assert record["accepted_tokens"] == 1
    assert record["relocation_pairs"] == 1
    assert record["tree_compact_bias_kernel_requested"]
    assert record["tree_compact_bias_kernel_used"]
    assert record["tree_cudagraph_runtime"]["capture_count_delta"] == 2
    assert record["tree_cudagraph_runtime"]["replay_count_delta"] == 3
    assert record["near_tie_fallback_rows"] == 1
    assert record["serial_repair_rows"] == 2
    assert record["serial_repair_prefix_rows"] == 1
    assert record["serial_repair_nonprefix_rows"] == 1
    assert record["serial_repair_near_tie_rows"] == 1
    assert record["tree_cudagraph_metadata_buffered"]
    assert record["tree_metadata_device_buffered"]
    assert record["tree_metadata_typed_view"]
    assert record["tree_dynamic_select_vectorized"]
    assert record["draft_stage_ms"] == {"draft_ms": 0.5}
    assert record["dynamic_metadata_stage_ms"] == {"select_ms": 0.25}
    assert record["sample_stage_ms"] == {"verify_ms": 0.75}
    assert record["stage_total_ms"] == 3.0
