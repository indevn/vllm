# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID
from vllm.v1.spec_decode.dynamic_tree_near_tie import (
    NearTieCandidate,
    apply_tree_near_tie_q1_fallback,
    tree_near_tie_q1_replacement_candidates,
)
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata


def _install_minimal_tree_metadata(
    metadata: SpecDecodeMetadata,
    *,
    batch_size: int,
    width: int,
) -> None:
    metadata.tree_target_logits_indices = torch.arange(
        batch_size * width, dtype=torch.int32
    ).view(batch_size, width)
    metadata.tree_retrieve_index = metadata.tree_target_logits_indices.clone()
    metadata.tree_retrieve_next_token = torch.full(
        (batch_size, width), -1, dtype=torch.int32
    )
    metadata.tree_retrieve_next_sibling = torch.full(
        (batch_size, width), -1, dtype=torch.int32
    )
    metadata.tree_num_spec_steps = width


def test_near_tie_candidates_ignore_no_draft_and_invalid_tree_rows():
    logits = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [0.0, 3.0, 0.0],
        ]
    )
    metadata = SpecDecodeMetadata.make_dummy([[], [1]], device=torch.device("cpu"))
    _install_minimal_tree_metadata(metadata, batch_size=2, width=2)
    metadata.tree_near_tie_q1_fallback_threshold = 0.0
    metadata.tree_valid = torch.tensor([True, False])
    sampler_output = SamplerOutput(
        sampled_token_ids=torch.tensor([[0, -1], [1, -1]], dtype=torch.int32),
        logprobs_tensors=None,
        spec_decode_accept_indices=torch.tensor([[0, -1], [0, -1]], dtype=torch.int32),
    )

    assert (
        tree_near_tie_q1_replacement_candidates(
            sampler_output,
            logits,
            metadata,
        )
        == []
    )


def test_near_tie_fallback_truncates_before_non_root_tie():
    logits = torch.tensor(
        [
            [10.0, 0.0, 0.0],
            [0.0, 7.0, 7.0],
            [0.0, 0.0, 8.0],
        ]
    )
    metadata = SpecDecodeMetadata.make_dummy([[1, 2]], device=torch.device("cpu"))
    _install_minimal_tree_metadata(metadata, batch_size=1, width=3)
    metadata.tree_near_tie_q1_fallback_threshold = 0.0
    sampler_output = SamplerOutput(
        sampled_token_ids=torch.tensor([[1, 2, -1]], dtype=torch.int32),
        logprobs_tensors=None,
        spec_decode_accept_indices=torch.tensor([[0, 1, -1]], dtype=torch.int32),
    )

    output = apply_tree_near_tie_q1_fallback(
        sampler_output=sampler_output,
        logits=logits,
        spec_decode_metadata=metadata,
        logits_indices=torch.arange(3),
        hidden_states=None,
        sample_hidden_states=None,
        aux_hidden_states=None,
        serial_q1_outputs_for_candidates=None,
    )

    assert output.sampled_token_ids.tolist() == [[1, PLACEHOLDER_TOKEN_ID, -1]]
    assert output.spec_decode_accept_indices is not None
    assert output.spec_decode_accept_indices.tolist() == [
        [0, PLACEHOLDER_TOKEN_ID, -1]
    ]
    assert metadata.tree_near_tie_q1_fallback_applied == [
        {
            "fallback": "truncate_before_near_tie",
            "logits_idx": 1,
            "local_idx": 1,
            "margin": 0.0,
            "output_idx": 1,
            "req_idx": 0,
            "threshold": 0.0,
            "truncated_accepted_tokens": 1,
        }
    ]


def test_near_tie_root_fallback_replaces_output_and_state():
    logits = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [0.0, 3.0, 0.0],
        ]
    )
    metadata = SpecDecodeMetadata.make_dummy([[1]], device=torch.device("cpu"))
    _install_minimal_tree_metadata(metadata, batch_size=1, width=2)
    metadata.tree_near_tie_q1_fallback_threshold = 0.0
    sampler_output = SamplerOutput(
        sampled_token_ids=torch.tensor([[0, -1]], dtype=torch.int32),
        logprobs_tensors=None,
        spec_decode_accept_indices=torch.tensor([[0, -1]], dtype=torch.int32),
    )
    hidden_states = torch.zeros((2, 4))
    sample_hidden_states = hidden_states.clone()
    aux_hidden_states = [torch.zeros((2, 4))]
    serial_hidden = torch.full((1, 4), 2.0)
    serial_aux_hidden = [torch.full((1, 4), 3.0)]
    serial_logits = torch.tensor([[0.0, 5.0, 1.0]])

    def serial_q1(
        candidates: list[NearTieCandidate],
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        assert candidates == [
            {
                "req_idx": 0,
                "output_idx": 0,
                "local_idx": 0,
                "logits_idx": 0,
                "margin": 0.0,
            }
        ]
        return serial_hidden, serial_aux_hidden, serial_logits

    output = apply_tree_near_tie_q1_fallback(
        sampler_output=sampler_output,
        logits=logits,
        spec_decode_metadata=metadata,
        logits_indices=torch.arange(2),
        hidden_states=hidden_states,
        sample_hidden_states=sample_hidden_states,
        aux_hidden_states=aux_hidden_states,
        serial_q1_outputs_for_candidates=serial_q1,
    )

    assert output.sampled_token_ids.tolist() == [[1, PLACEHOLDER_TOKEN_ID]]
    assert output.spec_decode_accept_indices is not None
    assert output.spec_decode_accept_indices.tolist() == [[0, PLACEHOLDER_TOKEN_ID]]
    assert logits[0].tolist() == [0.0, 5.0, 1.0]
    assert hidden_states[0].tolist() == [2.0, 2.0, 2.0, 2.0]
    assert sample_hidden_states[0].tolist() == [2.0, 2.0, 2.0, 2.0]
    assert aux_hidden_states[0][0].tolist() == [3.0, 3.0, 3.0, 3.0]
    assert metadata.tree_near_tie_q1_fallback_applied == [
        {
            "fallback": "root_q1",
            "input_idx": 0,
            "logits_idx": 0,
            "local_idx": 0,
            "margin": 0.0,
            "output_idx": 0,
            "q1_margin": 4.0,
            "q1_token_id": 1,
            "q1_top_token_ids": [1, 2, 0],
            "q1_top_values": [5.0, 1.0, 0.0],
            "req_idx": 0,
            "threshold": 0.0,
            "truncated_accepted_tokens": 0,
        }
    ]
