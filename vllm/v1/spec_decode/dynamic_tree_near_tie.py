# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeAlias

import torch

from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata

NearTieCandidate: TypeAlias = dict[str, int | float]
NearTieRecord: TypeAlias = dict[
    str,
    int | float | str | list[int] | list[float] | None,
]
SerialQ1OutputFn: TypeAlias = Callable[
    [Sequence[NearTieCandidate]],
    tuple[torch.Tensor, list[torch.Tensor] | None, torch.Tensor],
]


def logits_top2_margin(logits_row: torch.Tensor) -> float:
    top2 = torch.topk(logits_row.to(torch.float32), k=2).values
    return float((top2[0] - top2[1]).item())


def logits_topk_trace(
    logits: torch.Tensor,
    *,
    k: int = 5,
) -> tuple[list[list[int]], list[list[float]], list[float | None]]:
    if logits.numel() == 0 or k <= 0:
        return [], [], []
    k = min(k, logits.shape[-1])
    top_values, top_indices = torch.topk(logits.to(torch.float32), k=k, dim=-1)
    top_values_cpu = top_values.detach().cpu().tolist()
    top_indices_cpu = top_indices.detach().cpu().tolist()
    margins: list[float | None] = []
    for values in top_values_cpu:
        if len(values) < 2:
            margins.append(None)
        else:
            margins.append(float(values[0] - values[1]))
    return (
        [[int(token_id) for token_id in row] for row in top_indices_cpu],
        [[float(value) for value in row] for row in top_values_cpu],
        margins,
    )


def tree_near_tie_q1_replacement_candidates(
    sampler_output: SamplerOutput,
    logits: torch.Tensor | None,
    spec_decode_metadata: SpecDecodeMetadata | None,
) -> list[NearTieCandidate]:
    if (
        logits is None
        or spec_decode_metadata is None
        or not spec_decode_metadata.has_tree_metadata
        or spec_decode_metadata.tree_near_tie_q1_fallback_threshold is None
        or spec_decode_metadata.tree_target_logits_indices is None
        or sampler_output.spec_decode_accept_indices is None
    ):
        return []
    threshold = spec_decode_metadata.tree_near_tie_q1_fallback_threshold
    accept_indices = sampler_output.spec_decode_accept_indices
    output_token_ids = sampler_output.sampled_token_ids
    tree_indices = spec_decode_metadata.tree_target_logits_indices
    candidates: list[NearTieCandidate] = []
    num_reqs = min(accept_indices.shape[0], output_token_ids.shape[0])
    for req_idx in range(num_reqs):
        if req_idx >= len(spec_decode_metadata.num_draft_tokens):
            break
        if int(spec_decode_metadata.num_draft_tokens[req_idx]) <= 0:
            continue
        if (
            spec_decode_metadata.tree_valid is not None
            and not bool(spec_decode_metadata.tree_valid[req_idx].item())
        ):
            continue
        row_width = min(accept_indices.shape[1], output_token_ids.shape[1])
        for output_idx in range(row_width):
            token_id = int(output_token_ids[req_idx, output_idx].item())
            if token_id < 0:
                break
            local_idx = int(accept_indices[req_idx, output_idx].item())
            if local_idx < 0 or local_idx >= tree_indices.shape[1]:
                break
            logits_idx = int(tree_indices[req_idx, local_idx].item())
            if logits_idx < 0 or logits_idx >= logits.shape[0]:
                break
            margin = logits_top2_margin(logits[logits_idx])
            if margin <= threshold:
                candidates.append(
                    {
                        "req_idx": req_idx,
                        "output_idx": output_idx,
                        "local_idx": local_idx,
                        "logits_idx": logits_idx,
                        "margin": margin,
                    }
                )
                break
    return candidates


def apply_tree_near_tie_q1_fallback(
    *,
    sampler_output: SamplerOutput,
    logits: torch.Tensor | None,
    spec_decode_metadata: SpecDecodeMetadata | None,
    logits_indices: torch.Tensor,
    hidden_states: torch.Tensor | None,
    sample_hidden_states: torch.Tensor | None,
    aux_hidden_states: list[torch.Tensor] | None,
    serial_q1_outputs_for_candidates: SerialQ1OutputFn | None,
) -> SamplerOutput:
    candidates = tree_near_tie_q1_replacement_candidates(
        sampler_output,
        logits,
        spec_decode_metadata,
    )
    if not candidates or spec_decode_metadata is None or logits is None:
        return sampler_output

    root_candidates = [
        candidate for candidate in candidates if int(candidate["output_idx"]) == 0
    ]
    applied: list[NearTieRecord] = []
    if root_candidates and serial_q1_outputs_for_candidates is not None:
        (
            serial_hidden_states,
            serial_aux_hidden_states,
            serial_logits,
        ) = serial_q1_outputs_for_candidates(root_candidates)
        for row_idx, candidate in enumerate(root_candidates):
            logits_idx = int(candidate["logits_idx"])
            req_idx = int(candidate["req_idx"])
            input_idx = int(logits_indices[logits_idx].item())
            old_output = sampler_output.sampled_token_ids[req_idx]
            truncated_tokens = int((old_output[1:] >= 0).sum().item())
            logits[logits_idx].copy_(serial_logits[row_idx])
            if hidden_states is not None and input_idx < hidden_states.shape[0]:
                hidden_states[input_idx].copy_(serial_hidden_states[row_idx])
            if (
                sample_hidden_states is not None
                and logits_idx < sample_hidden_states.shape[0]
            ):
                sample_hidden_states[logits_idx].copy_(serial_hidden_states[row_idx])
            if aux_hidden_states is not None and serial_aux_hidden_states is not None:
                for aux_hidden, serial_aux_hidden in zip(
                    aux_hidden_states,
                    serial_aux_hidden_states,
                ):
                    if input_idx < aux_hidden.shape[0]:
                        aux_hidden[input_idx].copy_(serial_aux_hidden[row_idx])
            token_id = int(serial_logits[row_idx].argmax(dim=-1).item())
            q1_top_token_ids, q1_top_values, q1_top_margins = logits_topk_trace(
                serial_logits[row_idx : row_idx + 1],
                k=5,
            )
            sampler_output.sampled_token_ids[req_idx].fill_(PLACEHOLDER_TOKEN_ID)
            sampler_output.sampled_token_ids[req_idx, 0] = token_id
            if sampler_output.spec_decode_accept_indices is not None:
                sampler_output.spec_decode_accept_indices[req_idx].fill_(
                    PLACEHOLDER_TOKEN_ID
                )
                sampler_output.spec_decode_accept_indices[req_idx, 0] = 0
            applied.append(
                {
                    **candidate,
                    "input_idx": input_idx,
                    "fallback": "root_q1",
                    "q1_token_id": token_id,
                    "q1_top_token_ids": q1_top_token_ids[0]
                    if q1_top_token_ids
                    else [],
                    "q1_top_values": q1_top_values[0] if q1_top_values else [],
                    "q1_margin": q1_top_margins[0] if q1_top_margins else None,
                    "threshold": (
                        spec_decode_metadata.tree_near_tie_q1_fallback_threshold
                    ),
                    "truncated_accepted_tokens": truncated_tokens,
                }
            )

    for candidate in candidates:
        output_idx = int(candidate["output_idx"])
        if output_idx == 0:
            continue
        req_idx = int(candidate["req_idx"])
        truncated_tokens = int(
            (sampler_output.sampled_token_ids[req_idx, output_idx:] >= 0)
            .sum()
            .item()
        )
        sampler_output.sampled_token_ids[req_idx, output_idx:].fill_(
            PLACEHOLDER_TOKEN_ID
        )
        if sampler_output.spec_decode_accept_indices is not None:
            sampler_output.spec_decode_accept_indices[req_idx, output_idx:].fill_(
                PLACEHOLDER_TOKEN_ID
            )
        applied.append(
            {
                **candidate,
                "fallback": "truncate_before_near_tie",
                "threshold": (
                    spec_decode_metadata.tree_near_tie_q1_fallback_threshold
                ),
                "truncated_accepted_tokens": truncated_tokens,
            }
        )

    if applied:
        spec_decode_metadata.tree_near_tie_q1_fallback_applied = applied
    return sampler_output
