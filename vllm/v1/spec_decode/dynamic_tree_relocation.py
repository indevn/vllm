# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Sequence

import torch

from vllm.v1.outputs import SamplerOutput
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata


def dynamic_tree_relocation_pairs(
    *,
    sampler_output: SamplerOutput,
    spec_decode_metadata: SpecDecodeMetadata | None,
    num_reqs: int,
    query_start_loc: torch.Tensor,
) -> list[dict[str, int]]:
    if (
        spec_decode_metadata is None
        or not spec_decode_metadata.has_tree_metadata
        or spec_decode_metadata.tree_linear_kv_safe
    ):
        return []
    if spec_decode_metadata.tree_relocation_pairs_cache is not None:
        return spec_decode_metadata.tree_relocation_pairs_cache

    accept_indices = sampler_output.spec_decode_accept_indices
    if accept_indices is None or accept_indices.shape[-1] <= 1:
        spec_decode_metadata.tree_relocation_pairs_cache = []
        return []

    num_reqs = min(num_reqs, accept_indices.shape[0])
    if num_reqs == 0:
        spec_decode_metadata.tree_relocation_pairs_cache = []
        return []

    pairs: list[dict[str, int]] = []
    for req_idx in range(num_reqs):
        row_start = int(query_start_loc[req_idx].item())
        row_end = int(query_start_loc[req_idx + 1].item())
        row_width = row_end - row_start
        if row_width <= 1:
            continue
        max_outputs = min(accept_indices.shape[1], row_width)
        for out_pos in range(1, max_outputs):
            src_local = int(accept_indices[req_idx, out_pos].item())
            if src_local < 0 or src_local >= row_width:
                continue
            dst_local = out_pos
            if src_local == dst_local:
                continue
            pairs.append(
                {
                    "req_idx": req_idx,
                    "src_local": src_local,
                    "dst_local": dst_local,
                    "src_index": row_start + src_local,
                    "dst_index": row_start + dst_local,
                }
            )
    spec_decode_metadata.tree_relocation_pairs_cache = pairs
    return pairs


def dynamic_tree_sample_relocation_pairs(
    *,
    pairs: Sequence[dict[str, int]],
    spec_decode_metadata: SpecDecodeMetadata | None,
) -> list[dict[str, int]]:
    if (
        spec_decode_metadata is None
        or spec_decode_metadata.tree_target_logits_indices is None
        or not pairs
    ):
        return []
    if spec_decode_metadata.tree_sample_relocation_pairs_cache is not None:
        return spec_decode_metadata.tree_sample_relocation_pairs_cache

    sample_pairs: list[dict[str, int]] = []
    tree_indices = spec_decode_metadata.tree_target_logits_indices.detach().cpu()
    for pair in pairs:
        req_idx = pair["req_idx"]
        if req_idx >= tree_indices.shape[0]:
            continue
        if (
            pair["src_local"] >= tree_indices.shape[1]
            or pair["dst_local"] >= tree_indices.shape[1]
        ):
            continue
        sample_pairs.append(
            {
                **pair,
                "src_index": int(tree_indices[req_idx, pair["src_local"]].item()),
                "dst_index": int(tree_indices[req_idx, pair["dst_local"]].item()),
            }
        )
    spec_decode_metadata.tree_sample_relocation_pairs_cache = sample_pairs
    return sample_pairs


def clear_dynamic_tree_relocation_cache(
    spec_decode_metadata: SpecDecodeMetadata | None,
) -> None:
    if spec_decode_metadata is None:
        return
    spec_decode_metadata.tree_relocation_pairs_cache = None
    spec_decode_metadata.tree_sample_relocation_pairs_cache = None
    spec_decode_metadata.tree_relocation_index_cache = None
    spec_decode_metadata.tree_sample_relocation_index_cache = None


def dynamic_tree_relocation_local_tensors(
    *,
    sampler_output: SamplerOutput,
    spec_decode_metadata: SpecDecodeMetadata | None,
    num_reqs: int,
    query_start_loc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if (
        spec_decode_metadata is None
        or not spec_decode_metadata.has_tree_metadata
        or spec_decode_metadata.tree_linear_kv_safe
    ):
        return None
    accept_indices = sampler_output.spec_decode_accept_indices
    if accept_indices is None or accept_indices.shape[-1] <= 1:
        return None

    num_reqs = min(
        num_reqs,
        accept_indices.shape[0],
        len(spec_decode_metadata.num_draft_tokens),
    )
    if num_reqs == 0:
        return None

    device = accept_indices.device
    query_start_loc = query_start_loc[: num_reqs + 1].to(
        device=device,
        dtype=torch.long,
    )
    row_starts = query_start_loc[:-1]
    row_widths = query_start_loc[1:] - row_starts
    max_outputs = accept_indices.shape[1]
    dst_local = torch.arange(
        max_outputs,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0).expand(num_reqs, -1)
    src_local = accept_indices[:num_reqs, :max_outputs].to(torch.long)
    row_widths_expanded = row_widths.unsqueeze(1)
    valid = (
        (dst_local > 0)
        & (dst_local < row_widths_expanded)
        & (src_local >= 0)
        & (src_local < row_widths_expanded)
        & (src_local != dst_local)
    )
    req_indices = torch.arange(
        num_reqs,
        dtype=torch.long,
        device=device,
    ).unsqueeze(1).expand_as(dst_local)
    req_indices = req_indices[valid]
    if req_indices.numel() == 0:
        return None
    return (
        req_indices,
        src_local[valid],
        dst_local[valid],
        row_starts[req_indices],
    )


def dynamic_tree_relocation_index_tensors(
    *,
    sampler_output: SamplerOutput,
    spec_decode_metadata: SpecDecodeMetadata | None,
    num_reqs: int,
    query_start_loc: torch.Tensor,
    device: torch.device,
    use_tensor_indices: bool,
) -> tuple[torch.Tensor, torch.Tensor, int] | None:
    if spec_decode_metadata is None:
        return None
    if spec_decode_metadata.tree_relocation_index_cache is not None:
        return spec_decode_metadata.tree_relocation_index_cache

    if use_tensor_indices:
        local_tensors = dynamic_tree_relocation_local_tensors(
            sampler_output=sampler_output,
            spec_decode_metadata=spec_decode_metadata,
            num_reqs=num_reqs,
            query_start_loc=query_start_loc,
        )
        if local_tensors is None:
            return None
        _, src_local, dst_local, row_starts = local_tensors
        src_indices = row_starts + src_local
        dst_indices = row_starts + dst_local
        max_index = int(torch.maximum(src_indices.max(), dst_indices.max()).item())
        cache = (
            src_indices.to(device=device, dtype=torch.long),
            dst_indices.to(device=device, dtype=torch.long),
            max_index,
        )
    else:
        pairs = dynamic_tree_relocation_pairs(
            sampler_output=sampler_output,
            spec_decode_metadata=spec_decode_metadata,
            num_reqs=num_reqs,
            query_start_loc=query_start_loc,
        )
        if not pairs:
            return None
        src_indices = [pair["src_index"] for pair in pairs]
        dst_indices = [pair["dst_index"] for pair in pairs]
        max_index = max(max(src_indices), max(dst_indices))
        cache = (
            torch.tensor(src_indices, dtype=torch.long, device=device),
            torch.tensor(dst_indices, dtype=torch.long, device=device),
            max_index,
        )
    spec_decode_metadata.tree_relocation_index_cache = cache
    return cache


def dynamic_tree_sample_relocation_index_tensors(
    *,
    sampler_output: SamplerOutput,
    spec_decode_metadata: SpecDecodeMetadata | None,
    num_reqs: int,
    query_start_loc: torch.Tensor,
    device: torch.device,
    use_tensor_indices: bool,
) -> tuple[torch.Tensor, torch.Tensor, int] | None:
    if spec_decode_metadata is None:
        return None
    if spec_decode_metadata.tree_sample_relocation_index_cache is not None:
        return spec_decode_metadata.tree_sample_relocation_index_cache

    if use_tensor_indices:
        local_tensors = dynamic_tree_relocation_local_tensors(
            sampler_output=sampler_output,
            spec_decode_metadata=spec_decode_metadata,
            num_reqs=num_reqs,
            query_start_loc=query_start_loc,
        )
        if (
            local_tensors is None
            or spec_decode_metadata.tree_target_logits_indices is None
        ):
            return None
        req_indices, src_local, dst_local, _ = local_tensors
        tree_indices = spec_decode_metadata.tree_target_logits_indices.to(
            device=req_indices.device,
            dtype=torch.long,
        )
        tree_width = tree_indices.shape[1]
        in_tree_bounds = (src_local < tree_width) & (dst_local < tree_width)
        if not bool(in_tree_bounds.any().item()):
            return None
        req_indices = req_indices[in_tree_bounds]
        src_local = src_local[in_tree_bounds]
        dst_local = dst_local[in_tree_bounds]
        src_indices = tree_indices[req_indices, src_local]
        dst_indices = tree_indices[req_indices, dst_local]
        valid_indices = (src_indices >= 0) & (dst_indices >= 0)
        if not bool(valid_indices.any().item()):
            return None
        src_indices = src_indices[valid_indices]
        dst_indices = dst_indices[valid_indices]
        max_index = int(torch.maximum(src_indices.max(), dst_indices.max()).item())
        cache = (
            src_indices.to(device=device, dtype=torch.long),
            dst_indices.to(device=device, dtype=torch.long),
            max_index,
        )
    else:
        pairs = dynamic_tree_sample_relocation_pairs(
            pairs=dynamic_tree_relocation_pairs(
                sampler_output=sampler_output,
                spec_decode_metadata=spec_decode_metadata,
                num_reqs=num_reqs,
                query_start_loc=query_start_loc,
            ),
            spec_decode_metadata=spec_decode_metadata,
        )
        if not pairs:
            return None
        src_indices = [pair["src_index"] for pair in pairs]
        dst_indices = [pair["dst_index"] for pair in pairs]
        max_index = max(max(src_indices), max(dst_indices))
        cache = (
            torch.tensor(src_indices, dtype=torch.long, device=device),
            torch.tensor(dst_indices, dtype=torch.long, device=device),
            max_index,
        )
    spec_decode_metadata.tree_sample_relocation_index_cache = cache
    return cache
