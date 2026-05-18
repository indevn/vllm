# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Dynamic Draft Tree verifier reference and kernel boundary."""

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.v1.spec_decode.dynamic_tree import (
    DynamicTreeDraftOutput,
    DynamicTreeVerifyOutput,
)

__all__ = [
    "DynamicTreeVerifyOutput",
    "verify_dynamic_tree_greedy",
    "verify_dynamic_tree_greedy_from_draft",
    "verify_dynamic_tree_greedy_kernel",
]


@triton.jit
def _verify_dynamic_tree_greedy_triton_kernel(
    candidates_ptr,
    retrieve_index_ptr,
    retrieve_next_token_ptr,
    retrieve_next_sibling_ptr,
    target_predict_ptr,
    tree_valid_ptr,
    target_mask_ptr,
    predicts_ptr,
    accept_index_ptr,
    accept_token_num_ptr,
    accept_token_ptr,
    candidates_stride_b: tl.constexpr,
    candidates_stride_w: tl.constexpr,
    retrieve_index_stride_b: tl.constexpr,
    retrieve_index_stride_w: tl.constexpr,
    retrieve_next_token_stride_b: tl.constexpr,
    retrieve_next_token_stride_w: tl.constexpr,
    retrieve_next_sibling_stride_b: tl.constexpr,
    retrieve_next_sibling_stride_w: tl.constexpr,
    target_predict_stride_b: tl.constexpr,
    target_predict_stride_w: tl.constexpr,
    tree_valid_stride: tl.constexpr,
    target_mask_stride_b: tl.constexpr,
    target_mask_stride_w: tl.constexpr,
    predicts_stride_b: tl.constexpr,
    predicts_stride_w: tl.constexpr,
    accept_index_stride_b: tl.constexpr,
    accept_index_stride_s: tl.constexpr,
    accept_token_stride_b: tl.constexpr,
    accept_token_stride_s: tl.constexpr,
    NUM_SPEC_STEPS: tl.constexpr,
    MAX_TREE_NODES: tl.constexpr,
    HAS_TARGET_MASK: tl.constexpr,
    LINEAR_KV_SAFE: tl.constexpr,
):
    req_idx = tl.program_id(0)

    valid = tl.load(tree_valid_ptr + req_idx * tree_valid_stride)
    last_accepted = tl.load(
        retrieve_index_ptr
        + req_idx * retrieve_index_stride_b
        + 0 * retrieve_index_stride_w
    ).to(tl.int64)
    last_accepted = tl.where(valid, last_accepted, 0)

    root_token = tl.load(
        target_predict_ptr
        + req_idx * target_predict_stride_b
        + last_accepted * target_predict_stride_w
    )
    tl.store(
        accept_index_ptr
        + req_idx * accept_index_stride_b
        + 0 * accept_index_stride_s,
        last_accepted,
    )
    tl.store(
        accept_token_ptr
        + req_idx * accept_token_stride_b
        + 0 * accept_token_stride_s,
        root_token,
    )
    invalid_root_token = tl.load(target_predict_ptr + req_idx * target_predict_stride_b)
    tl.store(
        predicts_ptr + req_idx * predicts_stride_b,
        invalid_root_token,
        mask=~valid,
    )

    active = valid
    cur_index = tl.full((), 0, tl.int64)
    num_accepted = tl.full((), 0, tl.int64)

    for output_idx in range(1, NUM_SPEC_STEPS):
        safe_cur_index = tl.maximum(cur_index, 0)
        next_child = tl.load(
            retrieve_next_token_ptr
            + req_idx * retrieve_next_token_stride_b
            + safe_cur_index * retrieve_next_token_stride_w
        ).to(tl.int64)
        cur_index = tl.where(active, next_child, cur_index)

        found = tl.full((), False, tl.int1)
        accepted_local_idx = tl.full((), 0, tl.int64)
        next_cur_index = cur_index

        for _ in range(MAX_TREE_NODES):
            sibling_active = active & (~found) & (next_cur_index >= 0)
            safe_sibling_idx = tl.maximum(next_cur_index, 0)
            draft_local_idx = tl.load(
                retrieve_index_ptr
                + req_idx * retrieve_index_stride_b
                + safe_sibling_idx * retrieve_index_stride_w,
                mask=sibling_active,
                other=0,
            ).to(tl.int64)
            draft_token_id = tl.load(
                candidates_ptr
                + req_idx * candidates_stride_b
                + safe_sibling_idx * candidates_stride_w,
                mask=sibling_active,
                other=-1,
            )
            target_token_id = tl.load(
                target_predict_ptr
                + req_idx * target_predict_stride_b
                + last_accepted * target_predict_stride_w
            )
            if HAS_TARGET_MASK:
                allowed_by_mask = tl.load(
                    target_mask_ptr
                    + req_idx * target_mask_stride_b
                    + safe_sibling_idx * target_mask_stride_w,
                    mask=sibling_active,
                    other=0,
                ) != 0
            else:
                allowed_by_mask = tl.full((), True, tl.int1)
            if LINEAR_KV_SAFE:
                next_linear_local_idx = num_accepted + 1
                kv_slot_is_linear_prefix = (
                    (draft_local_idx == next_linear_local_idx)
                    & (safe_sibling_idx == next_linear_local_idx)
                )
            else:
                kv_slot_is_linear_prefix = tl.full((), True, tl.int1)

            match = (
                sibling_active
                & (draft_token_id == target_token_id)
                & allowed_by_mask
                & kv_slot_is_linear_prefix
            )
            accepted_local_idx = tl.where(
                match,
                draft_local_idx,
                accepted_local_idx,
            )
            next_sibling = tl.load(
                retrieve_next_sibling_ptr
                + req_idx * retrieve_next_sibling_stride_b
                + safe_sibling_idx * retrieve_next_sibling_stride_w,
                mask=sibling_active,
                other=next_cur_index,
            ).to(tl.int64)
            next_cur_index = tl.where(
                match,
                safe_sibling_idx,
                tl.where(sibling_active, next_sibling, next_cur_index),
            )
            found = found | match

        matched = active & found
        prev_target_token_id = tl.load(
            target_predict_ptr
            + req_idx * target_predict_stride_b
            + last_accepted * target_predict_stride_w
        )
        tl.store(
            predicts_ptr
            + req_idx * predicts_stride_b
            + last_accepted * predicts_stride_w,
            prev_target_token_id,
            mask=matched,
        )
        num_accepted = tl.where(matched, num_accepted + 1, num_accepted)
        accepted_token = tl.load(
            target_predict_ptr
            + req_idx * target_predict_stride_b
            + accepted_local_idx * target_predict_stride_w,
            mask=matched,
            other=0,
        )
        tl.store(
            accept_index_ptr
            + req_idx * accept_index_stride_b
            + output_idx * accept_index_stride_s,
            accepted_local_idx,
            mask=matched,
        )
        tl.store(
            accept_token_ptr
            + req_idx * accept_token_stride_b
            + output_idx * accept_token_stride_s,
            accepted_token,
            mask=matched,
        )
        last_accepted = tl.where(matched, accepted_local_idx, last_accepted)
        active = matched
        cur_index = next_cur_index

    final_token = tl.load(
        target_predict_ptr
        + req_idx * target_predict_stride_b
        + last_accepted * target_predict_stride_w
    )
    tl.store(
        predicts_ptr
        + req_idx * predicts_stride_b
        + last_accepted * predicts_stride_w,
        final_token,
        mask=valid,
    )
    tl.store(accept_token_num_ptr + req_idx, num_accepted)


def verify_dynamic_tree_greedy_kernel(
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
    *,
    num_spec_steps: int,
    tree_valid: torch.Tensor | None = None,
    linear_kv_safe: bool = False,
    target_mask: torch.Tensor | None = None,
) -> DynamicTreeVerifyOutput:
    """Triton implementation of :func:`verify_dynamic_tree_greedy`."""

    if candidates.device.type != "cuda" or not HAS_TRITON:
        return verify_dynamic_tree_greedy(
            candidates,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            target_predict,
            num_spec_steps=num_spec_steps,
            tree_valid=tree_valid,
            linear_kv_safe=linear_kv_safe,
            target_mask=target_mask,
        )

    if candidates.ndim != 2:
        raise ValueError(f"candidates must be 2D, got {candidates.shape}")
    if target_predict.shape != candidates.shape:
        raise ValueError(
            "target_predict must have the same shape as candidates, got "
            f"{target_predict.shape} and {candidates.shape}"
        )
    for name, tensor in (
        ("retrieve_index", retrieve_index),
        ("retrieve_next_token", retrieve_next_token),
        ("retrieve_next_sibling", retrieve_next_sibling),
    ):
        if tensor.shape != candidates.shape:
            raise ValueError(
                f"{name} must have the same shape as candidates, got "
                f"{tensor.shape} and {candidates.shape}"
            )
    if num_spec_steps <= 0:
        raise ValueError(f"num_spec_steps must be positive, got {num_spec_steps}")

    batch_size, num_draft_tokens = candidates.shape
    device = candidates.device
    if tree_valid is None:
        tree_valid = torch.ones(batch_size, dtype=torch.bool, device=device)
    elif tree_valid.shape != (batch_size,):
        raise ValueError(
            f"tree_valid must have shape ({batch_size},), got {tree_valid.shape}"
        )
    if target_mask is not None and target_mask.shape != (batch_size, num_draft_tokens):
        raise ValueError(
            f"target_mask must have shape ({batch_size}, {num_draft_tokens}), "
            f"got {target_mask.shape}"
        )

    predicts = torch.zeros_like(target_predict)
    accept_index = torch.zeros(
        (batch_size, num_spec_steps), dtype=torch.int64, device=device
    )
    accept_token_num = torch.zeros(batch_size, dtype=torch.int64, device=device)
    accept_token = torch.zeros(
        (batch_size, num_spec_steps), dtype=torch.int64, device=device
    )
    target_mask_arg = target_mask if target_mask is not None else candidates

    _verify_dynamic_tree_greedy_triton_kernel[(batch_size,)](
        candidates,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        target_predict,
        tree_valid,
        target_mask_arg,
        predicts,
        accept_index,
        accept_token_num,
        accept_token,
        candidates.stride(0),
        candidates.stride(1),
        retrieve_index.stride(0),
        retrieve_index.stride(1),
        retrieve_next_token.stride(0),
        retrieve_next_token.stride(1),
        retrieve_next_sibling.stride(0),
        retrieve_next_sibling.stride(1),
        target_predict.stride(0),
        target_predict.stride(1),
        tree_valid.stride(0),
        target_mask_arg.stride(0),
        target_mask_arg.stride(1),
        predicts.stride(0),
        predicts.stride(1),
        accept_index.stride(0),
        accept_index.stride(1),
        accept_token.stride(0),
        accept_token.stride(1),
        NUM_SPEC_STEPS=num_spec_steps,
        MAX_TREE_NODES=num_draft_tokens,
        HAS_TARGET_MASK=target_mask is not None,
        LINEAR_KV_SAFE=linear_kv_safe,
    )

    return DynamicTreeVerifyOutput(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        accept_token=accept_token,
    )


def verify_dynamic_tree_greedy_from_draft(
    draft_tree: DynamicTreeDraftOutput,
    target_predict: torch.Tensor,
    *,
    tree_valid: torch.Tensor | None = None,
) -> DynamicTreeVerifyOutput:
    """Run greedy tree verification on a logits-built dynamic tree."""

    candidates = draft_tree.candidates()
    if target_predict.shape != candidates.shape:
        raise ValueError(
            "target_predict must have root-inclusive candidate shape "
            f"{candidates.shape}, got {target_predict.shape}"
        )
    build = draft_tree.build_output
    return verify_dynamic_tree_greedy(
        candidates,
        build.retrieve_index,
        build.retrieve_next_token,
        build.retrieve_next_sibling,
        target_predict,
        num_spec_steps=draft_tree.depth + 1,
        tree_valid=tree_valid,
    )


def verify_dynamic_tree_greedy(
    candidates: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
    *,
    num_spec_steps: int,
    tree_valid: torch.Tensor | None = None,
    linear_kv_safe: bool = False,
    target_mask: torch.Tensor | None = None,
) -> DynamicTreeVerifyOutput:
    """Verify a dynamic draft tree using greedy target predictions."""

    if candidates.ndim != 2:
        raise ValueError(f"candidates must be 2D, got {candidates.shape}")
    if target_predict.shape != candidates.shape:
        raise ValueError(
            "target_predict must have the same shape as candidates, got "
            f"{target_predict.shape} and {candidates.shape}"
        )
    for name, tensor in (
        ("retrieve_index", retrieve_index),
        ("retrieve_next_token", retrieve_next_token),
        ("retrieve_next_sibling", retrieve_next_sibling),
    ):
        if tensor.shape != candidates.shape:
            raise ValueError(
                f"{name} must have the same shape as candidates, got "
                f"{tensor.shape} and {candidates.shape}"
            )
    if num_spec_steps <= 0:
        raise ValueError(f"num_spec_steps must be positive, got {num_spec_steps}")

    batch_size, num_draft_tokens = candidates.shape
    device = candidates.device
    if tree_valid is None:
        tree_valid = torch.ones(batch_size, dtype=torch.bool, device=device)
    elif tree_valid.shape != (batch_size,):
        raise ValueError(
            f"tree_valid must have shape ({batch_size},), got {tree_valid.shape}"
        )
    if target_mask is not None and target_mask.shape != (batch_size, num_draft_tokens):
        raise ValueError(
            f"target_mask must have shape ({batch_size}, {num_draft_tokens}), "
            f"got {target_mask.shape}"
        )

    predicts = torch.zeros_like(target_predict)
    accept_index = torch.zeros(
        (batch_size, num_spec_steps), dtype=torch.int64, device=device
    )
    accept_token_num = torch.zeros(batch_size, dtype=torch.int64, device=device)
    accept_token = torch.zeros(
        (batch_size, num_spec_steps), dtype=torch.int64, device=device
    )

    batch_arange = torch.arange(batch_size, dtype=torch.int64, device=device)
    last_accepted_local_idx = retrieve_index[:, 0].to(torch.int64)
    active = tree_valid.to(torch.bool)
    accept_index[:, 0] = torch.where(
        active,
        last_accepted_local_idx,
        torch.zeros_like(last_accepted_local_idx),
    )
    accept_token[:, 0] = target_predict[
        batch_arange,
        accept_index[:, 0],
    ]
    predicts[~active, 0] = target_predict[~active, 0]
    cur_index = torch.zeros(batch_size, dtype=torch.int64, device=device)
    num_accepted_tokens = torch.zeros(batch_size, dtype=torch.int64, device=device)

    if target_mask is None:
        target_mask = torch.ones(
            (batch_size, num_draft_tokens), dtype=torch.bool, device=device
        )
    else:
        target_mask = target_mask.to(torch.bool)

    max_sibling_steps = num_draft_tokens
    for output_idx in range(1, num_spec_steps):
        cur_index = torch.where(
            active,
            retrieve_next_token[
                batch_arange,
                cur_index.clamp_min(0),
            ].to(torch.int64),
            cur_index,
        )
        found = torch.zeros(batch_size, dtype=torch.bool, device=device)
        accepted_draft_local_idx = torch.zeros(
            batch_size, dtype=torch.int64, device=device
        )
        next_cur_index = cur_index

        for _ in range(max_sibling_steps):
            sibling_active = active & ~found & (next_cur_index >= 0)
            if not bool(sibling_active.any().item()):
                break
            safe_cur_index = next_cur_index.clamp_min(0)
            draft_local_idx = retrieve_index[
                batch_arange,
                safe_cur_index,
            ].to(torch.int64)
            draft_token_id = candidates[batch_arange, safe_cur_index]
            target_token_id = target_predict[
                batch_arange,
                last_accepted_local_idx,
            ]
            allowed_by_mask = target_mask[batch_arange, safe_cur_index]
            if linear_kv_safe:
                next_linear_local_idx = num_accepted_tokens + 1
                kv_slot_is_linear_prefix = (
                    (draft_local_idx == next_linear_local_idx)
                    & (safe_cur_index == next_linear_local_idx)
                )
            else:
                kv_slot_is_linear_prefix = torch.ones_like(allowed_by_mask)

            match = (
                sibling_active
                & (draft_token_id == target_token_id)
                & allowed_by_mask
                & kv_slot_is_linear_prefix
            )
            accepted_draft_local_idx = torch.where(
                match,
                draft_local_idx,
                accepted_draft_local_idx,
            )
            next_sibling = retrieve_next_sibling[
                batch_arange,
                safe_cur_index,
            ].to(torch.int64)
            next_cur_index = torch.where(
                match,
                safe_cur_index,
                torch.where(sibling_active, next_sibling, next_cur_index),
            )
            found |= match

        target_token_id = target_predict[
            batch_arange,
            last_accepted_local_idx,
        ]
        predicts[
            batch_arange[found],
            last_accepted_local_idx[found],
        ] = target_token_id[found]
        matched = active & found
        num_accepted_tokens = torch.where(
            matched,
            num_accepted_tokens + 1,
            num_accepted_tokens,
        )
        accept_index[:, output_idx] = torch.where(
            matched,
            accepted_draft_local_idx,
            accept_index[:, output_idx],
        )
        accept_token[:, output_idx] = torch.where(
            matched,
            target_predict[
                batch_arange,
                accepted_draft_local_idx,
            ],
            accept_token[:, output_idx],
        )
        last_accepted_local_idx = torch.where(
            matched,
            accepted_draft_local_idx,
            last_accepted_local_idx,
        )
        active = matched
        cur_index = next_cur_index
        if not bool(active.any().item()):
            break

    accept_token_num = num_accepted_tokens
    predicts[
        batch_arange[tree_valid],
        last_accepted_local_idx[tree_valid],
    ] = target_predict[
        batch_arange[tree_valid],
        last_accepted_local_idx[tree_valid],
    ]

    return DynamicTreeVerifyOutput(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        accept_token=accept_token,
    )
