# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Reference dynamic draft tree helpers for speculative decoding.

This module mirrors the tensor contract used by TensorRT-LLM's dynamic tree
CUDA kernels, but intentionally keeps the implementation in torch/Python.  It
is meant to pin down the build and greedy-verification semantics before these
operations are moved to vLLM-native CUDA or Triton kernels.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DynamicTreeBuildOutput:
    """Outputs needed by tree attention and tree-aware verification.

    All tensors include the root node at local index 0.  Draft nodes occupy
    local indices ``1..num_draft_tokens - 1`` in the order given by
    ``selected_index``.
    """

    # [batch_size, num_draft_tokens, num_draft_tokens], int32 0/1.
    tree_mask: torch.Tensor
    # [batch_size, num_draft_tokens], int32 depth offsets from the root.
    positions: torch.Tensor
    # [batch_size, num_draft_tokens], int32 local index indirection.
    retrieve_index: torch.Tensor
    # [batch_size, num_draft_tokens], int32 first-child linked-list pointer.
    retrieve_next_token: torch.Tensor
    # [batch_size, num_draft_tokens], int32 next-sibling linked-list pointer.
    retrieve_next_sibling: torch.Tensor


@dataclass(frozen=True)
class DynamicTreeVerifyOutput:
    """Greedy tree verification result.

    ``accept_token_num`` counts accepted draft tokens and does not include the
    root/bonus token stored at ``accept_token[:, 0]``.
    """

    # [batch_size, num_draft_tokens], target predictions written at accepted
    # local positions and at the final bonus position.
    predicts: torch.Tensor
    # [batch_size, num_spec_steps], local indices of root + accepted path.
    accept_index: torch.Tensor
    # [batch_size], number of accepted draft tokens.
    accept_token_num: torch.Tensor
    # [batch_size, num_spec_steps], target token for root/accepted/bonus slots.
    accept_token: torch.Tensor


def build_dynamic_tree(
    parent_list: torch.Tensor,
    selected_index: torch.Tensor,
    *,
    top_k: int,
    depth: int,
) -> DynamicTreeBuildOutput:
    """Build dynamic tree metadata from TRT-style parent/index buffers.

    Args:
        parent_list: ``[B, top_k * (depth - 1) + 1]`` int tensor.  Entries map
            history table rows to their selected parent token index.
        selected_index: ``[B, N - 1]`` int tensor.  The chosen history table
            indices for the final tree, excluding the root.
        top_k: Maximum top-k branch count per draft layer.
        depth: Maximum draft tree depth.

    Returns:
        DynamicTreeBuildOutput with root-inclusive tensors.  The output matches
        the non-packed TRT kernel convention: mask rows can attend to the root,
        themselves, and their ancestors.
    """

    if parent_list.ndim != 2:
        raise ValueError(f"parent_list must be 2D, got {parent_list.shape}")
    if selected_index.ndim != 2:
        raise ValueError(f"selected_index must be 2D, got {selected_index.shape}")
    if parent_list.shape[0] != selected_index.shape[0]:
        raise ValueError(
            "parent_list and selected_index must have the same batch size, "
            f"got {parent_list.shape[0]} and {selected_index.shape[0]}"
        )
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if depth <= 0:
        raise ValueError(f"depth must be positive, got {depth}")
    expected_parent_width = top_k * (depth - 1) + 1
    if parent_list.shape[1] < expected_parent_width:
        raise ValueError(
            "parent_list is too narrow for top_k/depth: "
            f"need at least {expected_parent_width}, got {parent_list.shape[1]}"
        )

    device = selected_index.device
    batch_size = selected_index.shape[0]
    num_draft_tokens = selected_index.shape[1] + 1

    tree_mask = torch.zeros(
        (batch_size, num_draft_tokens, num_draft_tokens),
        dtype=torch.int32,
        device=device,
    )
    positions = torch.zeros(
        (batch_size, num_draft_tokens), dtype=torch.int32, device=device
    )
    retrieve_index = (
        torch.arange(num_draft_tokens, dtype=torch.int32, device=device)
        .expand(batch_size, -1)
        .clone()
    )
    retrieve_next_token = torch.full(
        (batch_size, num_draft_tokens), -1, dtype=torch.int32, device=device
    )
    retrieve_next_sibling = torch.full(
        (batch_size, num_draft_tokens), -1, dtype=torch.int32, device=device
    )

    for batch_idx in range(batch_size):
        selected = selected_index[batch_idx].tolist()
        parent = parent_list[batch_idx].tolist()
        selected_to_position = {
            int(history_idx): pos + 1 for pos, history_idx in enumerate(selected)
        }

        # Every node can attend to the root.  Non-root nodes also attend to
        # themselves and ancestors discovered via parent_list.
        tree_mask[batch_idx, :, 0] = 1

        for local_idx in range(num_draft_tokens - 1, 0, -1):
            parent_position = _find_parent_position(
                selected[local_idx - 1],
                parent,
                selected_to_position,
                top_k,
            )
            if parent_position is None:
                continue

            old_first_child = retrieve_next_token[batch_idx, parent_position].item()
            retrieve_next_token[batch_idx, parent_position] = local_idx
            if old_first_child != -1:
                retrieve_next_sibling[batch_idx, local_idx] = old_first_child

        for local_idx in range(1, num_draft_tokens):
            position = 0
            selected_pos = local_idx - 1
            while position < depth + 1:
                position += 1
                tree_mask[batch_idx, local_idx, selected_pos + 1] = 1

                parent_table_idx = int(selected[selected_pos]) // top_k
                if parent_table_idx == 0:
                    break

                parent_history_idx = int(parent[parent_table_idx])
                parent_position = selected_to_position.get(parent_history_idx)
                if parent_position is None:
                    break
                selected_pos = parent_position - 1
            positions[batch_idx, local_idx] = position

    return DynamicTreeBuildOutput(
        tree_mask=tree_mask,
        positions=positions,
        retrieve_index=retrieve_index,
        retrieve_next_token=retrieve_next_token,
        retrieve_next_sibling=retrieve_next_sibling,
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
) -> DynamicTreeVerifyOutput:
    """Verify a dynamic draft tree using greedy target predictions.

    Starting from the root, each step scans the current node's children in
    linked-list order.  The first child whose candidate token equals the target
    prediction at the last accepted local index is accepted and traversal moves
    to that child.  If no child matches, verification stops and the target
    prediction at the last accepted local index becomes the bonus token.
    """

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

    predicts = torch.zeros_like(target_predict)
    accept_index = torch.zeros(
        (batch_size, num_spec_steps), dtype=torch.int64, device=device
    )
    accept_token_num = torch.zeros(batch_size, dtype=torch.int64, device=device)
    accept_token = torch.zeros(
        (batch_size, num_spec_steps), dtype=torch.int64, device=device
    )

    for batch_idx in range(batch_size):
        if not bool(tree_valid[batch_idx].item()):
            accept_index[batch_idx, 0] = 0
            accept_token[batch_idx, 0] = target_predict[batch_idx, 0]
            predicts[batch_idx, 0] = target_predict[batch_idx, 0]
            continue

        last_accepted_local_idx = int(retrieve_index[batch_idx, 0].item())
        accept_index[batch_idx, 0] = last_accepted_local_idx
        accept_token[batch_idx, 0] = target_predict[batch_idx, last_accepted_local_idx]
        cur_index = 0
        num_accepted_tokens = 0

        for _ in range(1, num_spec_steps):
            cur_index = int(retrieve_next_token[batch_idx, cur_index].item())

            while cur_index != -1:
                draft_local_idx = int(retrieve_index[batch_idx, cur_index].item())
                draft_token_id = candidates[batch_idx, cur_index]
                target_token_id = target_predict[batch_idx, last_accepted_local_idx]

                if bool((draft_token_id == target_token_id).item()):
                    predicts[batch_idx, last_accepted_local_idx] = target_token_id
                    num_accepted_tokens += 1
                    accept_index[batch_idx, num_accepted_tokens] = draft_local_idx
                    if num_accepted_tokens < num_spec_steps:
                        accept_token[batch_idx, num_accepted_tokens] = target_predict[
                            batch_idx, draft_local_idx
                        ]
                    last_accepted_local_idx = draft_local_idx
                    break

                cur_index = int(retrieve_next_sibling[batch_idx, cur_index].item())

            if cur_index == -1:
                break

        accept_token_num[batch_idx] = num_accepted_tokens
        predicts[batch_idx, last_accepted_local_idx] = target_predict[
            batch_idx, last_accepted_local_idx
        ]

    return DynamicTreeVerifyOutput(
        predicts=predicts,
        accept_index=accept_index,
        accept_token_num=accept_token_num,
        accept_token=accept_token,
    )


def _find_parent_position(
    selected_history_idx: int,
    parent_list: list[int],
    selected_to_position: dict[int, int],
    top_k: int,
) -> int | None:
    parent_table_idx = int(selected_history_idx) // top_k
    if parent_table_idx == 0:
        return 0
    if parent_table_idx >= len(parent_list):
        return None
    return selected_to_position.get(int(parent_list[parent_table_idx]))
