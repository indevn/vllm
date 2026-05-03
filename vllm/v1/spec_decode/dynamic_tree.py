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
class DynamicTreeDraftOutput:
    """Logits-driven dynamic draft tree selection result.

    ``draft_token_ids`` and ``selected_index`` exclude the synthetic root node.
    ``build_output`` is root-inclusive and can be fed to the greedy verifier.
    """

    # [batch_size, max_total_draft_tokens], selected draft token ids.
    draft_token_ids: torch.Tensor
    # [batch_size, max_total_draft_tokens], cumulative path scores.
    draft_scores: torch.Tensor
    # [batch_size, max_total_draft_tokens], selected history table rows.
    selected_index: torch.Tensor
    # [batch_size, top_k * (depth - 1) + 1], TRT-style parent table.
    parent_list: torch.Tensor
    # [batch_size, history_size], all tokens considered during expansion.
    history_draft_token_ids: torch.Tensor
    # [batch_size, history_size], cumulative score for each history row.
    history_scores: torch.Tensor
    build_output: DynamicTreeBuildOutput
    top_k: int
    depth: int

    def candidates(self, root_token_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Return root-inclusive candidate tokens for tree verification."""
        batch_size = self.draft_token_ids.shape[0]
        if root_token_ids is None:
            root_token_ids = torch.zeros(
                (batch_size, 1),
                dtype=self.draft_token_ids.dtype,
                device=self.draft_token_ids.device,
            )
        elif root_token_ids.ndim == 1:
            root_token_ids = root_token_ids.view(batch_size, 1)
        elif root_token_ids.shape != (batch_size, 1):
            raise ValueError(
                "root_token_ids must have shape "
                f"({batch_size},) or ({batch_size}, 1), got "
                f"{root_token_ids.shape}"
            )
        return torch.cat(
            [root_token_ids.to(self.draft_token_ids), self.draft_token_ids],
            dim=1,
        )


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


class DynamicDraftTreeManager:
    """Greedy-only Dynamic Draft Tree planner.

    This is the Python/Torch control-path equivalent of TRT-LLM's dynamic tree
    draft loop: each layer expands the current top-k frontier, accumulates path
    scores, keeps the best top-k frontier for the next layer, then resamples the
    final tree from all history nodes.
    """

    def __init__(self, *, top_k: int, depth: int, max_total_draft_tokens: int):
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        history_size = _history_size(top_k, depth)
        if max_total_draft_tokens <= 0:
            raise ValueError(
                f"max_total_draft_tokens must be positive, got {max_total_draft_tokens}"
            )
        if max_total_draft_tokens > history_size:
            raise ValueError(
                "max_total_draft_tokens cannot exceed dynamic tree history "
                f"size {history_size}, got {max_total_draft_tokens}"
            )
        self.top_k = top_k
        self.depth = depth
        self.max_total_draft_tokens = max_total_draft_tokens

    def build_from_logits(
        self,
        root_logits: torch.Tensor,
        child_logits_by_depth: list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> DynamicTreeDraftOutput:
        """Build a dynamic tree from draft logits.

        Args:
            root_logits: ``[B, vocab_size]`` logits for the root expansion.
            child_logits_by_depth: ``depth - 1`` tensors, each shaped
                ``[B, top_k, vocab_size]``.  The top-k dimension is the current
                frontier selected from the previous layer.
        """

        return build_dynamic_tree_from_logits(
            root_logits,
            child_logits_by_depth,
            top_k=self.top_k,
            depth=self.depth,
            max_total_draft_tokens=self.max_total_draft_tokens,
        )

    def verify_greedy(
        self,
        draft_tree: DynamicTreeDraftOutput,
        target_predict: torch.Tensor,
        tree_valid: torch.Tensor | None = None,
    ) -> DynamicTreeVerifyOutput:
        if draft_tree.top_k != self.top_k or draft_tree.depth != self.depth:
            raise ValueError(
                "draft_tree was built with a different DynamicDraftTreeManager "
                f"configuration: got top_k={draft_tree.top_k}, "
                f"depth={draft_tree.depth}; expected top_k={self.top_k}, "
                f"depth={self.depth}"
            )
        return verify_dynamic_tree_greedy_from_draft(
            draft_tree, target_predict, tree_valid=tree_valid
        )


def build_dynamic_tree_from_logits(
    root_logits: torch.Tensor,
    child_logits_by_depth: list[torch.Tensor] | tuple[torch.Tensor, ...],
    *,
    top_k: int,
    depth: int,
    max_total_draft_tokens: int,
) -> DynamicTreeDraftOutput:
    """Select a Dynamic Draft Tree directly from draft logits.

    The algorithm mirrors the TRT-LLM PyTorch backend:

    1. top-k expand the root.
    2. For each later depth, top-k expand each current frontier node.
    3. Accumulate path probabilities and keep the best top-k frontier.
    4. Resample ``max_total_draft_tokens`` nodes from the full history.
    5. Build root-inclusive tree attention and retrieve metadata.
    """

    _validate_logits_inputs(root_logits, child_logits_by_depth, top_k, depth)
    history_size = _history_size(top_k, depth)
    if max_total_draft_tokens <= 0 or max_total_draft_tokens > history_size:
        raise ValueError(
            "max_total_draft_tokens must be in "
            f"[1, {history_size}], got {max_total_draft_tokens}"
        )

    device = root_logits.device
    batch_size = root_logits.shape[0]
    parent_width = top_k * (depth - 1) + 1
    parent_list = torch.full(
        (batch_size, parent_width), -1, dtype=torch.int64, device=device
    )
    history_tokens = torch.zeros(
        (batch_size, history_size), dtype=torch.int64, device=device
    )
    history_scores = torch.full(
        (batch_size, history_size),
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )

    root_tokens, root_scores = _topk_probs(root_logits, top_k)
    history_tokens[:, :top_k] = root_tokens
    history_scores[:, :top_k] = root_scores

    current_scores = root_scores
    current_history_indices = (
        torch.arange(top_k, dtype=torch.int64, device=device)
        .expand(batch_size, -1)
        .clone()
    )

    layer_history_start = top_k
    for layer_idx, child_logits in enumerate(child_logits_by_depth, start=1):
        child_tokens, child_scores = _topk_probs(child_logits, top_k)
        flat_tokens = child_tokens.reshape(batch_size, top_k * top_k)
        flat_scores = (child_scores * current_scores.unsqueeze(2)).reshape(
            batch_size, top_k * top_k
        )

        history_end = layer_history_start + top_k * top_k
        history_tokens[:, layer_history_start:history_end] = flat_tokens
        history_scores[:, layer_history_start:history_end] = flat_scores

        table_start = layer_history_start // top_k
        parent_list[:, table_start : table_start + top_k] = current_history_indices

        next_scores, next_offsets = torch.topk(flat_scores, k=top_k, dim=-1)
        current_scores = next_scores
        current_history_indices = layer_history_start + next_offsets
        layer_history_start = history_end

    selected_index = torch.topk(
        history_scores, k=max_total_draft_tokens, dim=-1
    ).indices
    selected_index = torch.sort(selected_index, dim=-1).values
    draft_token_ids = torch.gather(history_tokens, dim=1, index=selected_index)
    draft_scores = torch.gather(history_scores, dim=1, index=selected_index)

    build_output = build_dynamic_tree(
        parent_list,
        selected_index,
        top_k=top_k,
        depth=depth,
    )
    return DynamicTreeDraftOutput(
        draft_token_ids=draft_token_ids,
        draft_scores=draft_scores,
        selected_index=selected_index,
        parent_list=parent_list,
        history_draft_token_ids=history_tokens,
        history_scores=history_scores,
        build_output=build_output,
        top_k=top_k,
        depth=depth,
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
    linear_kv_safe: bool = False,
) -> DynamicTreeVerifyOutput:
    """Verify a dynamic draft tree using greedy target predictions.

    Starting from the root, each step scans the current node's children in
    linked-list order.  The first child whose candidate token equals the target
    prediction at the last accepted local index is accepted and traversal moves
    to that child.  If no child matches, verification stops and the target
    prediction at the last accepted local index becomes the bonus token.

    When ``linear_kv_safe`` is set, verification additionally requires the
    accepted tree node to be the next contiguous node in the flattened draft
    order.  This is the safe runtime bridge before paged KV relocation exists:
    branching candidates can still be inspected, but accepting a non-prefix
    tree slot would leave the target KV cache in the wrong linear layout.
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
                next_linear_local_idx = num_accepted_tokens + 1
                kv_slot_is_linear_prefix = (
                    draft_local_idx == next_linear_local_idx
                    and cur_index == next_linear_local_idx
                )

                if bool((draft_token_id == target_token_id).item()) and (
                    not linear_kv_safe or kv_slot_is_linear_prefix
                ):
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


def _history_size(top_k: int, depth: int) -> int:
    if depth <= 0:
        return 0
    return top_k + top_k * top_k * (depth - 1)


def _validate_logits_inputs(
    root_logits: torch.Tensor,
    child_logits_by_depth: list[torch.Tensor] | tuple[torch.Tensor, ...],
    top_k: int,
    depth: int,
) -> None:
    if root_logits.ndim != 2:
        raise ValueError(f"root_logits must be 2D, got {root_logits.shape}")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if depth <= 0:
        raise ValueError(f"depth must be positive, got {depth}")
    if root_logits.shape[1] < top_k:
        raise ValueError(
            f"root vocab size {root_logits.shape[1]} is smaller than top_k={top_k}"
        )
    if len(child_logits_by_depth) != depth - 1:
        raise ValueError(
            "child_logits_by_depth must contain depth - 1 tensors, got "
            f"{len(child_logits_by_depth)} for depth={depth}"
        )
    for layer_idx, child_logits in enumerate(child_logits_by_depth, start=1):
        expected_shape = (root_logits.shape[0], top_k)
        if child_logits.ndim != 3 or child_logits.shape[:2] != expected_shape:
            raise ValueError(
                f"child logits at depth {layer_idx} must have shape "
                f"({root_logits.shape[0]}, {top_k}, vocab_size), got "
                f"{child_logits.shape}"
            )
        if child_logits.shape[2] < top_k:
            raise ValueError(
                f"child vocab size {child_logits.shape[2]} at depth "
                f"{layer_idx} is smaller than top_k={top_k}"
            )


def _topk_probs(logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    probs = torch.softmax(logits.to(torch.float32), dim=-1)
    values, indices = torch.topk(probs, k=top_k, dim=-1)
    return indices.to(torch.int64), values
