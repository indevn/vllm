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

from vllm.triton_utils import HAS_TRITON, tl, triton


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


@dataclass(frozen=True)
class StaticTopKCompactMetadataOutput:
    """Fixed-width compact metadata built from static-tree top-k nodes.

    This is the oracle contract for the DDT selected-subtree handoff kernel.
    All metadata tensors are root-inclusive.  ``selected_static_nodes`` and
    ``selected_token_ids`` exclude the root and are padded to
    ``max_tree_nodes - 1``.
    """

    selected_static_nodes: torch.Tensor
    selected_token_ids: torch.Tensor
    retrieve_index: torch.Tensor
    retrieve_next_token: torch.Tensor
    retrieve_next_sibling: torch.Tensor
    parent: torch.Tensor
    target_mask: torch.Tensor
    position_offsets: torch.Tensor
    tree_valid: torch.Tensor
    num_nodes: torch.Tensor
    num_spec_steps: torch.Tensor


def build_static_topk_compact_metadata(
    topk_static_nodes: torch.Tensor,
    valid_topk: torch.Tensor,
    static_tokens: torch.Tensor,
    static_parent_indices: torch.Tensor,
    static_position_offsets: torch.Tensor,
    *,
    max_tree_nodes: int | None = None,
) -> StaticTopKCompactMetadataOutput:
    """Build compact DDT metadata from static-tree top-k selections.

    The output mirrors ``EagleProposer._get_dynamic_tree_metadata_template``
    but returns fixed-width tensors that are suitable for a later GPU-side
    compaction kernel.  The reference implementation intentionally uses Python
    loops so the tensor contract is easy to inspect and test first.
    """

    if topk_static_nodes.ndim != 2:
        raise ValueError(
            f"topk_static_nodes must be 2D, got {topk_static_nodes.shape}"
        )
    if valid_topk.shape != topk_static_nodes.shape:
        raise ValueError(
            "valid_topk must have the same shape as topk_static_nodes, got "
            f"{valid_topk.shape} and {topk_static_nodes.shape}"
        )
    if static_tokens.ndim != 2:
        raise ValueError(f"static_tokens must be 2D, got {static_tokens.shape}")
    if static_tokens.shape[0] != topk_static_nodes.shape[0]:
        raise ValueError(
            "static_tokens and topk_static_nodes batch size mismatch: "
            f"{static_tokens.shape[0]} vs {topk_static_nodes.shape[0]}"
        )
    if static_parent_indices.ndim != 1:
        raise ValueError(
            "static_parent_indices must be 1D, got "
            f"{static_parent_indices.shape}"
        )
    if static_position_offsets.ndim != 1:
        raise ValueError(
            "static_position_offsets must be 1D, got "
            f"{static_position_offsets.shape}"
        )
    static_width = static_tokens.shape[1]
    if static_parent_indices.shape[0] < static_width:
        raise ValueError(
            "static_parent_indices is narrower than static_tokens: "
            f"{static_parent_indices.shape[0]} < {static_width}"
        )
    if static_position_offsets.shape[0] < static_width:
        raise ValueError(
            "static_position_offsets is narrower than static_tokens: "
            f"{static_position_offsets.shape[0]} < {static_width}"
        )

    batch_size, topk_count = topk_static_nodes.shape
    if max_tree_nodes is None:
        max_tree_nodes = static_width
    if max_tree_nodes <= 0:
        raise ValueError(f"max_tree_nodes must be positive, got {max_tree_nodes}")
    if max_tree_nodes > static_width:
        raise ValueError(
            "max_tree_nodes cannot exceed static tree width including root: "
            f"{max_tree_nodes} > {static_width}"
        )

    device = topk_static_nodes.device
    selected_width = max_tree_nodes - 1
    selected_static_nodes = torch.zeros(
        (batch_size, selected_width), dtype=torch.int32, device=device
    )
    selected_token_ids = torch.zeros(
        (batch_size, selected_width), dtype=static_tokens.dtype, device=device
    )
    retrieve_index = (
        torch.arange(max_tree_nodes, dtype=torch.int32, device=device)
        .expand(batch_size, -1)
        .clone()
    )
    retrieve_next_token = torch.full(
        (batch_size, max_tree_nodes), -1, dtype=torch.int32, device=device
    )
    retrieve_next_sibling = torch.full(
        (batch_size, max_tree_nodes), -1, dtype=torch.int32, device=device
    )
    parent = torch.full(
        (batch_size, max_tree_nodes), -1, dtype=torch.int32, device=device
    )
    target_mask = torch.zeros(
        (batch_size, max_tree_nodes), dtype=torch.int32, device=device
    )
    position_offsets = torch.zeros(
        (batch_size, max_tree_nodes), dtype=torch.int64, device=device
    )
    tree_valid = torch.zeros(batch_size, dtype=torch.bool, device=device)
    num_nodes = torch.ones(batch_size, dtype=torch.int32, device=device)
    num_spec_steps = torch.ones(batch_size, dtype=torch.int32, device=device)

    topk_static_nodes_cpu = topk_static_nodes.detach().cpu().tolist()
    valid_topk_cpu = valid_topk.detach().cpu().tolist()
    static_tokens_cpu = static_tokens.detach().cpu().tolist()
    static_parent_cpu = static_parent_indices.detach().cpu().tolist()
    static_position_cpu = static_position_offsets.detach().cpu().tolist()

    for batch_idx in range(batch_size):
        token_by_static_node: dict[int, int] = {}
        for static_idx, is_valid in zip(
            topk_static_nodes_cpu[batch_idx],
            valid_topk_cpu[batch_idx],
            strict=True,
        ):
            if not bool(is_valid):
                continue
            cur_static_idx = int(static_idx)
            while cur_static_idx > 0:
                if cur_static_idx >= static_width:
                    break
                token_by_static_node[cur_static_idx] = int(
                    static_tokens_cpu[batch_idx][cur_static_idx]
                )
                cur_static_idx = int(static_parent_cpu[cur_static_idx])

        if not token_by_static_node:
            continue
        packed_static_nodes = sorted(token_by_static_node)
        row_num_nodes = min(len(packed_static_nodes) + 1, max_tree_nodes)
        packed_static_nodes = packed_static_nodes[: row_num_nodes - 1]
        static_to_packed = {
            static_idx: packed_idx + 1
            for packed_idx, static_idx in enumerate(packed_static_nodes)
        }
        children_by_parent: dict[int, list[int]] = {}
        for packed_idx, static_idx in enumerate(packed_static_nodes, start=1):
            selected_static_nodes[batch_idx, packed_idx - 1] = static_idx
            selected_token_ids[batch_idx, packed_idx - 1] = token_by_static_node[
                static_idx
            ]
            position_offsets[batch_idx, packed_idx] = int(
                static_position_cpu[static_idx]
            )
            parent_static_idx = int(static_parent_cpu[static_idx])
            if parent_static_idx == 0 or parent_static_idx in static_to_packed:
                parent_idx = static_to_packed.get(parent_static_idx, 0)
                parent[batch_idx, packed_idx] = parent_idx
                children_by_parent.setdefault(parent_idx, []).append(packed_idx)

        for parent_idx, child_indices in children_by_parent.items():
            first_child = -1
            for child_idx in sorted(child_indices, reverse=True):
                retrieve_next_sibling[batch_idx, child_idx] = first_child
                first_child = child_idx
            retrieve_next_token[batch_idx, parent_idx] = first_child

        target_mask[batch_idx, :row_num_nodes] = 1
        tree_valid[batch_idx] = True
        num_nodes[batch_idx] = row_num_nodes
        max_depth = max(
            (
                int(static_position_cpu[static_idx])
                for static_idx in packed_static_nodes
            ),
            default=0,
        )
        num_spec_steps[batch_idx] = max_depth + 1

    return StaticTopKCompactMetadataOutput(
        selected_static_nodes=selected_static_nodes,
        selected_token_ids=selected_token_ids,
        retrieve_index=retrieve_index,
        retrieve_next_token=retrieve_next_token,
        retrieve_next_sibling=retrieve_next_sibling,
        parent=parent,
        target_mask=target_mask,
        position_offsets=position_offsets,
        tree_valid=tree_valid,
        num_nodes=num_nodes,
        num_spec_steps=num_spec_steps,
    )


def build_selected_bool_compact_metadata(
    selected_bool: torch.Tensor,
    static_tokens: torch.Tensor,
    static_parent_indices: torch.Tensor,
    static_position_offsets: torch.Tensor,
    *,
    max_tree_nodes: int | None = None,
) -> StaticTopKCompactMetadataOutput:
    """Build compact DDT metadata from an ancestor-closed static bool mask."""

    if selected_bool.ndim != 2:
        raise ValueError(f"selected_bool must be 2D, got {selected_bool.shape}")
    if static_tokens.ndim != 2:
        raise ValueError(f"static_tokens must be 2D, got {static_tokens.shape}")
    if static_tokens.shape != selected_bool.shape:
        raise ValueError(
            "static_tokens and selected_bool shape mismatch: "
            f"{static_tokens.shape} vs {selected_bool.shape}"
        )
    if static_parent_indices.ndim != 1:
        raise ValueError(
            "static_parent_indices must be 1D, got "
            f"{static_parent_indices.shape}"
        )
    if static_position_offsets.ndim != 1:
        raise ValueError(
            "static_position_offsets must be 1D, got "
            f"{static_position_offsets.shape}"
        )
    static_width = static_tokens.shape[1]
    if static_parent_indices.shape[0] < static_width:
        raise ValueError(
            "static_parent_indices is narrower than static_tokens: "
            f"{static_parent_indices.shape[0]} < {static_width}"
        )
    if static_position_offsets.shape[0] < static_width:
        raise ValueError(
            "static_position_offsets is narrower than static_tokens: "
            f"{static_position_offsets.shape[0]} < {static_width}"
        )

    batch_size = selected_bool.shape[0]
    if max_tree_nodes is None:
        max_tree_nodes = static_width
    if max_tree_nodes <= 0:
        raise ValueError(f"max_tree_nodes must be positive, got {max_tree_nodes}")
    if max_tree_nodes > static_width:
        raise ValueError(
            "max_tree_nodes cannot exceed static tree width including root: "
            f"{max_tree_nodes} > {static_width}"
        )

    device = selected_bool.device
    selected_width = max_tree_nodes - 1
    selected_static_nodes = torch.zeros(
        (batch_size, selected_width), dtype=torch.int32, device=device
    )
    selected_token_ids = torch.zeros(
        (batch_size, selected_width), dtype=static_tokens.dtype, device=device
    )
    retrieve_index = (
        torch.arange(max_tree_nodes, dtype=torch.int32, device=device)
        .expand(batch_size, -1)
        .clone()
    )
    retrieve_next_token = torch.full(
        (batch_size, max_tree_nodes), -1, dtype=torch.int32, device=device
    )
    retrieve_next_sibling = torch.full(
        (batch_size, max_tree_nodes), -1, dtype=torch.int32, device=device
    )
    parent = torch.full(
        (batch_size, max_tree_nodes), -1, dtype=torch.int32, device=device
    )
    target_mask = torch.zeros(
        (batch_size, max_tree_nodes), dtype=torch.int32, device=device
    )
    position_offsets = torch.zeros(
        (batch_size, max_tree_nodes), dtype=torch.int64, device=device
    )
    tree_valid = torch.zeros(batch_size, dtype=torch.bool, device=device)
    num_nodes = torch.ones(batch_size, dtype=torch.int32, device=device)
    num_spec_steps = torch.ones(batch_size, dtype=torch.int32, device=device)

    selected_bool_cpu = selected_bool.detach().cpu().tolist()
    static_tokens_cpu = static_tokens.detach().cpu().tolist()
    static_parent_cpu = static_parent_indices.detach().cpu().tolist()
    static_position_cpu = static_position_offsets.detach().cpu().tolist()

    for batch_idx in range(batch_size):
        packed_static_nodes = [
            static_idx
            for static_idx in range(1, static_width)
            if bool(selected_bool_cpu[batch_idx][static_idx])
        ][:selected_width]
        if not packed_static_nodes:
            continue
        row_num_nodes = len(packed_static_nodes) + 1
        static_to_packed = {
            static_idx: packed_idx + 1
            for packed_idx, static_idx in enumerate(packed_static_nodes)
        }
        children_by_parent: dict[int, list[int]] = {}
        for packed_idx, static_idx in enumerate(packed_static_nodes, start=1):
            selected_static_nodes[batch_idx, packed_idx - 1] = static_idx
            selected_token_ids[batch_idx, packed_idx - 1] = int(
                static_tokens_cpu[batch_idx][static_idx]
            )
            position_offsets[batch_idx, packed_idx] = int(
                static_position_cpu[static_idx]
            )
            parent_static_idx = int(static_parent_cpu[static_idx])
            if parent_static_idx == 0 or parent_static_idx in static_to_packed:
                parent_idx = static_to_packed.get(parent_static_idx, 0)
                parent[batch_idx, packed_idx] = parent_idx
                children_by_parent.setdefault(parent_idx, []).append(packed_idx)

        for parent_idx, child_indices in children_by_parent.items():
            first_child = -1
            for child_idx in sorted(child_indices, reverse=True):
                retrieve_next_sibling[batch_idx, child_idx] = first_child
                first_child = child_idx
            retrieve_next_token[batch_idx, parent_idx] = first_child

        target_mask[batch_idx, :row_num_nodes] = 1
        tree_valid[batch_idx] = True
        num_nodes[batch_idx] = row_num_nodes
        max_depth = max(
            (
                int(static_position_cpu[static_idx])
                for static_idx in packed_static_nodes
            ),
            default=0,
        )
        num_spec_steps[batch_idx] = max_depth + 1

    return StaticTopKCompactMetadataOutput(
        selected_static_nodes=selected_static_nodes,
        selected_token_ids=selected_token_ids,
        retrieve_index=retrieve_index,
        retrieve_next_token=retrieve_next_token,
        retrieve_next_sibling=retrieve_next_sibling,
        parent=parent,
        target_mask=target_mask,
        position_offsets=position_offsets,
        tree_valid=tree_valid,
        num_nodes=num_nodes,
        num_spec_steps=num_spec_steps,
    )


@triton.jit
def _build_static_topk_compact_metadata_triton_kernel(
    topk_static_nodes_ptr,
    valid_topk_ptr,
    static_tokens_ptr,
    static_parent_indices_ptr,
    static_position_offsets_ptr,
    selected_static_nodes_ptr,
    selected_token_ids_ptr,
    retrieve_index_ptr,
    retrieve_next_token_ptr,
    retrieve_next_sibling_ptr,
    parent_ptr,
    target_mask_ptr,
    position_offsets_ptr,
    tree_valid_ptr,
    num_nodes_ptr,
    num_spec_steps_ptr,
    topk_static_nodes_stride_b: tl.constexpr,
    topk_static_nodes_stride_k: tl.constexpr,
    valid_topk_stride_b: tl.constexpr,
    valid_topk_stride_k: tl.constexpr,
    static_tokens_stride_b: tl.constexpr,
    static_tokens_stride_w: tl.constexpr,
    selected_static_nodes_stride_b: tl.constexpr,
    selected_static_nodes_stride_w: tl.constexpr,
    selected_token_ids_stride_b: tl.constexpr,
    selected_token_ids_stride_w: tl.constexpr,
    retrieve_index_stride_b: tl.constexpr,
    retrieve_index_stride_w: tl.constexpr,
    retrieve_next_token_stride_b: tl.constexpr,
    retrieve_next_token_stride_w: tl.constexpr,
    retrieve_next_sibling_stride_b: tl.constexpr,
    retrieve_next_sibling_stride_w: tl.constexpr,
    parent_stride_b: tl.constexpr,
    parent_stride_w: tl.constexpr,
    target_mask_stride_b: tl.constexpr,
    target_mask_stride_w: tl.constexpr,
    position_offsets_stride_b: tl.constexpr,
    position_offsets_stride_w: tl.constexpr,
    TOPK_COUNT: tl.constexpr,
    STATIC_WIDTH: tl.constexpr,
    MAX_TREE_NODES: tl.constexpr,
):
    req_idx = tl.program_id(0)

    for local_idx in range(MAX_TREE_NODES):
        tl.store(
            retrieve_index_ptr
            + req_idx * retrieve_index_stride_b
            + local_idx * retrieve_index_stride_w,
            local_idx,
        )
        tl.store(
            retrieve_next_token_ptr
            + req_idx * retrieve_next_token_stride_b
            + local_idx * retrieve_next_token_stride_w,
            -1,
        )
        tl.store(
            retrieve_next_sibling_ptr
            + req_idx * retrieve_next_sibling_stride_b
            + local_idx * retrieve_next_sibling_stride_w,
            -1,
        )
        tl.store(
            parent_ptr + req_idx * parent_stride_b + local_idx * parent_stride_w,
            -1,
        )
        tl.store(
            target_mask_ptr
            + req_idx * target_mask_stride_b
            + local_idx * target_mask_stride_w,
            0,
        )
        tl.store(
            position_offsets_ptr
            + req_idx * position_offsets_stride_b
            + local_idx * position_offsets_stride_w,
            0,
        )

    for selected_pos in range(MAX_TREE_NODES - 1):
        tl.store(
            selected_static_nodes_ptr
            + req_idx * selected_static_nodes_stride_b
            + selected_pos * selected_static_nodes_stride_w,
            0,
        )
        tl.store(
            selected_token_ids_ptr
            + req_idx * selected_token_ids_stride_b
            + selected_pos * selected_token_ids_stride_w,
            0,
        )

    packed_count = tl.full((), 0, tl.int64)
    max_depth = tl.full((), 0, tl.int64)

    for static_idx in range(1, STATIC_WIDTH):
        selected = tl.full((), False, tl.int1)
        for topk_idx in range(TOPK_COUNT):
            valid = tl.load(
                valid_topk_ptr
                + req_idx * valid_topk_stride_b
                + topk_idx * valid_topk_stride_k
            )
            cur_static_idx = tl.load(
                topk_static_nodes_ptr
                + req_idx * topk_static_nodes_stride_b
                + topk_idx * topk_static_nodes_stride_k
            ).to(tl.int64)
            for _ in range(STATIC_WIDTH):
                active = valid & (cur_static_idx > 0) & (
                    cur_static_idx < STATIC_WIDTH
                )
                selected = selected | (active & (cur_static_idx == static_idx))
                cur_static_idx = tl.load(
                    static_parent_indices_ptr + cur_static_idx,
                    mask=active,
                    other=0,
                ).to(tl.int64)

        can_pack = selected & (packed_count < MAX_TREE_NODES - 1)
        packed_local_idx = packed_count + 1
        token_id = tl.load(
            static_tokens_ptr
            + req_idx * static_tokens_stride_b
            + static_idx * static_tokens_stride_w,
            mask=can_pack,
            other=0,
        )
        static_position = tl.load(
            static_position_offsets_ptr + static_idx,
            mask=can_pack,
            other=0,
        ).to(tl.int64)
        tl.store(
            selected_static_nodes_ptr
            + req_idx * selected_static_nodes_stride_b
            + packed_count * selected_static_nodes_stride_w,
            static_idx,
            mask=can_pack,
        )
        tl.store(
            selected_token_ids_ptr
            + req_idx * selected_token_ids_stride_b
            + packed_count * selected_token_ids_stride_w,
            token_id,
            mask=can_pack,
        )
        tl.store(
            target_mask_ptr
            + req_idx * target_mask_stride_b
            + packed_local_idx * target_mask_stride_w,
            1,
            mask=can_pack,
        )
        tl.store(
            position_offsets_ptr
            + req_idx * position_offsets_stride_b
            + packed_local_idx * position_offsets_stride_w,
            static_position,
            mask=can_pack,
        )

        parent_static_idx = tl.load(
            static_parent_indices_ptr + static_idx,
            mask=can_pack,
            other=0,
        ).to(tl.int64)
        parent_local_idx = tl.full((), 0, tl.int64)
        found_parent = parent_static_idx == 0
        for packed_pos in range(MAX_TREE_NODES - 1):
            prior_active = can_pack & (packed_pos < packed_count)
            prior_static_idx = tl.load(
                selected_static_nodes_ptr
                + req_idx * selected_static_nodes_stride_b
                + packed_pos * selected_static_nodes_stride_w,
                mask=prior_active,
                other=-1,
            ).to(tl.int64)
            match_parent = prior_active & (prior_static_idx == parent_static_idx)
            parent_local_idx = tl.where(
                match_parent,
                packed_pos + 1,
                parent_local_idx,
            )
            found_parent = found_parent | match_parent
        tl.store(
            parent_ptr
            + req_idx * parent_stride_b
            + packed_local_idx * parent_stride_w,
            parent_local_idx,
            mask=can_pack & found_parent,
        )

        packed_count = tl.where(can_pack, packed_count + 1, packed_count)
        max_depth = tl.where(
            can_pack & (static_position > max_depth),
            static_position,
            max_depth,
        )

    valid_tree = packed_count > 0
    row_num_nodes = packed_count + 1
    tl.store(tree_valid_ptr + req_idx, valid_tree)
    tl.store(num_nodes_ptr + req_idx, row_num_nodes)
    tl.store(num_spec_steps_ptr + req_idx, max_depth + 1)
    tl.store(
        target_mask_ptr + req_idx * target_mask_stride_b,
        1,
        mask=valid_tree,
    )

    for parent_idx in range(MAX_TREE_NODES):
        first_child = tl.full((), -1, tl.int64)
        for reverse_child_idx in range(MAX_TREE_NODES - 1):
            child_idx = MAX_TREE_NODES - 1 - reverse_child_idx
            child_active = child_idx <= packed_count
            child_parent = tl.load(
                parent_ptr + req_idx * parent_stride_b + child_idx * parent_stride_w,
                mask=child_active,
                other=-2,
            ).to(tl.int64)
            is_child = child_active & (child_parent == parent_idx)
            tl.store(
                retrieve_next_sibling_ptr
                + req_idx * retrieve_next_sibling_stride_b
                + child_idx * retrieve_next_sibling_stride_w,
                first_child,
                mask=is_child,
            )
            first_child = tl.where(is_child, child_idx, first_child)
        tl.store(
            retrieve_next_token_ptr
            + req_idx * retrieve_next_token_stride_b
            + parent_idx * retrieve_next_token_stride_w,
            first_child,
            mask=parent_idx < row_num_nodes,
        )


def build_static_topk_compact_metadata_kernel(
    topk_static_nodes: torch.Tensor,
    valid_topk: torch.Tensor,
    static_tokens: torch.Tensor,
    static_parent_indices: torch.Tensor,
    static_position_offsets: torch.Tensor,
    *,
    max_tree_nodes: int | None = None,
) -> StaticTopKCompactMetadataOutput:
    """Triton candidate for fixed-width static top-k compact metadata build."""

    if topk_static_nodes.device.type != "cuda" or not HAS_TRITON:
        return build_static_topk_compact_metadata(
            topk_static_nodes,
            valid_topk,
            static_tokens,
            static_parent_indices,
            static_position_offsets,
            max_tree_nodes=max_tree_nodes,
        )

    if max_tree_nodes is None:
        max_tree_nodes = static_tokens.shape[1]
    reference_error_check = build_static_topk_compact_metadata
    if (
        topk_static_nodes.ndim != 2
        or valid_topk.shape != topk_static_nodes.shape
        or static_tokens.ndim != 2
        or static_tokens.shape[0] != topk_static_nodes.shape[0]
        or static_parent_indices.ndim != 1
        or static_position_offsets.ndim != 1
        or max_tree_nodes <= 0
        or max_tree_nodes > static_tokens.shape[1]
    ):
        return reference_error_check(
            topk_static_nodes,
            valid_topk,
            static_tokens,
            static_parent_indices,
            static_position_offsets,
            max_tree_nodes=max_tree_nodes,
        )

    topk_static_nodes = topk_static_nodes.contiguous()
    valid_topk = valid_topk.contiguous()
    static_tokens = static_tokens.contiguous()
    static_parent_indices = static_parent_indices.contiguous()
    static_position_offsets = static_position_offsets.contiguous()

    batch_size, topk_count = topk_static_nodes.shape
    static_width = static_tokens.shape[1]
    device = topk_static_nodes.device
    selected_width = max_tree_nodes - 1
    selected_static_nodes = torch.empty(
        (batch_size, selected_width), dtype=torch.int32, device=device
    )
    selected_token_ids = torch.empty(
        (batch_size, selected_width), dtype=static_tokens.dtype, device=device
    )
    retrieve_index = torch.empty(
        (batch_size, max_tree_nodes), dtype=torch.int32, device=device
    )
    retrieve_next_token = torch.empty_like(retrieve_index)
    retrieve_next_sibling = torch.empty_like(retrieve_index)
    parent = torch.empty_like(retrieve_index)
    target_mask = torch.empty_like(retrieve_index)
    position_offsets = torch.empty(
        (batch_size, max_tree_nodes), dtype=torch.int64, device=device
    )
    tree_valid = torch.empty(batch_size, dtype=torch.bool, device=device)
    num_nodes = torch.empty(batch_size, dtype=torch.int32, device=device)
    num_spec_steps = torch.empty(batch_size, dtype=torch.int32, device=device)

    _build_static_topk_compact_metadata_triton_kernel[(batch_size,)](
        topk_static_nodes,
        valid_topk,
        static_tokens,
        static_parent_indices,
        static_position_offsets,
        selected_static_nodes,
        selected_token_ids,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        parent,
        target_mask,
        position_offsets,
        tree_valid,
        num_nodes,
        num_spec_steps,
        topk_static_nodes.stride(0),
        topk_static_nodes.stride(1),
        valid_topk.stride(0),
        valid_topk.stride(1),
        static_tokens.stride(0),
        static_tokens.stride(1),
        selected_static_nodes.stride(0),
        selected_static_nodes.stride(1),
        selected_token_ids.stride(0),
        selected_token_ids.stride(1),
        retrieve_index.stride(0),
        retrieve_index.stride(1),
        retrieve_next_token.stride(0),
        retrieve_next_token.stride(1),
        retrieve_next_sibling.stride(0),
        retrieve_next_sibling.stride(1),
        parent.stride(0),
        parent.stride(1),
        target_mask.stride(0),
        target_mask.stride(1),
        position_offsets.stride(0),
        position_offsets.stride(1),
        TOPK_COUNT=topk_count,
        STATIC_WIDTH=static_width,
        MAX_TREE_NODES=max_tree_nodes,
    )

    return StaticTopKCompactMetadataOutput(
        selected_static_nodes=selected_static_nodes,
        selected_token_ids=selected_token_ids,
        retrieve_index=retrieve_index,
        retrieve_next_token=retrieve_next_token,
        retrieve_next_sibling=retrieve_next_sibling,
        parent=parent,
        target_mask=target_mask,
        position_offsets=position_offsets,
        tree_valid=tree_valid,
        num_nodes=num_nodes,
        num_spec_steps=num_spec_steps,
    )


@triton.jit
def _build_selected_bool_compact_metadata_triton_kernel(
    selected_bool_ptr,
    static_tokens_ptr,
    static_parent_indices_ptr,
    static_position_offsets_ptr,
    selected_static_nodes_ptr,
    selected_token_ids_ptr,
    retrieve_index_ptr,
    retrieve_next_token_ptr,
    retrieve_next_sibling_ptr,
    parent_ptr,
    target_mask_ptr,
    position_offsets_ptr,
    tree_valid_ptr,
    num_nodes_ptr,
    num_spec_steps_ptr,
    selected_bool_stride_b: tl.constexpr,
    selected_bool_stride_w: tl.constexpr,
    static_tokens_stride_b: tl.constexpr,
    static_tokens_stride_w: tl.constexpr,
    selected_static_nodes_stride_b: tl.constexpr,
    selected_static_nodes_stride_w: tl.constexpr,
    selected_token_ids_stride_b: tl.constexpr,
    selected_token_ids_stride_w: tl.constexpr,
    retrieve_index_stride_b: tl.constexpr,
    retrieve_index_stride_w: tl.constexpr,
    retrieve_next_token_stride_b: tl.constexpr,
    retrieve_next_token_stride_w: tl.constexpr,
    retrieve_next_sibling_stride_b: tl.constexpr,
    retrieve_next_sibling_stride_w: tl.constexpr,
    parent_stride_b: tl.constexpr,
    parent_stride_w: tl.constexpr,
    target_mask_stride_b: tl.constexpr,
    target_mask_stride_w: tl.constexpr,
    position_offsets_stride_b: tl.constexpr,
    position_offsets_stride_w: tl.constexpr,
    STATIC_WIDTH: tl.constexpr,
    MAX_TREE_NODES: tl.constexpr,
):
    req_idx = tl.program_id(0)

    for local_idx in range(MAX_TREE_NODES):
        tl.store(
            retrieve_index_ptr
            + req_idx * retrieve_index_stride_b
            + local_idx * retrieve_index_stride_w,
            local_idx,
        )
        tl.store(
            retrieve_next_token_ptr
            + req_idx * retrieve_next_token_stride_b
            + local_idx * retrieve_next_token_stride_w,
            -1,
        )
        tl.store(
            retrieve_next_sibling_ptr
            + req_idx * retrieve_next_sibling_stride_b
            + local_idx * retrieve_next_sibling_stride_w,
            -1,
        )
        tl.store(
            parent_ptr + req_idx * parent_stride_b + local_idx * parent_stride_w,
            -1,
        )
        tl.store(
            target_mask_ptr
            + req_idx * target_mask_stride_b
            + local_idx * target_mask_stride_w,
            0,
        )
        tl.store(
            position_offsets_ptr
            + req_idx * position_offsets_stride_b
            + local_idx * position_offsets_stride_w,
            0,
        )

    for selected_pos in range(MAX_TREE_NODES - 1):
        tl.store(
            selected_static_nodes_ptr
            + req_idx * selected_static_nodes_stride_b
            + selected_pos * selected_static_nodes_stride_w,
            0,
        )
        tl.store(
            selected_token_ids_ptr
            + req_idx * selected_token_ids_stride_b
            + selected_pos * selected_token_ids_stride_w,
            0,
        )

    packed_count = tl.full((), 0, tl.int64)
    max_depth = tl.full((), 0, tl.int64)

    for static_idx in range(1, STATIC_WIDTH):
        selected = tl.load(
            selected_bool_ptr
            + req_idx * selected_bool_stride_b
            + static_idx * selected_bool_stride_w
        )
        can_pack = selected & (packed_count < MAX_TREE_NODES - 1)
        packed_local_idx = packed_count + 1
        token_id = tl.load(
            static_tokens_ptr
            + req_idx * static_tokens_stride_b
            + static_idx * static_tokens_stride_w,
            mask=can_pack,
            other=0,
        )
        static_position = tl.load(
            static_position_offsets_ptr + static_idx,
            mask=can_pack,
            other=0,
        ).to(tl.int64)
        tl.store(
            selected_static_nodes_ptr
            + req_idx * selected_static_nodes_stride_b
            + packed_count * selected_static_nodes_stride_w,
            static_idx,
            mask=can_pack,
        )
        tl.store(
            selected_token_ids_ptr
            + req_idx * selected_token_ids_stride_b
            + packed_count * selected_token_ids_stride_w,
            token_id,
            mask=can_pack,
        )
        tl.store(
            target_mask_ptr
            + req_idx * target_mask_stride_b
            + packed_local_idx * target_mask_stride_w,
            1,
            mask=can_pack,
        )
        tl.store(
            position_offsets_ptr
            + req_idx * position_offsets_stride_b
            + packed_local_idx * position_offsets_stride_w,
            static_position,
            mask=can_pack,
        )

        parent_static_idx = tl.load(
            static_parent_indices_ptr + static_idx,
            mask=can_pack,
            other=0,
        ).to(tl.int64)
        parent_local_idx = tl.full((), 0, tl.int64)
        found_parent = parent_static_idx == 0
        for packed_pos in range(MAX_TREE_NODES - 1):
            prior_active = can_pack & (packed_pos < packed_count)
            prior_static_idx = tl.load(
                selected_static_nodes_ptr
                + req_idx * selected_static_nodes_stride_b
                + packed_pos * selected_static_nodes_stride_w,
                mask=prior_active,
                other=-1,
            ).to(tl.int64)
            match_parent = prior_active & (prior_static_idx == parent_static_idx)
            parent_local_idx = tl.where(
                match_parent,
                packed_pos + 1,
                parent_local_idx,
            )
            found_parent = found_parent | match_parent
        tl.store(
            parent_ptr
            + req_idx * parent_stride_b
            + packed_local_idx * parent_stride_w,
            parent_local_idx,
            mask=can_pack & found_parent,
        )

        packed_count = tl.where(can_pack, packed_count + 1, packed_count)
        max_depth = tl.where(
            can_pack & (static_position > max_depth),
            static_position,
            max_depth,
        )

    valid_tree = packed_count > 0
    row_num_nodes = packed_count + 1
    tl.store(tree_valid_ptr + req_idx, valid_tree)
    tl.store(num_nodes_ptr + req_idx, row_num_nodes)
    tl.store(num_spec_steps_ptr + req_idx, max_depth + 1)
    tl.store(
        target_mask_ptr + req_idx * target_mask_stride_b,
        1,
        mask=valid_tree,
    )

    for parent_idx in range(MAX_TREE_NODES):
        first_child = tl.full((), -1, tl.int64)
        for reverse_child_idx in range(MAX_TREE_NODES - 1):
            child_idx = MAX_TREE_NODES - 1 - reverse_child_idx
            child_active = child_idx <= packed_count
            child_parent = tl.load(
                parent_ptr + req_idx * parent_stride_b + child_idx * parent_stride_w,
                mask=child_active,
                other=-2,
            ).to(tl.int64)
            is_child = child_active & (child_parent == parent_idx)
            tl.store(
                retrieve_next_sibling_ptr
                + req_idx * retrieve_next_sibling_stride_b
                + child_idx * retrieve_next_sibling_stride_w,
                first_child,
                mask=is_child,
            )
            first_child = tl.where(is_child, child_idx, first_child)
        tl.store(
            retrieve_next_token_ptr
            + req_idx * retrieve_next_token_stride_b
            + parent_idx * retrieve_next_token_stride_w,
            first_child,
            mask=parent_idx < row_num_nodes,
        )


def build_selected_bool_compact_metadata_kernel(
    selected_bool: torch.Tensor,
    static_tokens: torch.Tensor,
    static_parent_indices: torch.Tensor,
    static_position_offsets: torch.Tensor,
    *,
    max_tree_nodes: int | None = None,
) -> StaticTopKCompactMetadataOutput:
    """Triton candidate for bool-mask compact metadata build."""

    if selected_bool.device.type != "cuda" or not HAS_TRITON:
        return build_selected_bool_compact_metadata(
            selected_bool,
            static_tokens,
            static_parent_indices,
            static_position_offsets,
            max_tree_nodes=max_tree_nodes,
        )

    if max_tree_nodes is None:
        max_tree_nodes = static_tokens.shape[1]
    reference_error_check = build_selected_bool_compact_metadata
    if (
        selected_bool.ndim != 2
        or static_tokens.ndim != 2
        or static_tokens.shape != selected_bool.shape
        or static_parent_indices.ndim != 1
        or static_position_offsets.ndim != 1
        or max_tree_nodes <= 0
        or max_tree_nodes > static_tokens.shape[1]
    ):
        return reference_error_check(
            selected_bool,
            static_tokens,
            static_parent_indices,
            static_position_offsets,
            max_tree_nodes=max_tree_nodes,
        )

    selected_bool = selected_bool.contiguous()
    static_tokens = static_tokens.contiguous()
    static_parent_indices = static_parent_indices.contiguous()
    static_position_offsets = static_position_offsets.contiguous()

    batch_size = selected_bool.shape[0]
    static_width = static_tokens.shape[1]
    device = selected_bool.device
    selected_width = max_tree_nodes - 1
    selected_static_nodes = torch.empty(
        (batch_size, selected_width), dtype=torch.int32, device=device
    )
    selected_token_ids = torch.empty(
        (batch_size, selected_width), dtype=static_tokens.dtype, device=device
    )
    retrieve_index = torch.empty(
        (batch_size, max_tree_nodes), dtype=torch.int32, device=device
    )
    retrieve_next_token = torch.empty_like(retrieve_index)
    retrieve_next_sibling = torch.empty_like(retrieve_index)
    parent = torch.empty_like(retrieve_index)
    target_mask = torch.empty_like(retrieve_index)
    position_offsets = torch.empty(
        (batch_size, max_tree_nodes), dtype=torch.int64, device=device
    )
    tree_valid = torch.empty(batch_size, dtype=torch.bool, device=device)
    num_nodes = torch.empty(batch_size, dtype=torch.int32, device=device)
    num_spec_steps = torch.empty(batch_size, dtype=torch.int32, device=device)

    _build_selected_bool_compact_metadata_triton_kernel[(batch_size,)](
        selected_bool,
        static_tokens,
        static_parent_indices,
        static_position_offsets,
        selected_static_nodes,
        selected_token_ids,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        parent,
        target_mask,
        position_offsets,
        tree_valid,
        num_nodes,
        num_spec_steps,
        selected_bool.stride(0),
        selected_bool.stride(1),
        static_tokens.stride(0),
        static_tokens.stride(1),
        selected_static_nodes.stride(0),
        selected_static_nodes.stride(1),
        selected_token_ids.stride(0),
        selected_token_ids.stride(1),
        retrieve_index.stride(0),
        retrieve_index.stride(1),
        retrieve_next_token.stride(0),
        retrieve_next_token.stride(1),
        retrieve_next_sibling.stride(0),
        retrieve_next_sibling.stride(1),
        parent.stride(0),
        parent.stride(1),
        target_mask.stride(0),
        target_mask.stride(1),
        position_offsets.stride(0),
        position_offsets.stride(1),
        STATIC_WIDTH=static_width,
        MAX_TREE_NODES=max_tree_nodes,
    )

    return StaticTopKCompactMetadataOutput(
        selected_static_nodes=selected_static_nodes,
        selected_token_ids=selected_token_ids,
        retrieve_index=retrieve_index,
        retrieve_next_token=retrieve_next_token,
        retrieve_next_sibling=retrieve_next_sibling,
        parent=parent,
        target_mask=target_mask,
        position_offsets=position_offsets,
        tree_valid=tree_valid,
        num_nodes=num_nodes,
        num_spec_steps=num_spec_steps,
    )


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
    """Triton implementation of :func:`verify_dynamic_tree_greedy`.

    The kernel keeps one program per request and performs the child/sibling
    traversal without the CPU synchronization points in the torch reference.
    CPU or non-Triton environments intentionally fall back to the reference.
    """

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
    target_mask: torch.Tensor | None = None,
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
