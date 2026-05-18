# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass(frozen=True)
class DynamicTreeDeviceMetadataHandle:
    """Opaque request-aligned DDT metadata handle for local device handoff.

    The scheduler can carry this object without inspecting per-node tree
    metadata.  The model runner may consume the resident device tensors when
    the handle is still local to the same worker process; otherwise callers can
    fall back to the existing Python ``DynamicTreeCompactMetadata`` path.
    """

    handle_id: int
    req_ids: tuple[str, ...]
    selected_static_nodes: torch.Tensor
    selected_token_ids: torch.Tensor
    retrieve_index: torch.Tensor
    retrieve_next_token: torch.Tensor
    retrieve_next_sibling: torch.Tensor
    parent: torch.Tensor
    target_mask: torch.Tensor
    position_offsets: torch.Tensor
    tree_valid: torch.Tensor
    num_nodes: tuple[int, ...]
    num_spec_steps: tuple[int, ...]
    target_mask_enabled: bool
    is_dynamic_tree: bool
    is_linear_chain: bool
    select_vectorized: bool = True
    # CPU request-aligned draft-token rows for the scheduler API. This lets the
    # runner keep the full tree metadata resident while still returning the
    # existing DraftTokenIds list[list[int]] contract.
    selected_token_ids_by_req: tuple[tuple[int, ...], ...] | None = None

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def copy(self) -> "DynamicTreeDeviceMetadataHandle":
        return self


@dataclass(frozen=True)
class DynamicTreeCompactArrayView:
    """Typed host view of request-local tree metadata.

    The scheduler still carries a compact Python object per request, but this
    view normalizes repeated list/dict fields into stable typed arrays before
    the target runner stages them into reusable host/device buffers.
    """

    retrieve_index: np.ndarray
    retrieve_next_token: np.ndarray
    retrieve_next_sibling: np.ndarray
    parent: np.ndarray | None
    target_mask: np.ndarray | None
    position_offsets: np.ndarray | None
    tree_attn_mask: np.ndarray | None
    selected_token_ids: np.ndarray | None
    selected_static_nodes: np.ndarray | None

    @staticmethod
    def _int_array(values: list[int], dtype: np.dtype) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(values, dtype=dtype))

    @staticmethod
    def _optional_int_array(
        values: list[int] | None, dtype: np.dtype
    ) -> np.ndarray | None:
        if values is None:
            return None
        return DynamicTreeCompactArrayView._int_array(values, dtype)

    @staticmethod
    def _optional_mask_array(values: list[list[int]] | None) -> np.ndarray | None:
        if values is None:
            return None
        return np.ascontiguousarray(np.asarray(values, dtype=np.bool_))

    def has_required_width(self, num_nodes: int) -> bool:
        return (
            self.retrieve_index.ndim == 1
            and self.retrieve_next_token.ndim == 1
            and self.retrieve_next_sibling.ndim == 1
            and self.retrieve_index.shape[0] >= num_nodes
            and self.retrieve_next_token.shape[0] >= num_nodes
            and self.retrieve_next_sibling.shape[0] >= num_nodes
        )

    def parent_or_derive(self, num_nodes: int) -> tuple[np.ndarray, bool]:
        if (
            self.parent is not None
            and self.parent.ndim == 1
            and self.parent.shape[0] >= num_nodes
        ):
            return self.parent[:num_nodes], True

        parent_by_child = np.full(num_nodes, -1, dtype=np.int32)
        for parent_idx, first_child in enumerate(
            self.retrieve_next_token[:num_nodes]
        ):
            child_idx = int(first_child)
            visited: set[int] = set()
            while 0 <= child_idx < num_nodes and child_idx not in visited:
                visited.add(child_idx)
                parent_by_child[child_idx] = parent_idx
                child_idx = int(self.retrieve_next_sibling[child_idx])
        return parent_by_child, False

    def tree_attn_mask_or_derive(self, num_nodes: int) -> np.ndarray:
        if (
            self.tree_attn_mask is not None
            and self.tree_attn_mask.ndim == 2
            and self.tree_attn_mask.shape[0] >= num_nodes
            and self.tree_attn_mask.shape[1] >= num_nodes
        ):
            return self.tree_attn_mask[:num_nodes, :num_nodes]

        parent_by_child, _ = self.parent_or_derive(num_nodes)
        tree_attn_mask = np.zeros((num_nodes, num_nodes), dtype=np.bool_)
        tree_attn_mask[:, 0] = True
        for local_idx in range(num_nodes):
            cur_idx = local_idx
            visited: set[int] = set()
            while 0 <= cur_idx < num_nodes and cur_idx not in visited:
                visited.add(cur_idx)
                tree_attn_mask[local_idx, cur_idx] = True
                if cur_idx == 0:
                    break
                cur_idx = int(parent_by_child[cur_idx])
        return tree_attn_mask


@dataclass(frozen=True, unsafe_hash=False)
class DynamicTreeCompactMetadata:
    """Request-local compact metadata for dynamic/static tree verification."""

    __hash__ = None

    retrieve_index: list[int]
    retrieve_next_token: list[int]
    retrieve_next_sibling: list[int]
    parent: list[int] | None
    target_mask: list[int] | None
    position_offsets: list[int] | None
    tree_attn_mask: list[list[int]] | None
    selected_token_ids: list[int] | None
    selected_static_nodes: list[int] | None
    target_mask_enabled: bool
    num_spec_steps: int
    tree_valid: bool
    is_dynamic_tree: bool
    is_linear_chain: bool
    select_vectorized: bool = False
    _array_view: DynamicTreeCompactArrayView | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @classmethod
    def from_mapping(cls, metadata: dict) -> "DynamicTreeCompactMetadata":
        return cls(
            retrieve_index=list(metadata["retrieve_index"]),
            retrieve_next_token=list(metadata["retrieve_next_token"]),
            retrieve_next_sibling=list(metadata["retrieve_next_sibling"]),
            parent=(
                None
                if metadata.get("parent") is None
                else list(metadata["parent"])
            ),
            target_mask=(
                None
                if metadata.get("target_mask") is None
                else list(metadata["target_mask"])
            ),
            position_offsets=(
                None
                if metadata.get("position_offsets") is None
                else list(metadata["position_offsets"])
            ),
            tree_attn_mask=(
                None
                if metadata.get("tree_attn_mask") is None
                else [list(row) for row in metadata["tree_attn_mask"]]
            ),
            selected_token_ids=(
                None
                if metadata.get("selected_token_ids") is None
                else list(metadata["selected_token_ids"])
            ),
            selected_static_nodes=(
                None
                if metadata.get("selected_static_nodes") is None
                else list(metadata["selected_static_nodes"])
            ),
            target_mask_enabled=bool(metadata.get("target_mask_enabled", True)),
            num_spec_steps=int(metadata["num_spec_steps"]),
            tree_valid=bool(metadata.get("tree_valid", True)),
            is_dynamic_tree=bool(metadata.get("is_dynamic_tree", False)),
            is_linear_chain=bool(metadata.get("is_linear_chain", False)),
            select_vectorized=bool(
                metadata.get(
                    "select_vectorized",
                    metadata.get("is_dynamic_tree", False),
                )
            ),
        )

    def copy(self) -> "DynamicTreeCompactMetadata":
        return self

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __eq__(self, other) -> bool:
        if isinstance(other, DynamicTreeCompactMetadata):
            return self.to_dict() == other.to_dict()
        if isinstance(other, dict):
            return self.to_dict() == DynamicTreeCompactMetadata.from_mapping(
                other
            ).to_dict()
        return False

    def to_dict(self) -> dict[str, list[int] | list[list[int]] | int | bool]:
        metadata: dict[str, list[int] | list[list[int]] | int | bool] = {
            "retrieve_index": list(self.retrieve_index),
            "retrieve_next_token": list(self.retrieve_next_token),
            "retrieve_next_sibling": list(self.retrieve_next_sibling),
            "target_mask_enabled": self.target_mask_enabled,
            "num_spec_steps": self.num_spec_steps,
            "tree_valid": self.tree_valid,
            "is_dynamic_tree": self.is_dynamic_tree,
            "is_linear_chain": self.is_linear_chain,
        }
        if self.target_mask is not None:
            metadata["target_mask"] = list(self.target_mask)
        if self.parent is not None:
            metadata["parent"] = list(self.parent)
        if self.position_offsets is not None:
            metadata["position_offsets"] = list(self.position_offsets)
        if self.tree_attn_mask is not None:
            metadata["tree_attn_mask"] = [list(row) for row in self.tree_attn_mask]
        if self.selected_token_ids is not None:
            metadata["selected_token_ids"] = list(self.selected_token_ids)
        if self.selected_static_nodes is not None:
            metadata["selected_static_nodes"] = list(self.selected_static_nodes)
        if self.select_vectorized:
            metadata["select_vectorized"] = True
        return metadata

    def as_array_view(self) -> DynamicTreeCompactArrayView:
        view = self._array_view
        if view is None:
            view = DynamicTreeCompactArrayView(
                retrieve_index=DynamicTreeCompactArrayView._int_array(
                    self.retrieve_index, np.int32
                ),
                retrieve_next_token=DynamicTreeCompactArrayView._int_array(
                    self.retrieve_next_token, np.int32
                ),
                retrieve_next_sibling=DynamicTreeCompactArrayView._int_array(
                    self.retrieve_next_sibling, np.int32
                ),
                parent=DynamicTreeCompactArrayView._optional_int_array(
                    self.parent, np.int32
                ),
                target_mask=DynamicTreeCompactArrayView._optional_int_array(
                    self.target_mask, np.int32
                ),
                position_offsets=DynamicTreeCompactArrayView._optional_int_array(
                    self.position_offsets, np.int64
                ),
                tree_attn_mask=DynamicTreeCompactArrayView._optional_mask_array(
                    self.tree_attn_mask
                ),
                selected_token_ids=(
                    DynamicTreeCompactArrayView._optional_int_array(
                        self.selected_token_ids, np.int64
                    )
                ),
                selected_static_nodes=(
                    DynamicTreeCompactArrayView._optional_int_array(
                        self.selected_static_nodes, np.int32
                    )
                ),
            )
            object.__setattr__(self, "_array_view", view)
        return view


TreeSpecMetadataByReq = dict[str, DynamicTreeCompactMetadata]


@dataclass
class SpecDecodeMetadata:
    # [num_tokens]
    draft_token_ids: torch.Tensor
    # [batch_size]
    num_draft_tokens: list[int]
    # [batch_size]
    cu_num_draft_tokens: torch.Tensor
    # [batch_size]
    cu_num_sampled_tokens: torch.Tensor
    # [num_tokens]
    target_logits_indices: torch.Tensor
    # [batch_size]
    bonus_logits_indices: torch.Tensor
    # [num_tokens + batch_size]
    logits_indices: torch.Tensor
    # Optional Dynamic Draft Tree metadata.  These tensors are root-inclusive
    # and padded to a common per-batch tree width.
    # [batch_size, max_tree_nodes]
    tree_target_logits_indices: torch.Tensor | None = None
    # [batch_size, max_tree_nodes]
    tree_retrieve_index: torch.Tensor | None = None
    # [batch_size, max_tree_nodes]
    tree_retrieve_next_token: torch.Tensor | None = None
    # [batch_size, max_tree_nodes]
    tree_retrieve_next_sibling: torch.Tensor | None = None
    # [batch_size, max_tree_nodes]
    tree_parent: torch.Tensor | None = None
    # [batch_size, max_tree_nodes]
    tree_target_mask: torch.Tensor | None = None
    # Optional per-request target attention bias for packed dynamic tree rows.
    # [batch_size, max_tree_nodes, max_tree_nodes]
    tree_attn_bias: torch.Tensor | None = None
    # Root-inclusive logical position offsets for tree verify rows. Physical KV
    # slots remain in scheduled order; these offsets drive RoPE/position inputs.
    # [batch_size, max_tree_nodes]
    tree_position_offsets: torch.Tensor | None = None
    # Number of output slots to verify: accepted draft path plus final target
    # recovery/bonus token.  This is usually tree depth + 1.
    tree_num_spec_steps: int | None = None
    # [batch_size]
    tree_valid: torch.Tensor | None = None
    # Runtime bridge safety switch. When enabled, tree verification only accepts
    # draft nodes that are already a linear prefix in the scheduled KV layout.
    # Full branching acceptance requires KV relocation and leaves this disabled
    # for pure semantic/unit tests.
    tree_linear_kv_safe: bool = False
    # Optional debug trace emitted by the tree verifier when
    # VLLM_TREE_SPEC_TRACE_PATH is set.
    tree_accept_trace: list[dict] | None = None
    # Correctness fallback for backends whose multi-token target verification
    # KV is not yet greedy-equivalent to step-by-step target decode.
    force_reject_all: bool = False
    # When force_reject_all is used for tree metadata, the target runner may
    # execute only the root row per request. In that mode the verifier consumes
    # logits shaped like a non-speculative decode step while scheduler metadata
    # still carries the original draft subtree for rollback/accounting.
    force_root_only_forward: bool = False
    # TREE_ATTN linear-chain diagnostic compatibility knob. It makes the final
    # target lm_head run one row at a time so the logits GEMM shape matches
    # target-only greedy decode.
    tree_force_single_row_logits: bool = False
    # TREE_ATTN linear-chain diagnostic compatibility knob. It makes the target
    # model run the verify rows one token at a time so all target-model GEMMs
    # and attention kernels match target-only greedy decode shapes.
    tree_force_serial_q1_forward: bool = False
    # Trace-only marker set by the runner when the serial q1 target forward path
    # actually executes for this batch.
    tree_serial_q1_forward_used: bool = False
    # Current DDT runtime safety mode. Static TREE_ATTN metadata leaves this as
    # None; dynamic root-only/prefix-only/branching modes set it explicitly.
    tree_runtime_mode: str | None = None
    # Correctness candidate: when set, the runner may collapse low-margin tree
    # outputs to q1-equivalent token emission.
    tree_near_tie_q1_fallback_threshold: float | None = None
    tree_near_tie_q1_fallback_applied: list[dict] | None = None
    # Correctness candidate: after tree verification/relocation, the runner may
    # q1-recompute accepted rows and overwrite linear target state.
    tree_serial_accepted_state_repair_applied: list[dict] | None = None
    # Diagnostic CUDA graph metadata for DDT/TREE_ATTN.  These records are
    # populated by the runner after graph dispatch and are emitted in the trace
    # so graph coverage/fallback can be correlated with tree shape.
    tree_cudagraph_key: dict | None = None
    tree_cudagraph_runtime: dict | None = None
    tree_cudagraph_metadata_buffered: bool = False
    tree_cudagraph_metadata_buffer_reason: str | None = None
    tree_metadata_device_buffered: bool = False
    tree_metadata_host_staged: bool = False
    tree_metadata_device_buffer_reason: str | None = None
    tree_dynamic_select_vectorized: bool = False
    tree_metadata_typed_view: bool = False
    # Per-sample relocation caches.  These are invalidated by the runner before
    # it applies relocation because near-tie fallback may mutate accept_indices.
    tree_relocation_pairs_cache: list[dict[str, int]] | None = None
    tree_sample_relocation_pairs_cache: list[dict[str, int]] | None = None
    tree_relocation_index_cache: tuple[torch.Tensor, torch.Tensor, int] | None = None
    tree_sample_relocation_index_cache: tuple[
        torch.Tensor, torch.Tensor, int
    ] | None = None

    def __post_init__(self):
        self.max_spec_len = max(self.num_draft_tokens)

    @property
    def has_tree_metadata(self) -> bool:
        return (
            self.tree_target_logits_indices is not None
            and self.tree_retrieve_index is not None
            and self.tree_retrieve_next_token is not None
            and self.tree_retrieve_next_sibling is not None
            and self.tree_num_spec_steps is not None
        )

    @classmethod
    def make_dummy(
        cls,
        draft_token_ids: list[list[int]],
        device: torch.device,
    ) -> "SpecDecodeMetadata":
        batch_size = len(draft_token_ids)
        num_draft_tokens = [len(ids) for ids in draft_token_ids]
        num_sampled_tokens = [len(ids) + 1 for ids in draft_token_ids]
        flattened_draft_token_ids = sum(draft_token_ids, [])
        num_tokens = len(flattened_draft_token_ids)

        draft_token_ids_tensor = torch.tensor(
            flattened_draft_token_ids, dtype=torch.int32, device=device
        )
        cu_num_draft_tokens = np.cumsum(num_draft_tokens, dtype=np.int32)
        cu_num_draft_tokens_tensor = torch.from_numpy(cu_num_draft_tokens).to(device)
        cu_num_sampled_tokens = np.cumsum(num_sampled_tokens, dtype=np.int32)
        cu_num_sampled_tokens_tensor = torch.from_numpy(cu_num_sampled_tokens).to(
            device
        )

        target_logits_indices = torch.zeros(
            num_tokens, dtype=torch.int32, device=device
        )
        bonus_logits_indices = torch.zeros(batch_size, dtype=torch.int32, device=device)
        logits_indices = torch.zeros(
            num_tokens + batch_size, dtype=torch.int32, device=device
        )
        return cls(
            draft_token_ids=draft_token_ids_tensor,
            num_draft_tokens=num_draft_tokens,
            cu_num_draft_tokens=cu_num_draft_tokens_tensor,
            cu_num_sampled_tokens=cu_num_sampled_tokens_tensor,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )
