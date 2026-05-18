# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with TreeAttention."""

import ast
import os
from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_varlen_func,
)
from vllm.v1.attention.backends.utils import (
    split_decodes_and_prefills,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


class TreeAttentionBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "TREE_ATTN"

    @staticmethod
    def get_impl_cls() -> type["TreeAttentionImpl"]:
        return TreeAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_builder_cls() -> type["TreeAttentionMetadataBuilder"]:
        return TreeAttentionMetadataBuilder

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


@dataclass
class TreeAttentionMetadata:
    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_decodes: int = 0

    tree_attn_bias: torch.Tensor | None = None
    use_tree_decode_bias: bool = False
    tree_target_mask: torch.Tensor | None = None
    runtime_tree_attn_bias: torch.Tensor | None = None
    tree_retrieve_next_token: torch.Tensor | None = None
    tree_retrieve_next_sibling: torch.Tensor | None = None
    tree_parent: torch.Tensor | None = None
    use_tree_compact_bias_kernel: bool = False
    tree_root_only: bool = False
    expand_linear_chain_decode_as_q1: bool = False

    # Cached Prefill/decode metadata.
    _cached_prefill_metadata: "TreeAttentionMetadata | None" = None
    _cached_decode_metadata: "TreeAttentionMetadata | None" = None

    @property
    def prefill_metadata(self) -> "TreeAttentionMetadata | None":
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            # Recover cached prefill-phase attention
            # metadata structure
            return self._cached_prefill_metadata

        q_start_loc = self.query_start_loc[self.num_decodes :]
        q_seqlens = torch.diff(q_start_loc)
        kv_seqlens = self.seq_lens[self.num_decodes :]
        # Construct & cache prefill-phase attention metadata structure
        self._cached_prefill_metadata = TreeAttentionMetadata(
            num_actual_tokens=self.num_prefill_tokens,
            max_query_len=int(q_seqlens.max().item()),
            query_start_loc=q_start_loc - q_start_loc[0],
            max_seq_len=int(kv_seqlens.max().item()),
            seq_lens=kv_seqlens,
            block_table=self.block_table[self.num_decodes :],
            slot_mapping=self.slot_mapping[self.num_decode_tokens :],
        )
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> "TreeAttentionMetadata | None":
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            # Recover cached decode-phase attention
            # metadata structure
            return self._cached_decode_metadata

        q_start_loc = self.query_start_loc[: self.num_decodes + 1]
        q_seqlens = torch.diff(q_start_loc)
        kv_seqlens = self.seq_lens[: self.num_decodes]
        if self.expand_linear_chain_decode_as_q1 and int(q_seqlens.max().item()) > 1:
            token_offsets = torch.arange(
                self.num_decode_tokens,
                dtype=kv_seqlens.dtype,
                device=kv_seqlens.device,
            )
            row_starts = torch.repeat_interleave(
                q_start_loc[:-1],
                q_seqlens.to(torch.long),
            )
            local_offsets = token_offsets - row_starts.to(kv_seqlens.dtype)
            context_lens = kv_seqlens - q_seqlens.to(kv_seqlens.dtype)
            expanded_seq_lens = torch.repeat_interleave(
                context_lens,
                q_seqlens.to(torch.long),
            ) + local_offsets + 1
            expanded_query_start_loc = torch.arange(
                self.num_decode_tokens + 1,
                dtype=q_start_loc.dtype,
                device=q_start_loc.device,
            )
            self._cached_decode_metadata = TreeAttentionMetadata(
                num_actual_tokens=self.num_decode_tokens,
                max_query_len=1,
                query_start_loc=expanded_query_start_loc,
                max_seq_len=int(expanded_seq_lens.max().item()),
                seq_lens=expanded_seq_lens,
                block_table=torch.repeat_interleave(
                    self.block_table[: self.num_decodes],
                    q_seqlens.to(torch.long),
                    dim=0,
                ),
                slot_mapping=self.slot_mapping[: self.num_decode_tokens],
                num_decode_tokens=self.num_decode_tokens,
                num_decodes=self.num_decode_tokens,
            )
            return self._cached_decode_metadata
        # Construct & cache decode-phase attention metadata structure
        self._cached_decode_metadata = TreeAttentionMetadata(
            num_actual_tokens=self.num_decode_tokens,
            max_query_len=int(q_seqlens.max().item()),
            query_start_loc=q_start_loc,
            max_seq_len=int(kv_seqlens.max().item()),
            seq_lens=kv_seqlens,
            block_table=self.block_table[: self.num_decodes],
            slot_mapping=self.slot_mapping[: self.num_decode_tokens],
            tree_attn_bias=self._decode_tree_attn_bias(q_seqlens),
            use_tree_decode_bias=self.use_tree_decode_bias,
            tree_target_mask=self.tree_target_mask[: self.num_decodes]
            if self.tree_target_mask is not None
            else None,
            runtime_tree_attn_bias=self.runtime_tree_attn_bias[: self.num_decodes]
            if self.runtime_tree_attn_bias is not None
            else None,
            tree_retrieve_next_token=self.tree_retrieve_next_token[
                : self.num_decodes
            ]
            if self.tree_retrieve_next_token is not None
            else None,
            tree_retrieve_next_sibling=self.tree_retrieve_next_sibling[
                : self.num_decodes
            ]
            if self.tree_retrieve_next_sibling is not None
            else None,
            tree_parent=self.tree_parent[: self.num_decodes]
            if self.tree_parent is not None
            else None,
            use_tree_compact_bias_kernel=(
                not self.tree_root_only
                and os.environ.get("VLLM_TREE_ATTN_DYNAMIC_MASK_KERNEL") == "1"
                and self.tree_parent is not None
                and self._can_use_tree_compact_bias_kernel(
                    q_seqlens,
                    self.tree_parent.shape[-1],
                )
            ),
            tree_root_only=self.tree_root_only,
        )
        return self._cached_decode_metadata

    def _decode_tree_attn_bias(self, q_seqlens: torch.Tensor) -> torch.Tensor | None:
        if self.tree_root_only:
            return None
        if not self.use_tree_decode_bias or (
            self.tree_attn_bias is None
            and self.tree_retrieve_next_token is None
            and self.tree_parent is None
        ):
            return None
        if (
            os.environ.get("VLLM_TREE_ATTN_DYNAMIC_MASK_KERNEL") == "1"
            and self.tree_parent is not None
        ):
            tree_width = self.tree_parent.shape[-1]
            if self._can_use_tree_compact_bias_kernel(q_seqlens, tree_width):
                return None
        if self.tree_attn_bias is None:
            return None
        if self.runtime_tree_attn_bias is not None:
            tree_width = self.runtime_tree_attn_bias.shape[-1]
            if self.runtime_tree_attn_bias.shape[-2:] != (tree_width, tree_width):
                raise ValueError(
                    "runtime tree attention bias must have shape "
                    f"(*, {tree_width}, {tree_width}), got "
                    f"{self.runtime_tree_attn_bias.shape}"
                )
            if int(q_seqlens.max().item()) != tree_width:
                return None
            if self.tree_target_mask is None:
                return self.runtime_tree_attn_bias
            return _apply_tree_target_mask(
                self.runtime_tree_attn_bias,
                self.tree_target_mask,
            )
        tree_width = self.tree_attn_bias.shape[0]
        if int(q_seqlens.max().item()) != tree_width:
            return None
        if self.tree_target_mask is None:
            return self.tree_attn_bias
        return _apply_tree_target_mask(self.tree_attn_bias, self.tree_target_mask)

    @staticmethod
    def _can_use_tree_compact_bias_kernel(
        q_seqlens: torch.Tensor,
        tree_width: int,
    ) -> bool:
        # The compact kernel consumes per-row qlen from query_start_loc and
        # uses tree_width only as the maximum local index space.  This covers
        # mixed batches such as q1/no-draft rows beside full-tree rows as long
        # as every row fits inside the padded runtime tree metadata.
        return bool(torch.all((q_seqlens > 0) & (q_seqlens <= tree_width)).item())


class TreeAttentionMetadataBuilder(AttentionMetadataBuilder[TreeAttentionMetadata]):
    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        if os.environ.get("VLLM_TREE_ATTN_CUDAGRAPH_PROBE") == "1":
            return AttentionCGSupport.UNIFORM_BATCH
        return AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.block_size = kv_cache_spec.block_size

        spec_config = vllm_config.speculative_config
        spec_token_tree: str | None = None
        if spec := spec_config:
            spec_token_tree = spec.speculative_token_tree
        tree_choices: list[tuple[int, ...]] = (
            ast.literal_eval(spec_token_tree) if spec_token_tree is not None else [(0,)]
        )
        self.is_linear_chain = _is_linear_chain(tree_choices)
        # Construct the tree attention bias.
        depth_counts = _get_depth_counts(tree_choices)
        self.tree_attn_bias = _prepare_tree_attn_bias(
            tree_choices,
            depth_counts,
            dtype=torch.float32,
            device=device,
        )

        self.enable_linear_chain_verify = (
            os.environ.get("VLLM_TREE_ATTN_ENABLE_LINEAR_CHAIN_VERIFY") == "1"
        )
        self.decode_threshold = (
            1
            if self.is_linear_chain and not self.enable_linear_chain_verify
            else self.tree_attn_bias.shape[0]
        )
        self.reorder_batch_threshold = self.decode_threshold
        self.use_tree_decode_bias = not self.is_linear_chain

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TreeAttentionMetadata:
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata,
                decode_threshold=self.decode_threshold,
                # Branching TREE_ATTN raises the decode threshold to the tree
                # width so full-tree verify rows stay on the decode path.  A
                # mixed batch can also contain short prefill rows with no draft
                # tokens; applying the tree qq-bias to those ordinary prompt
                # rows corrupts their causal prefill attention.
                treat_short_extends_as_decodes=(
                    not self.use_tree_decode_bias
                    or common_attn_metadata.is_prefilling is None
                ),
            )
        )

        num_actual_tokens = common_attn_metadata.num_actual_tokens
        q_start_loc = common_attn_metadata.query_start_loc
        max_query_len = common_attn_metadata.max_query_len
        kv_seqlens = common_attn_metadata.seq_lens
        max_seq_len = common_attn_metadata.max_seq_len
        block_table = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        tree_target_mask = common_attn_metadata.tree_target_mask
        runtime_tree_attn_bias = common_attn_metadata.tree_attn_bias
        tree_retrieve_next_token = common_attn_metadata.tree_retrieve_next_token
        tree_retrieve_next_sibling = common_attn_metadata.tree_retrieve_next_sibling
        tree_parent = common_attn_metadata.tree_parent

        return TreeAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            max_query_len=max_query_len,
            query_start_loc=q_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=kv_seqlens,
            block_table=block_table,
            slot_mapping=slot_mapping,
            tree_attn_bias=self.tree_attn_bias,
            use_tree_decode_bias=self.use_tree_decode_bias,
            tree_target_mask=tree_target_mask,
            runtime_tree_attn_bias=runtime_tree_attn_bias,
            tree_retrieve_next_token=tree_retrieve_next_token,
            tree_retrieve_next_sibling=tree_retrieve_next_sibling,
            tree_parent=tree_parent,
            tree_root_only=common_attn_metadata.tree_root_only,
            expand_linear_chain_decode_as_q1=(
                self.is_linear_chain and self.enable_linear_chain_verify
            ),
        )

    def build_for_drafting(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int,
    ) -> TreeAttentionMetadata:
        # Cache the original tree attention bias.
        orig_tree_attn_bias = self.tree_attn_bias

        if draft_index == 0:
            # Use prefill for drafting at the root level.
            self.tree_attn_bias = torch.empty(0)
        else:
            # Slice the tree attention bias for drafting. Exclude
            # the root level.
            start, end = 1, 1 + common_attn_metadata.max_query_len
            self.tree_attn_bias = self.tree_attn_bias[start:end, start:end].contiguous()

        # Build attention bias.
        attn_metadata = self.build(0, common_attn_metadata, fast_build=True)

        # Reset the tree attention bias to the original value.
        self.tree_attn_bias = orig_tree_attn_bias
        return attn_metadata


def _get_depth_counts(sorted_tree_choices: list[tuple[int, ...]]) -> list[int]:
    # Count the number of choices at each depth of the tree.
    depth_counts = []
    prev_depth = 0
    for path in sorted_tree_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth
    return depth_counts


def _is_linear_chain(sorted_tree_choices: list[tuple[int, ...]]) -> bool:
    if not sorted_tree_choices:
        return True
    return sorted_tree_choices == [
        (0,) * depth for depth in range(1, len(sorted_tree_choices) + 1)
    ]


def _prepare_tree_attn_bias(
    sorted_tree_choices: list[tuple[int, ...]],
    depth_counts: list[int],
    dtype: torch.dtype | None,
    device: torch.device | None,
) -> torch.Tensor:
    # +1 comes from the additional root node.
    tree_len = len(sorted_tree_choices) + 1
    tree_attn_mask = torch.full(
        (tree_len, tree_len), -torch.inf, device=device, dtype=dtype
    )

    # Set diagonal to all zeros. Each token should
    # attend to itself.
    mask_val = 0
    for i in range(tree_len):
        tree_attn_mask[i, i] = mask_val

    # Set root to all zeros. All tokens attend to it.
    tree_attn_mask[:, 0] = mask_val

    # Set all ancestors to zeros.
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_tree_choice = sorted_tree_choices[start + j]
            # Retrieve ancestor position.
            if len(cur_tree_choice) == 1:
                continue
            ancestor_idx = []
            for c in range(len(cur_tree_choice) - 1):
                ancestor_idx.append(
                    sorted_tree_choices.index(cur_tree_choice[: c + 1]) + 1
                )
            tree_attn_mask[j + start + 1, ancestor_idx] = mask_val
        start += depth_counts[i]
    return tree_attn_mask


def _apply_tree_target_mask(
    tree_attn_bias: torch.Tensor,
    tree_target_mask: torch.Tensor,
) -> torch.Tensor:
    if tree_target_mask.ndim != 2:
        raise ValueError(
            f"tree_target_mask must be 2D, got {tree_target_mask.shape}"
        )
    if tree_target_mask.shape[-1] != tree_attn_bias.shape[-1]:
        raise ValueError(
            "tree_target_mask width must match tree attention bias width, got "
            f"{tree_target_mask.shape[-1]} and {tree_attn_bias.shape[-1]}"
        )
    if tree_attn_bias.ndim == 2:
        masked_bias = tree_attn_bias.unsqueeze(0).expand(
            tree_target_mask.shape[0], -1, -1
        ).clone()
    elif tree_attn_bias.ndim == 3:
        if tree_attn_bias.shape[0] != tree_target_mask.shape[0]:
            raise ValueError(
                "per-request tree attention bias batch size must match "
                "tree_target_mask, got "
                f"{tree_attn_bias.shape[0]} and {tree_target_mask.shape[0]}"
            )
        masked_bias = tree_attn_bias.clone()
    else:
        raise ValueError(f"tree_attn_bias must be 2D or 3D, got {tree_attn_bias.shape}")
    disabled = tree_target_mask.to(torch.bool).logical_not()
    # Root stays visible even if a malformed runtime mask clears it.
    disabled[:, 0] = False
    masked_bias.masked_fill_(disabled[:, None, :], -torch.inf)
    return masked_bias.contiguous()


def materialize_tree_attn_bias_from_retrieve_metadata(
    tree_retrieve_next_token: torch.Tensor,
    tree_retrieve_next_sibling: torch.Tensor,
    tree_target_mask: torch.Tensor | None,
    dtype: torch.dtype | None,
    device: torch.device | None,
) -> torch.Tensor:
    """Materialize TREE_ATTN qq-bias from compact child/sibling metadata.

    This is the correctness oracle for the compact metadata contract. The
    current decode switch still returns a dense qq-bias for ``unified_attention``;
    a future kernel can consume the same compact tensors directly.
    """

    if tree_retrieve_next_token.ndim != 2:
        raise ValueError(
            "tree_retrieve_next_token must be 2D, got "
            f"{tree_retrieve_next_token.shape}"
        )
    if tree_retrieve_next_sibling.shape != tree_retrieve_next_token.shape:
        raise ValueError(
            "tree_retrieve_next_sibling shape must match "
            "tree_retrieve_next_token, got "
            f"{tree_retrieve_next_sibling.shape} and "
            f"{tree_retrieve_next_token.shape}"
        )

    batch_size, tree_width = tree_retrieve_next_token.shape
    device = device if device is not None else tree_retrieve_next_token.device
    if dtype is None:
        dtype = torch.float32
    next_token = tree_retrieve_next_token.to(device=device, dtype=torch.long)
    next_sibling = tree_retrieve_next_sibling.to(device=device, dtype=torch.long)
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)

    parent_by_child = torch.full(
        (batch_size, tree_width),
        -1,
        dtype=torch.long,
        device=device,
    )
    for parent_idx in range(tree_width):
        child_idx = next_token[:, parent_idx]
        visited = torch.zeros(
            (batch_size, tree_width),
            dtype=torch.bool,
            device=device,
        )
        for _ in range(tree_width):
            safe_child_idx = child_idx.clamp(0, tree_width - 1)
            valid = (
                (child_idx >= 0)
                & (child_idx < tree_width)
                & ~visited[batch_indices, safe_child_idx]
            )

            current_parent = parent_by_child.gather(
                1, safe_child_idx[:, None]
            ).squeeze(1)
            updated_parent = torch.where(
                valid,
                torch.full_like(current_parent, parent_idx),
                current_parent,
            )
            parent_by_child.scatter_(
                1,
                safe_child_idx[:, None],
                updated_parent[:, None],
            )

            current_visited = visited.gather(1, safe_child_idx[:, None]).squeeze(1)
            updated_visited = current_visited | valid
            visited.scatter_(1, safe_child_idx[:, None], updated_visited[:, None])

            next_child = next_sibling.gather(1, safe_child_idx[:, None]).squeeze(1)
            child_idx = torch.where(valid, next_child, torch.full_like(child_idx, -1))

    reachable = torch.zeros(
        (batch_size, tree_width),
        dtype=torch.bool,
        device=device,
    )
    reachable[:, 0] = True
    parent_valid = parent_by_child >= 0
    safe_parent = parent_by_child.clamp(0, tree_width - 1)
    for _ in range(tree_width):
        reachable |= parent_valid & reachable.gather(1, safe_parent)

    visible = torch.zeros(
        (batch_size, tree_width, tree_width),
        dtype=torch.bool,
        device=device,
    )
    for local_idx in range(tree_width):
        cur_idx = torch.full(
            (batch_size,),
            local_idx,
            dtype=torch.long,
            device=device,
        )
        active = reachable[:, local_idx]
        visited = torch.zeros(
            (batch_size, tree_width),
            dtype=torch.bool,
            device=device,
        )
        row = visible[:, local_idx, :]
        for _ in range(tree_width):
            safe_cur_idx = cur_idx.clamp(0, tree_width - 1)
            valid = (
                active
                & (cur_idx >= 0)
                & (cur_idx < tree_width)
                & ~visited[batch_indices, safe_cur_idx]
            )

            current_visible = row.gather(1, safe_cur_idx[:, None]).squeeze(1)
            row.scatter_(1, safe_cur_idx[:, None], (current_visible | valid)[:, None])

            current_visited = visited.gather(1, safe_cur_idx[:, None]).squeeze(1)
            visited.scatter_(
                1,
                safe_cur_idx[:, None],
                (current_visited | valid)[:, None],
            )

            next_cur = parent_by_child.gather(1, safe_cur_idx[:, None]).squeeze(1)
            cur_idx = torch.where(valid, next_cur, torch.full_like(cur_idx, -1))
            active = valid & (next_cur >= 0)

    tree_attn_bias = torch.full(
        (batch_size, tree_width, tree_width),
        -torch.inf,
        dtype=dtype,
        device=device,
    )
    tree_attn_bias.masked_fill_(visible, 0)
    if tree_target_mask is not None:
        tree_target_mask = tree_target_mask.to(device=device)
        tree_attn_bias = _apply_tree_target_mask(tree_attn_bias, tree_target_mask)
    return tree_attn_bias.contiguous()


def build_static_tree_retrieve_metadata(
    sorted_tree_choices: list[tuple[int, ...]],
) -> dict[str, list[int] | int | bool]:
    """Build verifier retrieve metadata for a static speculative token tree.

    The metadata is root-inclusive. Draft nodes are addressed by their
    breadth-first position in ``sorted_tree_choices`` plus one, which matches
    the flattened draft token order returned by ``EagleProposer.propose_tree``.
    """

    num_nodes = len(sorted_tree_choices) + 1
    retrieve_index = list(range(num_nodes))
    retrieve_next_token = [-1] * num_nodes
    retrieve_next_sibling = [-1] * num_nodes

    choice_to_local_idx = {
        choice: local_idx + 1 for local_idx, choice in enumerate(sorted_tree_choices)
    }
    tree_attn_mask = [[0] * num_nodes for _ in range(num_nodes)]
    for local_idx in range(num_nodes):
        tree_attn_mask[local_idx][0] = 1
        tree_attn_mask[local_idx][local_idx] = 1
    for local_idx in range(num_nodes - 1, 0, -1):
        choice = sorted_tree_choices[local_idx - 1]
        parent_idx = 0 if len(choice) == 1 else choice_to_local_idx[choice[:-1]]
        old_first_child = retrieve_next_token[parent_idx]
        retrieve_next_token[parent_idx] = local_idx
        if old_first_child != -1:
            retrieve_next_sibling[local_idx] = old_first_child
        while parent_idx > 0:
            tree_attn_mask[local_idx][parent_idx] = 1
            parent_choice = sorted_tree_choices[parent_idx - 1]
            parent_idx = (
                0
                if len(parent_choice) == 1
                else choice_to_local_idx[parent_choice[:-1]]
            )

    max_depth = max((len(choice) for choice in sorted_tree_choices), default=0)
    return {
        "retrieve_index": retrieve_index,
        "retrieve_next_token": retrieve_next_token,
        "retrieve_next_sibling": retrieve_next_sibling,
        "target_mask": [1] * num_nodes,
        "tree_attn_mask": tree_attn_mask,
        "position_offsets": [0]
        + [len(choice) for choice in sorted_tree_choices],
        "num_spec_steps": max_depth + 1,
        "tree_valid": True,
        "is_linear_chain": _is_linear_chain(sorted_tree_choices),
    }


class TreeAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if logits_soft_cap is None:
            # Setting logits_soft_cap to 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "TreeAttentionImpl."
            )

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        key_cache, value_cache = kv_cache.unbind(0)

        # Reshape the input keys and values and store them in the cache.
        # NOTE(woosuk): Here, key and value are padded while slot_mapping is
        # not padded. However, we don't need to do key[:num_actual_tokens]
        # and value[:num_actual_tokens] because the reshape_and_cache_flash
        # op uses the slot_mapping's shape to determine the number of
        # actual tokens.
        ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TreeAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with TreeAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for TreeAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        key_cache, value_cache = kv_cache.unbind(0)

        num_decode_tokens = attn_metadata.num_decode_tokens

        def run_flash_attn(
            metadata: TreeAttentionMetadata,
            start_idx: int,
        ) -> None:
            end_idx = start_idx + metadata.num_actual_tokens
            descale_shape = (metadata.query_start_loc.shape[0] - 1, key.shape[1])
            flash_attn_varlen_func(
                q=query[start_idx:end_idx],
                k=key_cache,
                v=value_cache,
                out=output[start_idx:end_idx],
                cu_seqlens_q=metadata.query_start_loc,
                max_seqlen_q=metadata.max_query_len,
                seqused_k=metadata.seq_lens,
                max_seqlen_k=metadata.max_seq_len,
                softmax_scale=self.scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                window_size=self.sliding_window,
                block_table=metadata.block_table,
                softcap=self.logits_soft_cap,
                q_descale=None,  # Not supported
                k_descale=layer._k_scale.expand(descale_shape),
                v_descale=layer._v_scale.expand(descale_shape),
            )

        if prefill_meta := attn_metadata.prefill_metadata:
            run_flash_attn(prefill_meta, num_decode_tokens)

        if decode_meta := attn_metadata.decode_metadata:
            if (
                decode_meta.tree_attn_bias is None
                and not decode_meta.use_tree_compact_bias_kernel
            ):
                run_flash_attn(decode_meta, 0)
                return output

            descale_shape = (decode_meta.query_start_loc.shape[0] - 1, key.shape[1])
            unified_attention(
                q=query[:num_decode_tokens],
                k=key_cache,
                v=value_cache,
                out=output[:num_decode_tokens],
                cu_seqlens_q=decode_meta.query_start_loc,
                max_seqlen_q=decode_meta.max_query_len,
                seqused_k=decode_meta.seq_lens,
                max_seqlen_k=decode_meta.max_seq_len,
                softmax_scale=self.scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                qq_bias=decode_meta.tree_attn_bias,
                tree_retrieve_next_token=decode_meta.tree_retrieve_next_token
                if decode_meta.use_tree_compact_bias_kernel
                else None,
                tree_retrieve_next_sibling=decode_meta.tree_retrieve_next_sibling
                if decode_meta.use_tree_compact_bias_kernel
                else None,
                tree_parent=decode_meta.tree_parent
                if decode_meta.use_tree_compact_bias_kernel
                else None,
                tree_target_mask=decode_meta.tree_target_mask
                if decode_meta.use_tree_compact_bias_kernel
                else None,
                window_size=self.sliding_window,
                block_table=decode_meta.block_table,
                softcap=self.logits_soft_cap,
                q_descale=None,  # Not supported
                k_descale=layer._k_scale.expand(descale_shape),
                v_descale=layer._v_scale.expand(descale_shape),
            )
        return output
