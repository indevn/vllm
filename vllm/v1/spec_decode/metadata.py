# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass

import numpy as np
import torch


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
