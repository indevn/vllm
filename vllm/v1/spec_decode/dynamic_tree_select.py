# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Dynamic Draft Tree selection and compact metadata public boundary.

This module is the upstream-facing import boundary for runtime selected-subtree
construction.  The implementation still lives in ``dynamic_tree`` while the DDT
prototype is being split into smaller reviewable units; callers should import
selection/build APIs from here so the physical implementation can move without
touching proposer or scheduler code again.
"""

from vllm.v1.spec_decode.dynamic_tree import (
    DynamicDraftTreeManager,
    DynamicTreeBuildOutput,
    DynamicTreeDraftOutput,
    StaticTopKCompactMetadataOutput,
    build_dynamic_tree,
    build_dynamic_tree_from_logits,
    build_selected_bool_compact_metadata,
    build_selected_bool_compact_metadata_kernel,
    build_static_topk_compact_metadata,
    build_static_topk_compact_metadata_kernel,
)

__all__ = [
    "DynamicDraftTreeManager",
    "DynamicTreeBuildOutput",
    "DynamicTreeDraftOutput",
    "StaticTopKCompactMetadataOutput",
    "build_dynamic_tree",
    "build_dynamic_tree_from_logits",
    "build_selected_bool_compact_metadata",
    "build_selected_bool_compact_metadata_kernel",
    "build_static_topk_compact_metadata",
    "build_static_topk_compact_metadata_kernel",
]
