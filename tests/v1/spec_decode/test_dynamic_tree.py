# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

torch = pytest.importorskip("torch")

from vllm.v1.spec_decode.dynamic_tree_select import (  # noqa: E402
    DynamicDraftTreeManager,
    build_dynamic_tree,
    build_dynamic_tree_from_logits,
    build_selected_bool_compact_metadata,
    build_selected_bool_compact_metadata_kernel,
    build_static_topk_compact_metadata,
    build_static_topk_compact_metadata_kernel,
)
from vllm.v1.spec_decode.dynamic_tree_verify import (  # noqa: E402
    verify_dynamic_tree_greedy,
    verify_dynamic_tree_greedy_from_draft,
    verify_dynamic_tree_greedy_kernel,
)


def _cuda_is_usable():
    try:
        torch.empty(1, device="cuda")
    except Exception:
        return False
    return True


def _reference_tree(parent_list, selected_index, top_k, depth):
    num_draft_tokens = len(selected_index) + 1
    selected_to_position = {
        history_idx: pos + 1 for pos, history_idx in enumerate(selected_index)
    }
    mask = [[0] * num_draft_tokens for _ in range(num_draft_tokens)]
    positions = [0] * num_draft_tokens
    retrieve_index = list(range(num_draft_tokens))
    next_token = [-1] * num_draft_tokens
    next_sibling = [-1] * num_draft_tokens

    for row in mask:
        row[0] = 1

    for local_idx in range(num_draft_tokens - 1, 0, -1):
        parent_table_idx = selected_index[local_idx - 1] // top_k
        if parent_table_idx == 0:
            parent_position = 0
        elif parent_table_idx >= len(parent_list):
            continue
        else:
            parent_position = selected_to_position.get(parent_list[parent_table_idx])
            if parent_position is None:
                continue

        old_first_child = next_token[parent_position]
        next_token[parent_position] = local_idx
        if old_first_child != -1:
            next_sibling[local_idx] = old_first_child

    for local_idx in range(1, num_draft_tokens):
        selected_pos = local_idx - 1
        position = 0
        while position < depth + 1:
            position += 1
            mask[local_idx][selected_pos + 1] = 1

            parent_table_idx = selected_index[selected_pos] // top_k
            if parent_table_idx == 0:
                break
            parent_position = selected_to_position.get(parent_list[parent_table_idx])
            if parent_position is None:
                break
            selected_pos = parent_position - 1
        positions[local_idx] = position

    return mask, positions, retrieve_index, next_token, next_sibling


def _reference_verify(
    candidates,
    retrieve_index,
    retrieve_next_token,
    retrieve_next_sibling,
    target_predict,
    num_spec_steps,
    tree_valid=True,
):
    num_draft_tokens = len(candidates)
    predicts = [0] * num_draft_tokens
    accept_index = [0] * num_spec_steps
    accept_token = [0] * num_spec_steps

    if not tree_valid:
        accept_token[0] = target_predict[0]
        predicts[0] = target_predict[0]
        return predicts, accept_index, 0, accept_token

    last_accepted_local_idx = retrieve_index[0]
    accept_index[0] = last_accepted_local_idx
    accept_token[0] = target_predict[last_accepted_local_idx]
    cur_index = 0
    num_accepted_tokens = 0

    for _ in range(1, num_spec_steps):
        cur_index = retrieve_next_token[cur_index]

        while cur_index != -1:
            draft_local_idx = retrieve_index[cur_index]
            if candidates[cur_index] == target_predict[last_accepted_local_idx]:
                predicts[last_accepted_local_idx] = target_predict[
                    last_accepted_local_idx
                ]
                num_accepted_tokens += 1
                accept_index[num_accepted_tokens] = draft_local_idx
                accept_token[num_accepted_tokens] = target_predict[draft_local_idx]
                last_accepted_local_idx = draft_local_idx
                break
            cur_index = retrieve_next_sibling[cur_index]

        if cur_index == -1:
            break

    predicts[last_accepted_local_idx] = target_predict[last_accepted_local_idx]
    return predicts, accept_index, num_accepted_tokens, accept_token


def test_build_dynamic_tree_reference_branching():
    # K=2 history layout:
    #   selected 0, 1 -> root children
    #   selected 2    -> child of selected 0
    #   selected 4    -> child of selected 2
    # Final local tree:
    #   0(root) -> 1 -> 3 -> 4
    #           -> 2
    parent_list = torch.tensor([[0, 0, 2, 0, 0]], dtype=torch.int64)
    selected_index = torch.tensor([[0, 1, 2, 4]], dtype=torch.int64)

    output = build_dynamic_tree(
        parent_list,
        selected_index,
        top_k=2,
        depth=3,
    )

    expected_mask = torch.tensor(
        [
            [
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 0, 1, 0, 0],
                [1, 1, 0, 1, 0],
                [1, 1, 0, 1, 1],
            ]
        ],
        dtype=torch.int32,
        device="cpu",
    )
    expected_positions = torch.tensor(
        [[0, 1, 1, 2, 3]], dtype=torch.int32, device="cpu"
    )
    expected_retrieve = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int32, device="cpu")
    expected_next_token = torch.tensor(
        [[1, 3, -1, 4, -1]], dtype=torch.int32, device="cpu"
    )
    expected_next_sibling = torch.tensor(
        [[-1, 2, -1, -1, -1]], dtype=torch.int32, device="cpu"
    )

    assert torch.equal(output.tree_mask.cpu(), expected_mask)
    assert torch.equal(output.positions.cpu(), expected_positions)
    assert torch.equal(output.retrieve_index.cpu(), expected_retrieve)
    assert torch.equal(output.retrieve_next_token.cpu(), expected_next_token)
    assert torch.equal(output.retrieve_next_sibling.cpu(), expected_next_sibling)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not _cuda_is_usable(), reason="CUDA is not usable"
            ),
        ),
    ],
)
def test_build_static_topk_compact_metadata_closes_and_packs_static_tree(device):
    # Static tree:
    #   0(root) -> 1 -> 3 -> 5
    #           -> 2 -> 6
    #                -> 4
    static_parent = torch.tensor([0, 0, 0, 1, 2, 3, 2], device=device)
    static_position_offsets = torch.tensor([0, 1, 1, 2, 2, 3, 2], device=device)
    static_tokens = torch.tensor(
        [
            [0, 11, 12, 13, 14, 15, 16],
            [0, 21, 22, 23, 24, 25, 26],
        ],
        dtype=torch.int64,
        device=device,
    )
    topk_static_nodes = torch.tensor(
        [
            [5, 4],
            [6, 0],
        ],
        dtype=torch.int64,
        device=device,
    )
    valid_topk = torch.tensor(
        [
            [True, True],
            [True, False],
        ],
        device=device,
    )

    output = build_static_topk_compact_metadata(
        topk_static_nodes,
        valid_topk,
        static_tokens,
        static_parent,
        static_position_offsets,
        max_tree_nodes=7,
    )

    assert output.tree_valid.cpu().tolist() == [True, True]
    assert output.num_nodes.cpu().tolist() == [6, 3]
    assert output.num_spec_steps.cpu().tolist() == [4, 3]
    assert output.selected_static_nodes.cpu().tolist() == [
        [1, 2, 3, 4, 5, 0],
        [2, 6, 0, 0, 0, 0],
    ]
    assert output.selected_token_ids.cpu().tolist() == [
        [11, 12, 13, 14, 15, 0],
        [22, 26, 0, 0, 0, 0],
    ]
    assert output.retrieve_index.cpu().tolist() == [
        [0, 1, 2, 3, 4, 5, 6],
        [0, 1, 2, 3, 4, 5, 6],
    ]
    assert output.retrieve_next_token.cpu().tolist() == [
        [1, 3, 4, 5, -1, -1, -1],
        [1, 2, -1, -1, -1, -1, -1],
    ]
    assert output.retrieve_next_sibling.cpu().tolist() == [
        [-1, 2, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1],
    ]
    assert output.parent.cpu().tolist() == [
        [-1, 0, 0, 1, 2, 3, -1],
        [-1, 0, 1, -1, -1, -1, -1],
    ]
    assert output.target_mask.cpu().tolist() == [
        [1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 0, 0, 0, 0],
    ]
    assert output.position_offsets.cpu().tolist() == [
        [0, 1, 1, 2, 2, 3, 0],
        [0, 1, 2, 0, 0, 0, 0],
    ]


def test_build_static_topk_compact_metadata_marks_empty_rows_invalid():
    output = build_static_topk_compact_metadata(
        torch.tensor([[1, 2]], dtype=torch.int64),
        torch.tensor([[False, False]]),
        torch.tensor([[0, 11, 12]], dtype=torch.int64),
        torch.tensor([0, 0, 0], dtype=torch.int64),
        torch.tensor([0, 1, 1], dtype=torch.int64),
        max_tree_nodes=3,
    )

    assert output.tree_valid.tolist() == [False]
    assert output.num_nodes.tolist() == [1]
    assert output.num_spec_steps.tolist() == [1]
    assert output.target_mask.tolist() == [[0, 0, 0]]
    assert output.retrieve_next_token.tolist() == [[-1, -1, -1]]


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not _cuda_is_usable(), reason="CUDA is not usable"
            ),
        ),
    ],
)
def test_build_selected_bool_compact_metadata_matches_topk_oracle(device):
    static_parent = torch.tensor([0, 0, 0, 1, 2, 3, 2], device=device)
    static_position_offsets = torch.tensor([0, 1, 1, 2, 2, 3, 2], device=device)
    static_tokens = torch.tensor(
        [
            [0, 11, 12, 13, 14, 15, 16],
            [0, 21, 22, 23, 24, 25, 26],
            [0, 31, 32, 33, 34, 35, 36],
        ],
        dtype=torch.int64,
        device=device,
    )
    topk_static_nodes = torch.tensor(
        [
            [5, 4],
            [6, 0],
            [1, 2],
        ],
        dtype=torch.int64,
        device=device,
    )
    valid_topk = torch.tensor(
        [
            [True, True],
            [True, False],
            [False, False],
        ],
        device=device,
    )
    selected_bool = torch.zeros((3, 7), dtype=torch.bool, device=device)
    selected_bool[0, [1, 2, 3, 4, 5]] = True
    selected_bool[1, [2, 6]] = True

    reference = build_static_topk_compact_metadata(
        topk_static_nodes,
        valid_topk,
        static_tokens,
        static_parent,
        static_position_offsets,
        max_tree_nodes=7,
    )
    output = build_selected_bool_compact_metadata(
        selected_bool,
        static_tokens,
        static_parent,
        static_position_offsets,
        max_tree_nodes=7,
    )

    assert torch.equal(output.selected_static_nodes, reference.selected_static_nodes)
    assert torch.equal(output.selected_token_ids, reference.selected_token_ids)
    assert torch.equal(output.retrieve_index, reference.retrieve_index)
    assert torch.equal(output.retrieve_next_token, reference.retrieve_next_token)
    assert torch.equal(output.retrieve_next_sibling, reference.retrieve_next_sibling)
    assert torch.equal(output.parent, reference.parent)
    assert torch.equal(output.target_mask, reference.target_mask)
    assert torch.equal(output.position_offsets, reference.position_offsets)
    assert torch.equal(output.tree_valid, reference.tree_valid)
    assert torch.equal(output.num_nodes, reference.num_nodes)
    assert torch.equal(output.num_spec_steps, reference.num_spec_steps)


@pytest.mark.skipif(not _cuda_is_usable(), reason="CUDA is not usable")
def test_build_selected_bool_compact_metadata_kernel_matches_oracle():
    static_parent = torch.tensor([0, 0, 0, 1, 2, 3, 2], device="cuda")
    static_position_offsets = torch.tensor([0, 1, 1, 2, 2, 3, 2], device="cuda")
    static_tokens = torch.tensor(
        [
            [0, 11, 12, 13, 14, 15, 16],
            [0, 21, 22, 23, 24, 25, 26],
            [0, 31, 32, 33, 34, 35, 36],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    selected_bool = torch.zeros((3, 7), dtype=torch.bool, device="cuda")
    selected_bool[0, [1, 2, 3, 4, 5]] = True
    selected_bool[1, [2, 6]] = True

    reference = build_selected_bool_compact_metadata(
        selected_bool,
        static_tokens,
        static_parent,
        static_position_offsets,
        max_tree_nodes=7,
    )
    kernel = build_selected_bool_compact_metadata_kernel(
        selected_bool,
        static_tokens,
        static_parent,
        static_position_offsets,
        max_tree_nodes=7,
    )

    assert torch.equal(kernel.selected_static_nodes, reference.selected_static_nodes)
    assert torch.equal(kernel.selected_token_ids, reference.selected_token_ids)
    assert torch.equal(kernel.retrieve_index, reference.retrieve_index)
    assert torch.equal(kernel.retrieve_next_token, reference.retrieve_next_token)
    assert torch.equal(kernel.retrieve_next_sibling, reference.retrieve_next_sibling)
    assert torch.equal(kernel.parent, reference.parent)
    assert torch.equal(kernel.target_mask, reference.target_mask)
    assert torch.equal(kernel.position_offsets, reference.position_offsets)
    assert torch.equal(kernel.tree_valid, reference.tree_valid)
    assert torch.equal(kernel.num_nodes, reference.num_nodes)
    assert torch.equal(kernel.num_spec_steps, reference.num_spec_steps)


@pytest.mark.skipif(not _cuda_is_usable(), reason="CUDA is not usable")
def test_build_static_topk_compact_metadata_kernel_matches_oracle():
    static_parent = torch.tensor([0, 0, 0, 1, 2, 3, 2], device="cuda")
    static_position_offsets = torch.tensor([0, 1, 1, 2, 2, 3, 2], device="cuda")
    static_tokens = torch.tensor(
        [
            [0, 11, 12, 13, 14, 15, 16],
            [0, 21, 22, 23, 24, 25, 26],
            [0, 31, 32, 33, 34, 35, 36],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    topk_static_nodes = torch.tensor(
        [
            [5, 4],
            [6, 0],
            [1, 2],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    valid_topk = torch.tensor(
        [
            [True, True],
            [True, False],
            [False, False],
        ],
        device="cuda",
    )

    reference = build_static_topk_compact_metadata(
        topk_static_nodes,
        valid_topk,
        static_tokens,
        static_parent,
        static_position_offsets,
        max_tree_nodes=7,
    )
    kernel = build_static_topk_compact_metadata_kernel(
        topk_static_nodes,
        valid_topk,
        static_tokens,
        static_parent,
        static_position_offsets,
        max_tree_nodes=7,
    )

    assert torch.equal(kernel.selected_static_nodes, reference.selected_static_nodes)
    assert torch.equal(kernel.selected_token_ids, reference.selected_token_ids)
    assert torch.equal(kernel.retrieve_index, reference.retrieve_index)
    assert torch.equal(kernel.retrieve_next_token, reference.retrieve_next_token)
    assert torch.equal(kernel.retrieve_next_sibling, reference.retrieve_next_sibling)
    assert torch.equal(kernel.parent, reference.parent)
    assert torch.equal(kernel.target_mask, reference.target_mask)
    assert torch.equal(kernel.position_offsets, reference.position_offsets)
    assert torch.equal(kernel.tree_valid, reference.tree_valid)
    assert torch.equal(kernel.num_nodes, reference.num_nodes)
    assert torch.equal(kernel.num_spec_steps, reference.num_spec_steps)


def test_build_dynamic_tree_ignores_unselected_parent_in_child_links():
    parent_list = torch.tensor([[0, 0, 99]], dtype=torch.int64)
    selected_index = torch.tensor([[0, 4]], dtype=torch.int64)

    output = build_dynamic_tree(
        parent_list,
        selected_index,
        top_k=2,
        depth=2,
    )

    # selected 4 points at parent table row 2, whose history token 99 was not
    # selected into the final tree.  The node remains addressable but is not
    # linked as a reachable child, matching TRT's "ignored token" behavior.
    assert output.retrieve_next_token.tolist() == [[1, -1, -1]]
    assert output.retrieve_next_sibling.tolist() == [[-1, -1, -1]]
    assert output.positions.tolist() == [[0, 1, 1]]
    assert output.tree_mask.tolist() == [
        [
            [1, 0, 0],
            [1, 1, 0],
            [1, 0, 1],
        ]
    ]


def test_verify_dynamic_tree_greedy_accepts_first_matching_path():
    build = build_dynamic_tree(
        torch.tensor([[0, 0, 2, 0, 0]], dtype=torch.int64),
        torch.tensor([[0, 1, 2, 4]], dtype=torch.int64),
        top_k=2,
        depth=3,
    )
    candidates = torch.tensor([[101, 11, 12, 13, 14]], dtype=torch.int64)
    target_predict = torch.tensor([[11, 13, 99, 14, 42]], dtype=torch.int64)

    output = verify_dynamic_tree_greedy(
        candidates,
        build.retrieve_index,
        build.retrieve_next_token,
        build.retrieve_next_sibling,
        target_predict,
        num_spec_steps=4,
    )

    assert output.accept_token_num.tolist() == [3]
    assert output.accept_index.tolist() == [[0, 1, 3, 4]]
    assert output.accept_token.tolist() == [[11, 13, 14, 42]]
    assert output.predicts.tolist() == [[11, 13, 0, 14, 42]]


def test_verify_dynamic_tree_greedy_linear_kv_safe_accepts_only_prefix_path():
    build = build_dynamic_tree(
        torch.tensor([[0, 0, 2, 0, 0]], dtype=torch.int64),
        torch.tensor([[0, 1, 2, 4]], dtype=torch.int64),
        top_k=2,
        depth=3,
    )
    candidates = torch.tensor([[101, 11, 12, 13, 14]], dtype=torch.int64)
    target_predict = torch.tensor([[11, 13, 99, 14, 42]], dtype=torch.int64)

    output = verify_dynamic_tree_greedy(
        candidates,
        build.retrieve_index,
        build.retrieve_next_token,
        build.retrieve_next_sibling,
        target_predict,
        num_spec_steps=4,
        linear_kv_safe=True,
    )

    assert output.accept_token_num.tolist() == [1]
    assert output.accept_index.tolist() == [[0, 1, 0, 0]]
    assert output.accept_token.tolist() == [[11, 13, 0, 0]]
    assert output.predicts.tolist() == [[11, 13, 0, 0, 0]]


def test_verify_dynamic_tree_greedy_scans_siblings_and_stops_on_miss():
    build = build_dynamic_tree(
        torch.tensor([[0, 0, 2, 0, 0]], dtype=torch.int64),
        torch.tensor([[0, 1, 2, 4]], dtype=torch.int64),
        top_k=2,
        depth=3,
    )
    candidates = torch.tensor([[101, 11, 12, 13, 14]], dtype=torch.int64)
    target_predict = torch.tensor([[12, 77, 88, 99, 100]], dtype=torch.int64)

    output = verify_dynamic_tree_greedy(
        candidates,
        build.retrieve_index,
        build.retrieve_next_token,
        build.retrieve_next_sibling,
        target_predict,
        num_spec_steps=4,
    )

    assert output.accept_token_num.tolist() == [1]
    assert output.accept_index.tolist() == [[0, 2, 0, 0]]
    assert output.accept_token.tolist() == [[12, 88, 0, 0]]
    assert output.predicts.tolist() == [[12, 0, 88, 0, 0]]


def test_verify_dynamic_tree_greedy_invalid_tree_accepts_only_bonus():
    build = build_dynamic_tree(
        torch.tensor([[0, 0, 2]], dtype=torch.int64),
        torch.tensor([[0, 1]], dtype=torch.int64),
        top_k=2,
        depth=2,
    )
    candidates = torch.tensor([[101, 11, 12]], dtype=torch.int64)
    target_predict = torch.tensor([[11, 13, 14]], dtype=torch.int64)

    output = verify_dynamic_tree_greedy(
        candidates,
        build.retrieve_index,
        build.retrieve_next_token,
        build.retrieve_next_sibling,
        target_predict,
        num_spec_steps=3,
        tree_valid=torch.tensor([False]),
    )

    assert output.accept_token_num.tolist() == [0]
    assert output.accept_index.tolist() == [[0, 0, 0]]
    assert output.accept_token.tolist() == [[11, 0, 0]]
    assert output.predicts.tolist() == [[11, 0, 0]]


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not _cuda_is_usable(), reason="CUDA is not usable"
            ),
        ),
    ],
)
def test_build_dynamic_tree_from_logits_selects_and_verifies_tree(device):
    top_k = 2
    depth = 3
    max_total_draft_tokens = 4

    root_logits = torch.tensor([[0.0, 4.0, 3.0, -2.0, -2.0, -2.0]], device=device)
    depth1_logits = torch.tensor(
        [
            [
                [-2.0, -2.0, -2.0, 4.0, 3.0, -2.0],
                [-2.0, -2.0, -2.0, -2.0, 1.0, 4.0],
            ]
        ],
        device=device,
    )
    depth2_logits = torch.tensor(
        [
            [
                [-2.0, -2.0, -2.0, -2.0, 4.0, 3.0],
                [-2.0, -2.0, -2.0, 4.0, -2.0, 3.0],
            ]
        ],
        device=device,
    )

    draft = build_dynamic_tree_from_logits(
        root_logits,
        [depth1_logits, depth2_logits],
        top_k=top_k,
        depth=depth,
        max_total_draft_tokens=max_total_draft_tokens,
    )

    assert draft.draft_token_ids.cpu().tolist() == [[1, 2, 3, 4]]
    assert draft.selected_index.cpu().tolist() == [[0, 1, 2, 6]]
    assert draft.parent_list.cpu().tolist() == [[-1, 0, 1, 2, 4]]
    assert draft.history_draft_token_ids.cpu().tolist() == [
        [1, 2, 3, 4, 5, 4, 4, 5, 3, 5]
    ]

    build = draft.build_output
    assert build.retrieve_next_token.cpu().tolist() == [[1, 3, -1, 4, -1]]
    assert build.retrieve_next_sibling.cpu().tolist() == [[-1, 2, -1, -1, -1]]
    assert build.positions.cpu().tolist() == [[0, 1, 1, 2, 3]]

    candidates = draft.candidates(root_token_ids=torch.tensor([101], device=device))
    assert candidates.cpu().tolist() == [[101, 1, 2, 3, 4]]

    target_predict = torch.tensor([[1, 3, 99, 4, 42]], device=device)
    verify = verify_dynamic_tree_greedy_from_draft(draft, target_predict)

    assert verify.accept_token_num.cpu().tolist() == [3]
    assert verify.accept_index.cpu().tolist() == [[0, 1, 3, 4]]
    assert verify.accept_token.cpu().tolist() == [[1, 3, 4, 42]]
    assert verify.predicts.cpu().tolist() == [[1, 3, 0, 4, 42]]


def test_dynamic_draft_tree_manager_matches_functional_api():
    manager = DynamicDraftTreeManager(
        top_k=2,
        depth=2,
        max_total_draft_tokens=3,
    )
    root_logits = torch.tensor([[0.0, 4.0, 3.0, -2.0, -2.0]])
    child_logits = torch.tensor(
        [
            [
                [-2.0, -2.0, -2.0, 4.0, 3.0],
                [-2.0, -2.0, -2.0, -2.0, 4.0],
            ]
        ]
    )

    direct = build_dynamic_tree_from_logits(
        root_logits,
        [child_logits],
        top_k=2,
        depth=2,
        max_total_draft_tokens=3,
    )
    managed = manager.build_from_logits(root_logits, [child_logits])

    assert torch.equal(managed.draft_token_ids, direct.draft_token_ids)
    assert torch.equal(managed.selected_index, direct.selected_index)
    assert torch.equal(managed.parent_list, direct.parent_list)

    target_predict = torch.tensor([[1, 3, 99, 42]])
    assert torch.equal(
        manager.verify_greedy(managed, target_predict).accept_token,
        verify_dynamic_tree_greedy_from_draft(direct, target_predict).accept_token,
    )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not _cuda_is_usable(), reason="CUDA is not usable"
            ),
        ),
    ],
)
def test_dynamic_tree_matches_independent_oracle_for_random_valid_trees(device):
    generator = torch.Generator(device=device).manual_seed(0)
    top_k = 3
    depth = 4
    batch_size = 4
    num_selected = 9
    parent_width = top_k * (depth - 1) + 1

    parent_rows = []
    selected_rows = []
    for batch_idx in range(batch_size):
        selected = list(range(num_selected))
        parent = [0] * parent_width
        for table_idx in range(1, parent_width):
            parent[table_idx] = selected[(table_idx + batch_idx - 1) % table_idx]
        parent_rows.append(parent)
        selected_rows.append(selected)

    parent_list = torch.tensor(parent_rows, dtype=torch.int64, device=device)
    selected_index = torch.tensor(selected_rows, dtype=torch.int64, device=device)

    build = build_dynamic_tree(
        parent_list,
        selected_index,
        top_k=top_k,
        depth=depth,
    )

    expected_masks = []
    expected_positions = []
    expected_retrieve = []
    expected_next_token = []
    expected_next_sibling = []
    for parent, selected in zip(parent_rows, selected_rows):
        mask, positions, retrieve, next_token, next_sibling = _reference_tree(
            parent, selected, top_k, depth
        )
        expected_masks.append(mask)
        expected_positions.append(positions)
        expected_retrieve.append(retrieve)
        expected_next_token.append(next_token)
        expected_next_sibling.append(next_sibling)

    assert build.tree_mask.cpu().tolist() == expected_masks
    assert build.positions.cpu().tolist() == expected_positions
    assert build.retrieve_index.cpu().tolist() == expected_retrieve
    assert build.retrieve_next_token.cpu().tolist() == expected_next_token
    assert build.retrieve_next_sibling.cpu().tolist() == expected_next_sibling

    candidates = torch.randint(
        10,
        1000,
        (batch_size, num_selected + 1),
        generator=generator,
        device=device,
    )
    target_predict = torch.randint(
        10,
        1000,
        (batch_size, num_selected + 1),
        generator=generator,
        device=device,
    )
    tree_valid = torch.tensor([True, True, False, True], device=device)

    # Force several deterministic accepted paths, including sibling fallback.
    target_predict[0, 0] = candidates[0, 1]
    target_predict[0, 1] = candidates[0, 4]
    target_predict[1, 0] = candidates[1, 3]
    target_predict[1, 3] = candidates[1, 8]
    target_predict[3, 0] = candidates[3, 2]

    verify = verify_dynamic_tree_greedy(
        candidates,
        build.retrieve_index,
        build.retrieve_next_token,
        build.retrieve_next_sibling,
        target_predict,
        num_spec_steps=depth + 1,
        tree_valid=tree_valid,
    )

    expected_predicts = []
    expected_accept_index = []
    expected_accept_token_num = []
    expected_accept_token = []
    for batch_idx in range(batch_size):
        predicts, accept_index, accept_token_num, accept_token = _reference_verify(
            candidates[batch_idx].cpu().tolist(),
            expected_retrieve[batch_idx],
            expected_next_token[batch_idx],
            expected_next_sibling[batch_idx],
            target_predict[batch_idx].cpu().tolist(),
            depth + 1,
            bool(tree_valid[batch_idx].item()),
        )
        expected_predicts.append(predicts)
        expected_accept_index.append(accept_index)
        expected_accept_token_num.append(accept_token_num)
        expected_accept_token.append(accept_token)

    assert verify.predicts.cpu().tolist() == expected_predicts
    assert verify.accept_index.cpu().tolist() == expected_accept_index
    assert verify.accept_token_num.cpu().tolist() == expected_accept_token_num
    assert verify.accept_token.cpu().tolist() == expected_accept_token


@pytest.mark.skipif(not _cuda_is_usable(), reason="CUDA is not usable")
@pytest.mark.parametrize("linear_kv_safe", [False, True])
@pytest.mark.parametrize("with_target_mask", [False, True])
def test_verify_dynamic_tree_greedy_kernel_matches_reference(
    linear_kv_safe,
    with_target_mask,
):
    top_k = 3
    depth = 4
    batch_size = 4
    num_selected = 9
    parent_width = top_k * (depth - 1) + 1
    parent_rows = []
    selected_rows = []
    for batch_idx in range(batch_size):
        selected = list(range(num_selected))
        parent = [0] * parent_width
        for table_idx in range(1, parent_width):
            parent[table_idx] = selected[(table_idx + batch_idx - 1) % table_idx]
        parent_rows.append(parent)
        selected_rows.append(selected)

    parent_list = torch.tensor(parent_rows, dtype=torch.int64, device="cuda")
    selected_index = torch.tensor(selected_rows, dtype=torch.int64, device="cuda")
    build = build_dynamic_tree(
        parent_list,
        selected_index,
        top_k=top_k,
        depth=depth,
    )

    candidates = torch.tensor(
        [
            [101, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            [201, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            [301, 31, 32, 33, 34, 35, 36, 37, 38, 39],
            [401, 41, 42, 43, 44, 45, 46, 47, 48, 49],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    target_predict = torch.tensor(
        [
            [11, 14, 99, 99, 17, 99, 99, 42, 99, 99],
            [23, 99, 99, 28, 99, 99, 99, 99, 55, 99],
            [31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
            [42, 99, 100, 101, 102, 103, 104, 105, 106, 107],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    tree_valid = torch.tensor([True, True, False, True], device="cuda")
    target_mask = None
    if with_target_mask:
        target_mask = torch.ones_like(candidates, dtype=torch.int32)
        target_mask[0, 4] = 0
        target_mask[1, 8] = 0

    reference = verify_dynamic_tree_greedy(
        candidates,
        build.retrieve_index,
        build.retrieve_next_token,
        build.retrieve_next_sibling,
        target_predict,
        num_spec_steps=depth + 1,
        tree_valid=tree_valid,
        linear_kv_safe=linear_kv_safe,
        target_mask=target_mask,
    )
    kernel = verify_dynamic_tree_greedy_kernel(
        candidates,
        build.retrieve_index,
        build.retrieve_next_token,
        build.retrieve_next_sibling,
        target_predict,
        num_spec_steps=depth + 1,
        tree_valid=tree_valid,
        linear_kv_safe=linear_kv_safe,
        target_mask=target_mask,
    )

    assert torch.equal(kernel.predicts, reference.predicts)
    assert torch.equal(kernel.accept_index, reference.accept_index)
    assert torch.equal(kernel.accept_token_num, reference.accept_token_num)
    assert torch.equal(kernel.accept_token, reference.accept_token)
