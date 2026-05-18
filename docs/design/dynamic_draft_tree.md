# Dynamic Draft Tree

This fork explores Dynamic Draft Tree (DDT) speculative decoding in vLLM. It is
intended as a reviewer-friendly branch for design discussion and correctness
hardening, not as a proposed one-shot upstream patch.

## Motivation

vLLM already has upstream support for static tree speculative decoding through
TREE_ATTN. Static Draft Tree (SDT) assumes a known tree topology that can be
prepared ahead of time. DDT generalizes that path by allowing the draft topology
to be selected at runtime, so each request and decode step can carry compact tree
metadata derived from the current drafter state.

The expected benefit is not only a larger or different tree. The main systems
question is whether vLLM can represent runtime-selected draft topology without
turning the decode step into Python-heavy metadata construction. The long-term
target is a correctness-safe DDT path with low dynamic overhead and benchmarked
acceptance-producing speedups over SDT.

## Scope

The current branch focuses on these pieces:

- Compact dynamic tree metadata for selected nodes, parents, siblings, masks,
  retrieve indices, position offsets, and accepted-token state.
- Runtime propagation of dynamic tree metadata from proposer/drafter paths into
  scheduler output, target attention, verifier, and state update.
- Dynamic target mask support, including a dense-mask oracle path and a compact
  metadata switch for TREE_ATTN decode.
- KV/state relocation for accepted non-prefix nodes, with correctness checks
  against target-only and static-prefix baselines.
- Near-tie repair policy for low-margin greedy-token flips observed in batched
  or graph-shaped verification paths.
- Regression harnesses for root-only, prefix-only, branching, expanded topology,
  CUDA graph stability, and concurrency subsets.

## Correctness Contract

The DDT runtime is validated in layers:

1. Chain verify should match target-only greedy decoding when no branching
   metadata is involved.
2. Prefix-only dynamic metadata should reduce to a correctness-safe linear
   speculative decode path.
3. Branching paths should preserve target logits after KV/state relocation for
   accepted non-prefix nodes.
4. DDT failures are only counted as hard correctness failures when the oracle is
   stable and the decision is not in a configured low-margin near-tie region.

This distinction matters because BF16 target logits can produce near-tie greedy
argmax flips when the same logical token is evaluated through different packed
batch shapes, query grouping, or physical KV layouts. The branch therefore keeps
both strict checks and explained near-tie classifications instead of hiding all
differences behind token-only pass/fail labels.

## Current Status

The branch contains an end-to-end DDT prototype that has been exercised across
root-only, prefix-only, static branching, dynamic branching, expanded topology,
and selected concurrency cases. Correctness harnesses and benchmark scripts have
been added under `benchmarks/spec_decode/` and focused unit/regression coverage
has been added under `tests/`.

The implementation is intentionally still being hardened. In particular,
`gpu_model_runner.py`, `llm_base_proposer.py`, and `dynamic_tree.py` currently
contain prototype-scale changes that should be split before upstream review.
The likely upstreamable path is a sequence of small PRs, starting with compact
metadata oracle tests and typed handoff structures before enabling DDT runtime
behavior.

## Reviewer Notes

This branch is useful for reviewing the systems shape of DDT:

- Which metadata must be dynamic instead of static.
- Where scheduler, attention, verifier, and state update boundaries need typed
  handoff APIs.
- Which correctness gates are required before performance claims are meaningful.
- Which prototype changes should become isolated modules rather than runner
  special cases.

It is not yet the desired final upstream diff. The intended next hardening steps
are to reduce runner intrusion, make near-tie behavior a documented production
policy, extend prefix-cache and batch-shape coverage, and separate correctness
infrastructure from performance optimization work.

## Suggested Upstreaming Shape

1. RFC: describe DDT metadata, correctness contract, near-tie policy, and
   proposed integration boundaries.
2. PR 1: add compact metadata data structures plus dense-mask oracle tests with
   no serving behavior change.
3. PR 2: add scheduler/request typed handoff for dynamic tree metadata behind a
   disabled feature flag.
4. PR 3: add target mask and verifier integration for root-only and prefix-only
   modes.
5. PR 4: add branching relocation correctness support with focused regression
   gates.
6. PR 5+: add kernelized mask/verifier paths, CUDA graph compatibility gates,
   monitoring, and benchmarked performance enablement.
