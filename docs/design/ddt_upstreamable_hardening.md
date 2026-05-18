# DDT Upstreamable Hardening

This document tracks the hardening work required before Dynamic Draft Tree
(DDT) can be proposed upstream as a sequence of small vLLM PRs. It is scoped to
correctness, observability, reviewer ergonomics, and regression evidence. It
does not claim that the current prototype is production-ready or that every DDT
cell is faster than SDT.

## Step 0: RFC Draft

The RFC position is:

- vLLM already supports static draft tree speculative decoding through
  TREE_ATTN/SDT.
- DDT changes the topology contract from static configuration to runtime
  compact metadata selected by the drafter for the current request/step.
- The runtime must carry that metadata through scheduler output, target
  attention, verifier, accepted-state update, and KV/state relocation.
- Correctness is claimed only after failures are classified against a stable
  oracle. `oracle_unstable` and `explained_low_margin` are not used to hide
  metadata, mask, verifier, or relocation bugs.
- Performance is claimable only for clean no-trace runs where DDT produces
  acceptance and beats an acceptance-capable SDT baseline for the same workload.

The initial upstream PR should be deterministic and behavior-preserving:
compact metadata to dense TREE_ATTN bias oracle. Runtime DDT enablement belongs
to later PRs.

## Step 1: Compact Metadata to Dense TREE_ATTN Bias Oracle

Acceptance boundary:

- Input fields: `retrieve_next_token`, `retrieve_next_sibling`, optional
  `target_mask`.
- Output: dense query-query TREE_ATTN bias for the selected runtime tree.
- Semantics: every query can see itself, root, and ancestors; unrelated
  siblings/descendants are hidden; padding rows/cols are hidden; target-mask can
  hide candidate nodes while preserving the existing root visibility rule.
- No scheduler, verifier, relocation, model runner, or serving behavior change
  is required for the first upstream PR.

Current repo evidence:

- `tests/v1/attention/test_attention_splitting.py::test_compact_tree_metadata_materializes_dense_bias_oracle`
- `tests/v1/attention/test_attention_splitting.py::test_compact_tree_metadata_keeps_padding_rows_invisible`
- `tests/v1/attention/test_attention_splitting.py::test_tree_attn_compact_decode_switch_uses_kernel_metadata`
- `tests/v1/attention/test_attention_splitting.py::test_tree_attn_compact_decode_switch_supports_mixed_q_lens`
- CUDA-only comparisons exist for compact kernel vs dense oracle and should be
  treated as PR 2 evidence unless reviewers want oracle and kernel in one PR.

## Step 2: Near-Tie Policy Hardening

Production policy:

- Near-tie serial q1 repair is a correctness fallback, not the default DDT fast
  path.
- Clean performance claims must run with near-tie repair disabled.
- Optional fallback can only inspect the output row that is about to commit.
- It must never trigger on prefill/no-draft/padded/invalid rows.
- Every application must record threshold, fallback reason, local/logits row,
  margin, and truncated accepted-token count.

Current repo evidence:

- `vllm/v1/spec_decode/dynamic_tree_near_tie.py` owns near-tie margin
  classification, candidate filtering, root q1 replacement records, and
  non-root truncation records. `gpu_model_runner.py` keeps only the serial q1
  forward callback and thin compatibility wrappers.
- `tests/v1/spec_decode/test_dynamic_tree_near_tie.py` covers helper-level
  candidate filtering, non-root truncation, root q1 replacement, state updates,
  and record schema.
- Worker tests cover disabled threshold, no-draft rows, invalid tree rows,
  root q1 replacement, non-root truncation, and fallback records.
- Replay classifier tests separate `hard_fail`, `explained_low_margin`, and
  `oracle_unstable`, with oracle instability taking precedence over low-margin
  classification.

Open hardening item:

- The fallback is still opt-in and still considered a correctness guard rather
  than the default fast path. The next policy hardening step is an explicit
  deterministic q1 guard that either rejects/truncates before the near-tie row
  or runs a documented serial q1 recompute for commit rows only.

## Step 3: Runner Intrusion Reduction

Current issue:

- The prototype has large DDT additions in `gpu_model_runner.py`,
  `llm_base_proposer.py`, and `dynamic_tree.py`.
- That shape is acceptable for proving feasibility but too intrusive for
  upstream review.

First landed split:

- `vllm/v1/spec_decode/dynamic_tree_relocation.py` now owns relocation
  pair/index/cache helper logic.
- `gpu_model_runner.py` keeps wrapper call sites so runtime behavior remains
  unchanged.
- `tests/v1/worker/test_gpu_model_runner.py::test_dynamic_tree_relocation_helper_matches_runner_contract`
  validates pair construction, sample pair conversion, index tensors, and cache
  clearing.
- `vllm/v1/spec_decode/dynamic_tree_near_tie.py` now owns near-tie candidate
  filtering and fallback record construction. `gpu_model_runner.py` was reduced
  by the extracted algorithm block while preserving wrapper method names.
- `vllm/v1/spec_decode/dynamic_tree_metrics.py` now owns CUDA graph runtime
  annotation and stage-profile record construction. Runner code passes runtime
  counters and small context values instead of assembling graph/fallback/
  metadata/acceptance records inline.

Next splits:

- `dynamic_tree_select.py`: import boundary is landed for proposer/tests;
  physical movement of compact metadata builders and runtime selection out of
  `dynamic_tree.py` remains.
- `dynamic_tree_verify.py`: reference verifier and Triton wrapper now live in
  the verify module; `dynamic_tree.py` keeps compatibility wrappers for legacy
  imports.
- typed metadata handoff: scheduler/request/runner should pass a compact
  metadata object or device handle instead of rebuilding list/dict/tensor glue.

Review target:

- Keep future `gpu_model_runner.py` deltas mostly as orchestration glue.
- Move algorithmic DDT blocks longer than roughly 30-50 lines behind helper
  functions with direct unit tests.

## Step 4: Production Monitoring Contract

DDT must be observable before production enablement. Required metric families:

- enable/disable: requests, steps, runtime mode, disable reason.
- acceptance: selected width, static width, accepted tokens, acceptance length.
- graph: graph key bucket, dispatch hit, eager fallback, replay delta.
- metadata: typed view, device-buffered path, host-staged fallback,
  vectorized-select use.
- mask: compact kernel requested/used, dense fallback reason.
- relocation: relocation rows, pairs, non-prefix pairs, cache hit/miss.
- near-tie: candidates, repaired rows, truncation count, threshold.
- prefix cache: enabled, hit, miss, DDT disabled due to prefix-cache risk.

Alert conditions:

- graph eager fallback grows unexpectedly on a graph-claimed cell.
- metadata host staging appears on a production graph/device path.
- compact mask is requested but dense fallback dominates.
- acceptance length collapses near 1 while DDT is enabled.
- relocation or near-tie fallback spikes after rollout.
- prefix-cache hit rate or p95/p99 ITL regresses against the SDT control bucket.

The current branch records many of these fields in trace and benchmark
summaries. Upstream production readiness still requires stable metrics instead
of only JSONL trace records.

## Step 5: Prefix Cache / Batch Shape / Context Length Matrix

Correctness oracle:

- pass iff `hard_fail_count=0` after low-margin and oracle-unstable
  classification.
- structural invariants must hold for compact metadata, dense target mask,
  compact-mask kernel when enabled, relocation pairs, slot mapping, and block
  table.
- first divergence capture should include input ids, positions, slot mapping,
  query start locations, logits indices, top-k logits/margins, and KV block
  table.

Performance is comparable only when:

- model, weights, prompts, output lengths, trace policy, graph/eager policy,
  static SDT acceptance capability, DDT acceptance, and prefix-cache hit
  distribution are matched.

Initial matrix:

| Axis | Claimable cells | Diagnostic-first cells |
| --- | --- | --- |
| Batch shape | batch 1/2/4 short uniform, batch 2 mixed short/medium | batch 4 mixed short/medium/long |
| Context length | short, medium, long with matched graph policy | near KV block boundary, multiple KV blocks/request |
| Prefix cache | off, on-no-hit, high-hit after partial-hit passes | partial hit, shared prefix across concurrent requests |
| Graph mode | piecewise graph with dispatch hit and eager fallback 0 | eager diagnostic, forced fallback, bucket pressure |

Existing harness entry:

```bash
./.conda/bin/python benchmarks/spec_decode/run_ddt_correctness_regression.py \
  --suite full \
  --target-trace /path/to/target_trace.jsonl \
  --ddt-trace /path/to/ddt_trace.jsonl \
  --output-dir /tmp/ddt_correctness_full \
  --low-margin-threshold 0.5
```

Cells graduate from diagnostic to claimable only after correctness passes,
metadata invariants pass, graph fallback is explained, and prefix-cache hit/miss
distribution is recorded when prefix cache is enabled.
