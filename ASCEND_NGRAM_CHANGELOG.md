# Ascend Ngram Change Log

Base: `93288799` (`[Core] Port Ascend ngram opt to v0.11.0-dev`)

## Commit-by-commit changes

1. `50f269d7` - Fix ngram proposer init before input batch is ready.
   - Avoided early init path accessing `input_batch` before runner setup finished.

2. `f783e16e` - Fix async ngram proposer token flow and remove duplicate CPU writes.
   - Corrected async token handoff path and removed duplicate `token_ids_cpu` updates.

3. `9ea3267a` - Add ngram runtime markers, accept-rate logs, and sync-path optimizations.
   - Added startup/runtime/execute markers and acceptance diagnostics.
   - Added initial sync-path reduction changes for sampled token handling.

4. `35e5e569` - Raise ngram markers to warning and add propose call trace.
   - Increased marker visibility in runtime logs.
   - Added propose-call trace marker.

5. `89e53b40` - Add ngram execute and draft handoff warning markers.
   - Added explicit markers for execute step and draft handoff for runtime validation.

6. `1e9288b0` - Align Ascend ngram request eligibility with upstream behavior.
   - Synced request eligibility checks with upstream vLLM semantics.

7. `623032ea` - Add ngram match stats warning logs for zero-draft diagnosis.
   - Added per-step match stats to diagnose `matched_reqs=0` / `total_draft_tokens=0`.

8. `e30d165b` - Fix missing ngram match-log state initialization.
   - Fixed missing internal counter/state that caused proposer runtime errors.

9. `01e241fd` - Fix ngram eligibility fallback and add empty-validation diagnostics.
   - Added fallback path for token-length edge conditions.
   - Added diagnostics when validation becomes empty.

10. `1052d407` - Fix ngram eligibility checks to use length semantics.
    - Corrected request validity checks to use token length semantics consistently.

11. `d96900bc` - Compat: support old rejection parse_output signature.
    - Added compatibility for older `RejectionSampler.parse_output` function signatures.

12. `165d9c67` - Reduce ngram debug overhead and optimize sampled-token tolist path.
    - Reduced high-frequency logging overhead.
    - Added faster sampled-token tolist path (guarded path).

13. `8ca29c36` - Fix sampled token normalization and guard fast tolist path.
    - Normalized sampled token structures to avoid detokenizer/type regressions.
    - Strengthened guardrails for fast tolist path.

14. `092baadd` - Remove accept-rate logs and reduce ngram decode overhead.
    - Removed hot-path accept-rate logging overhead.
    - Reduced decode-path runtime overhead for ngram mode.

15. `1aec077a` - Avoid per-step parse_output exceptions in ngram path.
    - Removed per-step signature exception overhead.
    - Added one-time capability detection for parse_output kwargs support.

16. `5c75cefc` - Vectorize Ascend rejection sampler random path.
    - Updated rejection-sampler hot path to reduce Python-loop overhead.

17. `e1169518` - Optimize ngram rejection path and numba matcher threading.
    - Experimental optimization: ngram-specific rejection path and proposer thread strategy.
    - Result: introduced throughput regression in user validation environment.

18. `384ce776` - Rollback ngram regression and skip rejection on zero-draft steps.
    - Rolled back regression-prone rejection path changes.
    - Added fast path: if `target_logits_indices` is empty (no draft tokens), skip rejection sampler.

19. `b18c8551` - Rollback ngram proposer threading changes.
    - Rolled back aggressive proposer thread behavior from `e1169518`.

20. `e4cb56b3` - Vectorize expand_batch_to_tokens in rejection sampler.
    - Replaced per-request Python expansion with `repeat_interleave` vectorization.
    - This aligns better with upstream kernel-style expansion behavior.

## Notes

- `e1169518` was partially rolled back by `384ce776` and `b18c8551` due measured regression.
- Current optimization direction focuses on:
  - reducing host-side overhead when ngram drafts are sparse,
  - minimizing Python hot-path work,
  - preserving compatibility with upstream vLLM semantics.
