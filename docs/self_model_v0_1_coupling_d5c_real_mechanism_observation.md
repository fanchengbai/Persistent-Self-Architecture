# Self Model v0.1 Coupling-D5C real mechanism observation

Date: 2026-08-21

## Execution identity and integrity

The project owner authorized exactly one real 2.9B non-Core mechanism smoke and observation. It ran on clean `main` commit `a8ef52a5390581666edf4e6ffecaf61aee912a9e`. The single-use claim status is `d5c_single_use_execution_claim_consumed`; its canonical content recomputes to SHA-256 `75d69ae3ad4550361cc53d03ae5d89fd636f045d31a6cd62974c4dc15496f12f`, matching the report. The report canonical content independently recomputes to `187cdfd4f43f4fbc990d08b120c25c36629010133693697b0bb42e48ea8cdb21`.

The run loaded and executed the locked RWKV 2.9B model once, completed in about 21.07 seconds, and used 6,418,167,808 bytes of peak CUDA memory. The report is structurally complete and records all 42 frozen calls. It is an effective, validly recorded experiment attempt, but its acceptance result is `valid=false` and `status=d5c_mechanism_smoke_failed`.

## Passed checks

- All 42 model-forward calls and all three groups of 24 comparisons are present.
- Every output is finite and every compared logits/state pair has compatible shape, dtype, device, and component paths.
- All 24 within-route comparisons are exact, so each scored route is internally stable.
- Both fixtures show the same deterministic route pattern; the failure is not isolated to only the single-token or sequence path.

## Failed checks and route signature

Four acceptance checks failed:

1. `all_control_pairs_exact=false`.
2. `active_differs_from_each_control=false`.
3. `callback_invocations_exact=false`.
4. `probe_applications_exact=false`.

For each fixture, all four OFF-versus-zero comparisons are exact, while all four original-versus-OFF and all four original-versus-zero comparisons differ. Conversely, all four active-versus-original comparisons are exact, while active differs from OFF and zero. Across both fixtures this gives only 8/24 exact control comparisons and 8/24 exact active-control comparisons. The same logits and `state[48]` digest grouping is repeated in all four scored rounds.

The callback recorded 576 invocations and 18 layer-15 applications instead of the frozen 320 and 10. The observed arithmetic is exact: `576 = 18 × 32`. The ten intended active calls account for ten applications; the eight additional applications match the eight scored `original_uninstrumented` calls, each scheduled immediately after an active call. Those eight original-labelled outputs are also bitwise identical to their active counterparts.

This is strong evidence of a post-active route-isolation or binding-lifecycle contamination signature. It does not yet prove whether the persistence is caused by Python instance bindings, method/decorator caching, the upstream dispatch boundary, or another implementation detail. No lower-level cause is claimed without a separate offline diagnostic.

## Decision boundary

The frozen decision is `stop_without_rerun`. The consumed D5C attempt must not be rerun, and the failed route-isolation gate blocks D5D and D5E. The output cannot support a Self-effect conclusion: the injected object was synthetic rather than a Self representation, and the route contamination prevents the intended causal contrast from being interpreted.

No D4, D4B, or D5C rerun is authorized. No automatic rerun, formal test set, real Self projection, Self Updater, D5D, or D5E is authorized. A possible next step would be a separately confirmed, offline-only binding-lifecycle diagnostic using the existing report and fake/source inspection; it must not load or execute the model or create a replacement D5C attempt.
