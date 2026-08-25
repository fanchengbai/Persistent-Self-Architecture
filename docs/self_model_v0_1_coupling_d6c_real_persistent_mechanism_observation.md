# Coupling-D6C real persistent mechanism observation

## Outcome

The single authorized D6C real 2.9B attempt failed after model loading and before the first model forward. Its status is `d6c_execution_attempt_failed_claim_consumed`; the single-use claim is consumed and D6C may not be rerun.

The run used clean-main commit `5416fc782df10b88741be2b9066f0c8f4967eb05`. Before execution, both the machine authorization and execution claim were absent. The runner then created both records, loaded the frozen 2.9B checkpoint, and stopped while constructing `RWKV7D6CPersistentRuntime`.

## Integrity chain

The pasted authorization, claim, and failure records were independently recomputed from their canonical JSON payloads:

- authorization internal digest: `57eb45083895d8b83a39c563b7665bf1567acaeeb6b16ba0c1692fad7354307c`;
- authorization file SHA-256: `7b1e128b0d4922c42ea35ec4c105e726b58ce67da2602fabb9d2e669fe27438d`, equal to the claim binding;
- claim file SHA-256: `82b94c33513da0137127ce44a85513c48c381d92b87e3a7c27916931821fe6a3`, equal to the failure binding;
- failure internal digest: `25fdd26b5b34bcb4b2adf81b8fc784d5c471c86fbedae24054b091015dadf273`.

All four recomputations matched their stored or downstream-bound values.

## Failure boundary

The runtime compiled the two instrumented methods, constructed the fixed dispatcher, and called `setattr` for the callback and both methods. It then called `_managed_snapshot`, which required all three names to be present directly in `base_model.__dict__`. The real RWKV object did not satisfy that storage assumption, producing `RuntimeError: D6C persistent bindings are incomplete`.

This is a real-object attribute-storage/object-protocol mismatch in the persistent installation check. It is not evidence that the AST transform, dispatcher, synthetic probe, or active residual effect failed. Conversely, none of those mechanisms was validated on the real model because execution never reached `execute_d6c_mechanism_core`.

The stack location is before the core and before `RWKV7D6CPersistentRuntime.forward` increments its execution count. Therefore the evidence supports zero D6C model-forward calls. The model weights were loaded, but no OFF, zero, or active-synthetic route was executed. Any partially installed Python bindings existed only in the failed process and were not persisted as an external model artifact.

## Research interpretation

D6C does not pass. It also does not answer the Self question. No real Self representation or projection was built, no Self condition was run, and no Self-effect conclusion is available.

The result closes this exact D6C attempt without rerun. D5C/P1/P2 and D6C reruns, automatic rerun, D6D/D6E execution, raw-original routing, the formal test set, real layer selection, real Self projection, Self-effect conclusions, and Self Updater remain unauthorized.

To avoid another mechanism-only detour, the recommended next decision is not a D6C patch-and-rerun. If the project continues, a new D6D design should combine a synthetic mechanism positive control and an actual frozen Self projection in the same non-Core experiment, while moving persistent ownership outside the real model instance dictionary. That would directly approach the core Self question while retaining a mechanism control. This is only a recommendation; D6D design or implementation is not authorized by the D6C result.
