# Self Model v0.1 D5C-P1 real engineering observation

## Outcome

The single authorized D5C-P1 attempt ran on 2026-08-24 at clean `main` commit `1bc58579ddc0ff91a8ad37f83044f2046d2ccc16`. The machine authorization and new P1 single-use claim were created. The 2.9B model was loaded, and the first `original_before` forward returned. The attempt then failed while fingerprinting that first output, before any patched OFF, active, post-active original, or zero route ran.

Status is `d5c_p1_attempt_failed_claim_consumed`, `valid=false`, and no rerun is authorized.

## Failure boundary

`_tensor_payload` classified an offline fixture by checking `hasattr(value, "values")`. A real `torch.Tensor` also exposes a callable `values` member. The code therefore treated the real Tensor as an offline fixture and passed the builtin method to `sha256_json`, which raised:

`TypeError: Object of type builtin_function_or_method is not JSON serializable`

This is a reporter/type-discrimination defect. It is not evidence that the transactional runtime patch failed, because no patched route was reached. It is not a model failure and supports no mechanism-connectivity or Self-effect conclusion.

## Integrity

The returned artifacts were independently recomputed with the project canonical JSON format:

- authorization internal digest valid: `c7d78281d93e85814a61fb8fdbb71696495b24981f5b2e944f42cf3e30daf37e`;
- authorization file digest bound by claim: `8b0f34cbc0b2b57a5ffffefb801128ff7e3243283ce6ae182c37ee414c3c2f0f`;
- claim file digest bound by failure: `7c49107b33a223be7ac11f3412328abc07e24e5fc6bcf68accfbe34b7ca97628`;
- failure report digest valid: `930c31ef6f70c431066cda3637c97fcc35344b8caabaeb2f2f7147a0b5d54483`.

The machine authorization preserved every exclusion. The claim explicitly records that the historical D5C claim was not reused and that historical D5C rerun was not authorized.

## Decision

The P1 opportunity is consumed and stopped. The historical D5C failure remains unchanged. D5D/D5E, the formal test set, Self-effect conclusions, a real Self projection, the Self Updater, and automatic rerun remain closed.

Any next work requires a separate authorization limited to offline failure diagnosis and reporter-fix design. Such work cannot authorize a repaired real-model run.
