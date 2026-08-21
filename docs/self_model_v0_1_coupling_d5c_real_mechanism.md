# Self Model v0.1 Coupling-D5C real 2.9B non-Core mechanism smoke

## Purpose

D5C asks only whether the project-local active coupling path can inject a deterministic synthetic residual into the locked real 2.9B RWKV execution path while OFF and zero-scale controls remain exactly equivalent. It is a mechanism smoke test, not a Self Model effectiveness experiment.

The injected vector is synthetic, untrained, semantically empty, and explicitly not a Self representation or a real Self projection. Layer 15 (zero-based) is fixed by `floor((32-1)/2)` before any output is observed; it is not selected from behavior or effect measurements.

## Frozen fixtures and schedule

Two non-Core fixtures are used: one single token `[2764]` with `full_output=false`, and one sequence `[187, 931, 2764]` with `full_output=true`. Each fixture has one unscored original prefix call, four fixed preconditioning calls, and sixteen scored calls in a 4x4 Latin schedule. The four routes are original uninstrumented, wrapper OFF, wrapper zero scale, and wrapper active synthetic probe. Total model-forward calls are 42.

For each active forward, the callback is visited once after the FFN residual at every layer and applies only at zero-based layer 15. Across ten active calls, the expected callback count is 320 and the expected application count is 10.

## Probe

A deterministic length-2560 vector is normalized to unit RMS. At layer 15, the delta magnitude is `0.01 * RMS(residual_x)` with gate `1.0`; for sequences the same vector is broadcast to each position. Inputs, deltas, and outputs must be finite, and shape, dtype, and device must be preserved. The callback fails closed on any mismatch.

## Acceptance and interpretation

All within-route comparisons must be exact. Original, OFF, and zero controls must be pairwise exact in every scored round. Active must differ from each control in logits or recurrent state while remaining compatible and finite. These checks establish only that the active mechanism is connected and bounded. They cannot establish a beneficial, persistent, self-related, behavioral, or causal Self effect.

## Safety gate

The committed runner refuses execution without all of the following: exact environment lock, exact owner text encoded in a machine authorization, clean `main`, frozen config and source digests, empty output directory, and a single-use claim created before model configuration, weights verification, loading, or execution. A post-claim failure is recorded and consumes the attempt. No automatic rerun is authorized.

D5D, D5E, formal test sets, Self-effect conclusions, real Self projection, and Self Updater remain outside D5C authority.
