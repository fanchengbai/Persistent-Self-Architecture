# Self Model v0.1 D5C offline fix design

This gate chooses a design for a future fake-first cleanup transaction. It does not implement that transaction, modify the real D5C runtime, import RWKV or Torch, access weights, execute a model, or authorize a D5C rerun.

## Strategy decision

Direct instance-dictionary deletion is rejected as an unverified cleanup because it bypasses the object deletion protocol and checks only visible keys. Replacing it with `delattr` alone is also insufficient: that worked in the cooperative synthetic fixture, but an unknown cache may not be cleared by `__delattr__`.

The recommended future fake-first design is a transaction with four phases: snapshot, install, restore, and verify. Before mutation it preserves the exact instance ownership state, static class descriptors, resolved method identity tokens, and callback absence. Restoration attempts every managed name in reverse order and keeps the primary forward exception. Verification compares ownership, values, static descriptors, resolved methods, and callback resolution with the snapshot.

A forward result may be returned only after cleanup verification succeeds. If cleanup or verification fails, the result is discarded and the operation fails closed. Verification must not add an extra real-model forward call.

## Future fake acceptance gate

The separate implementation gate must cover standard decorators, cooperative side state, noncooperative sticky state, partial installation, forward exceptions, cleanup exceptions, post-cleanup identity mismatch, and nested/concurrent rejection. A sticky state that survives `delattr` must be detected rather than silently accepted.

This design is not a real fix claim. The frozen D5C failure and `stop_without_rerun` decision remain unchanged. Fake implementation, real runtime modification, model validation, D5C rerun, D5D/D5E, formal test data, Self-effect claims, real Self projection, Self Updater, and automatic reruns all require separate authorization.
