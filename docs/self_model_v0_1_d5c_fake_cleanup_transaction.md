# Self Model v0.1 D5C fake-first cleanup transaction

This gate implements the cleanup transaction only for synthetic Python objects. It does not modify the real D5C runtime or instrumenter, import RWKV or Torch, access weights, execute a model, or authorize a D5C rerun.

## Transaction

The transaction snapshots instance ownership, static descriptors, resolved method functions, and callback absence before installing any managed name. It installs through `setattr`, attempts every restoration in reverse order, and verifies the complete snapshot before returning a produced output. Cleanup or verification failure discards the output and raises a structured failure. A primary forward exception is preserved when restoration succeeds and attached when restoration also fails.

Verification uses attribute and descriptor inspection only; it never adds a model forward call. A lock-protected weak object registry rejects reentrant or concurrent use of the same object before the inner transaction mutates managed names.

## Acceptance boundary

The synthetic suite covers both execution paths, three standard decorator forms, cooperative side state, sticky noncooperative side state, two partial-install failures, a forward exception, a cleanup exception, post-cleanup identity mismatch, nested rejection, and forward-call counting. Cooperative cases must restore. Sticky or cleanup-error cases must fail closed after discarding any produced output.

Passing this suite produces only a fake candidate. It does not show that the real RWKV/Torch object has the synthetic side state, does not prove a real fix, and does not authorize modifying `d5c_mechanism_runtime.py`. Real patch implementation, model validation, D5C rerun, D5D/D5E, formal test data, Self-effect claims, real Self projection, Self Updater, and automatic reruns remain separate gates.
