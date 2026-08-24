# Coupling-D6A persistent-instrumented dispatcher

Coupling-D6A is a pure-Python lifecycle contract for the post-P1 architecture pivot. It does not reuse or rename the failed D5C/P1 experiment.

The synthetic model receives three instance bindings exactly once in the runtime constructor: one fixed dispatcher and persistent `forward_one` and `forward_seq` methods. No model attribute is installed, replaced, or deleted during a forward. OFF, zero, and active behavior is carried by a `ContextVar` request visible to the fixed dispatcher only for the duration of one call.

OFF and zero return the residual unchanged and do not call the synthetic probe. Active calls a deterministic synthetic probe. The runtime rejects nested and concurrent requests before another inner forward, restores the context after success or callback failure, remains reusable after failure, and verifies that token/state inputs and model binding identities remain unchanged.

This round uses no RWKV or Torch import, weight, model, real layer, real projection, or Self-effect data. The historical D5 runtime is digest-locked and unchanged. D5C/P1/P2 reruns and D6B–D6E remain unauthorized.
