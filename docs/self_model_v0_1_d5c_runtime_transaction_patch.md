# Self Model v0.1 D5C runtime transaction patch

This round moves the previously accepted fake transaction into `RWKV7D5CActiveRuntime` and validates the real project wrapper only with synthetic Python fixtures.

The runtime now captures the original binding state, installs the callback and transformed methods, executes at most one public forward, restores every managed binding through the object protocol, verifies instance ownership plus static and resolved method identity, and returns an output only after verification succeeds. Cleanup or identity failure discards a produced output and raises `D5CCleanupTransactionError`. Nested or concurrent use of the same model is rejected before inner mutation.

The no-model acceptance covers single and sequence dispatch, cooperative and sticky side-dispatch behavior, partial installation, forward failure, cleanup failure, nested use, deterministic concurrency, active lifecycle restoration, and the no-extra-forward rule.

This does not rerun D5C, import RWKV or Torch, access weights, load or execute the 2.9B model, authorize D5D/D5E, or change the frozen D5C failure conclusion. The next gate is owner-run remote no-model verification and result observation.
