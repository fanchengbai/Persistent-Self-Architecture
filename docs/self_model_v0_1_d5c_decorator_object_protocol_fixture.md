# Self Model v0.1 D5C decorator/object-protocol boundary fixture

This gate uses synthetic Python objects only. It does not modify the real D5C runtime, import RWKV or Torch, read weights, load or execute a model, implement a fix, or authorize any rerun.

## Frozen matrix

Both `forward_one` and `forward_seq` are exercised. Standard Python objects cover plain methods, an identity decorator, and a non-caching non-data descriptor under both direct instance-dictionary deletion and `delattr`. A separate synthetic descriptor-based object explicitly maintains side-dispatch state from `__setattr__` and clears that state only from `__delattr__`.

The standard cases must all return to the original class method after cleanup. In the explicit side-dispatch fixture, direct `instance.__dict__.pop` must remove the visible instance keys while leaving the synthetic side state intact; a subsequent raw call then resolves the active method and callback. The same synthetic object must return to the original descriptor when cleanup uses `delattr`.

## Interpretation boundary

This matrix can establish that custom side-dispatch state plus direct dictionary cleanup is sufficient to reproduce the observed shape “instance keys absent, active method and callback still reachable.” It cannot establish that RWKV, Torch, `MyFunction`, or the real model has such state. Likewise, the synthetic `delattr` result is not a proposed, authorized, or model-verified fix.

The real D5C `stop_without_rerun` decision remains unchanged. Any real-runtime repair design is a separate owner-confirmed gate; D5C rerun, D5D/D5E, formal test data, Self-effect claims, real Self projection, Self Updater, and automatic reruns remain unauthorized.
