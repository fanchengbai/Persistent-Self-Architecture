# D5C-P1 offline reporter dispatch diagnosis and fix design

The P1 failure is reproduced without RWKV or Torch by a synthetic real-like tensor whose `values` member is callable. The frozen reporter uses only `hasattr(value, "values")`, then sends that member to JSON serialization. This reproduces the same name-collision boundary as the real failure.

Adding only `not callable(value.values)` is insufficient: an unrelated production object can expose a non-callable data property with the same name and still be silently routed to the fixture serializer. An object-owned marker is also insufficient as the sole boundary because it couples production dispatch to spoofable object state.

The recommended future design removes attribute-name inference. The default path always uses the real tensor serializer. Offline tests must explicitly inject a test-only adapter that accepts the exact offline fixture type and serializes it. The real runner supplies no adapter. Unknown objects fail closed instead of being guessed from their attributes.

This round does not implement that design. It does not modify the frozen reporter, import RWKV or Torch, access weights, load or execute a model, authorize a P1 rerun, or change D5C/P1 conclusions.
