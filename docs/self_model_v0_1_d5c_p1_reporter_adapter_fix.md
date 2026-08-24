# D5C-P1 reporter explicit-adapter fake-first fix

This change removes `values`-name inference from the D5C-P1 reporter. The default path now requires the real tensor serialization protocol. Pure-Python fixtures are supported only when the caller explicitly supplies an adapter whose `accepts` method selects the exact fixture type and whose `payload` method returns the offline fingerprint payload.

The real D5C-P1 runner does not construct or pass an offline adapter. Unknown tensor-like objects therefore fail closed instead of being silently reclassified by an attribute name.

The frozen nine-category synthetic acceptance covers the historical collision, the repaired real-like default path, exact fixture routing, unknown-object rejection, the two rejected alternative strategies, non-observation of callable `values`, name-independent default serialization, and source immutability during verification. The full 32-layer pure-Python fixture also exercises all 12 calls through the explicit adapter.

This is a no-model engineering fix. It does not import RWKV or Torch, access weights, load or execute a model, authorize or perform any D5C/P1 rerun, change either historical conclusion, or open D5D/D5E or any Self-effect gate.
