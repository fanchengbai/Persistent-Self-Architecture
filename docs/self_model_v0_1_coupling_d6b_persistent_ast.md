# Coupling-D6B project-owned persistent AST integration

D6B connects the existing locked post-FFN AST transformation to the repository's pure-Python 32-layer, 2560-hidden-dimension fixture. It does not probe an installed RWKV package and does not import RWKV or Torch.

The transformed `forward_one` and `forward_seq` methods and one fixed dispatcher are installed exactly once before the fixture's first forward. Every later call uses those same object identities. OFF, zero, and active-fake requests are transported through a runtime-owned `ContextVar`; the runtime forward contains no model `setattr` or `delattr` operation.

Both execution paths are checked. OFF and zero must be exact and must not invoke the fake probe. Active-fake must change output deterministically, visit all 32 post-FFN sites, and apply at exactly one fixed synthetic layer per call. State and tokens are cloned before the fixture call so caller-owned inputs remain unchanged.

This is static/no-model integration evidence only. It does not validate installed source, weights, a real model, a real layer or projection, or a Self effect. D5C/P1/P2 remain stopped, and D6C–D6E require separate decisions.
