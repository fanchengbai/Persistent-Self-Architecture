# Self Model v0.1 D5C failure dispatch/cache source audit

This development-only gate audits the frozen AST transformation, D5C wrapper, method/decorator dispatch boundary, and the existing plain-Python fake. It is source-only: it does not import RWKV or Torch, read weights, load or execute a model, implement a fix, or authorize a rerun.

## Confirmed source facts

The instrumenter calls `ast.parse` on locked source text, removes decorators from those newly parsed function nodes, compiles them into a copied globals dictionary, and then binds the new functions to the model instance. This rules out the specific theory that the AST transformer directly mutates the already-loaded class method objects.

The real frozen source evidence records `@MyFunction` on both original methods, while the compiled active methods have no decorators. The wrapper installs its callback and two temporary bound methods through `setattr`, but its `finally` block removes all three names with direct `instance.__dict__.pop`. It does not use `delattr`, verify the resolved method/callback identity after cleanup, or synchronize an upstream decorator/framework cache. This is a confirmed object-protocol asymmetry and an audit gap, not proof that any cache exists or caused the failure.

The existing lifecycle fake defines plain undecorated `forward_one` and `forward_seq` methods. Its successful cleanup therefore tests ordinary Python instance shadowing, but not the real transition from an instance-bound compiled undecorated function back to a class-level decorated method. The audited project files also do not contain the frozen definition or implementation of `MyFunction`, nor the internals of the `torch.nn.Module` attribute protocol, so this round cannot resolve either behavior.

## Interpretation

The result excludes direct mutation of loaded original method objects and confirms two coverage risks: asymmetric install/cleanup and an untested decorator/descriptor boundary. It does not select one as the root cause. The real D5C failure and `stop_without_rerun` decision remain unchanged.

Any decorator-aware offline boundary fixture, cleanup redesign, or repaired D5C execution is a separate gate requiring explicit project-owner confirmation. D5D/D5E, formal test data, Self-effect claims, real Self projection, Self Updater, and automatic reruns remain unauthorized.
