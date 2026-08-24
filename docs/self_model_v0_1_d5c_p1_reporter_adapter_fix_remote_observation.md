# D5C-P1 reporter adapter fix: remote no-model observation

The first remote no-model verification ran at commit `036d7ad1e998e21d7eb7038070d5d0bbedf9ed5f`.

The adapter-fix verifier passed all 13 report checks and all nine frozen acceptance categories. The full 32-layer pure-Python fixture completed its fixed 12-call plan, the real-runner AST contained one core call with no `offline_adapter` keyword, and the report digest matched the local result: `3200210fa67f871bf96821d6bac384f79bc37ed874e379706581c90db5bcb95f`. All safety fields remained closed, including model execution and P1 rerun.

The combined 21-test command reported one failure in the legacy real-entry test. That test asserted that the historical P1 machine authorization path did not exist. On the remote host it correctly still exists because the prior one-shot P1 attempt created and consumed it. This was an environment-state assumption in the test, not a reporter, adapter, runtime, or safety failure.

The test is revised to snapshot the optional historical authorization and claim before the wrong-config check and assert that their existence and SHA-256 values are unchanged afterward. It now works both in a clean local checkout and on a research host that preserves consumed historical evidence. It does not create, remove, or alter either artifact.

The correction remains no-model and does not authorize a D5C/P1 rerun or change any historical conclusion.

The second remote no-model verification passed all 21 tests. The verifier again passed all 13 report checks and all nine frozen acceptance categories, including the full fixed 12-call pure-Python plan. Its digest exactly matched the corrected local inventory: `7f36c30d41e7216435897bb0ee5fbccf7ea9a0b17a70376264fb037e60e5915a`. The report confirmed zero `values` attribute reads in the default reporter path, one real-runner core call with no offline adapter, and all RWKV, Torch, weight, model, rerun, D5D/D5E, formal-set, projection, updater, and Self-effect safety fields closed.

The pasted second-run excerpt did not repeat `git rev-parse HEAD` or `git status --short`. However, every source digest in the locked inventory, including the corrected real-entry test digest, matched the corrected local report. The evidence therefore closes the cross-host source and no-model behavior gate while making no separate claim about an unshown porcelain-status line.
