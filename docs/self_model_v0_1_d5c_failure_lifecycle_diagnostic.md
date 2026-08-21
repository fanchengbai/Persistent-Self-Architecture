# Self Model v0.1 D5C failure lifecycle diagnostic

## Authority

On 2026-08-21, the project owner confirmed:

> 确认进入 Self Model v0.1 D5C失败纯离线绑定生命周期诊断设计与无模型实现；仅使用现有报告、冻结源码和fake fixture，不导入RWKV/Torch、不访问权重、不加载或执行模型、不修改D5C失败结论，也不授权D5D/D5E、正式测试集、Self效果结论、真实Self projection、Self Updater或自动重跑。

This permits only deterministic analysis of the frozen D5C report summary, source inspection, and pure-Python fake fixtures. It does not authorize a corrected real entry, a replacement claim, or any model access.

## Frozen facts

D5C completed all 42 calls on commit `a8ef52a`; report digest `187cdfd4…db21` and consumed claim `75d69ae3…f12f` were independently verified. All 24 within-route comparisons were exact. In both fixtures, OFF equaled zero, active equaled every scored original, and original differed from OFF and zero. Callback counts were 576 invocations and 18 applications rather than 320 and 10.

The excess is arithmetically exact: eight extra applications and `8 × 32 = 256` extra callback invocations. The frozen schedule places all eight scored original calls immediately after an active call. Therefore the observed report establishes a post-active route-isolation contamination signature but cannot identify one low-level cause from order alone.

## Offline fake question

The diagnostic runs the exact project runtime binding code against a pure-Python 32-layer, 2560-wide fixture for both `forward_one` and `forward_seq`. It compares a raw baseline, one active call, and a raw original call after active. It also inspects the instance dictionary before and after the wrapper call and verifies callback counters.

If the post-active raw call returns to baseline and the callback counter does not advance, ordinary Python instance binding cleanup is not sufficient to reproduce the real failure. That result weakens—but does not eliminate—the direct-dictionary-cleanup hypothesis and leaves real upstream dispatch, decorator, compiled-method, or cache boundaries as unresolved candidates. The diagnostic must not rank one of those candidates as proven without new evidence.

## Decision boundary

A valid diagnostic package does not make D5C valid. The D5C result remains failed and non-rerunnable. D5D and D5E remain closed, and no Self-effect conclusion is permitted. Any future work requires a new, separately bounded authorization and must not silently recreate a D5C real execution opportunity.

## Offline result

Both fake paths passed. After one active call, the instance dictionary contained none of the two temporary forward methods or the callback attribute. The following raw original call returned exactly to its pre-active baseline, differed from the active output, and did not advance the callback beyond 32 invocations and one layer-15 application.

Therefore the real post-active signature is not reproduced by ordinary Python binding and cleanup on the locked-shape fake fixture. The confirmed schedule confound and real-only failure remain, while one low-level root cause remains unresolved. A later source-level dispatch/cache audit would require separate confirmation and still could not authorize a model run.
