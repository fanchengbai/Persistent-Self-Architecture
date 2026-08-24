# Self Model v0.1 post-P1 research review

Date: 2026-08-24  
Status: offline research review only; no implementation or model execution

## Decision

The Coupling-D5C and D5C-P1 line remains stopped. D5C consumed its single-use claim and failed route isolation. P1 consumed a separate single-use claim and stopped after the first original output because of a reporter defect. The reporter defect is now repaired and verified across hosts, but that tooling repair does not restore the consumed P1 opportunity or turn the missing patched-route evidence into a pass.

Creating a P2 with the same model, schedule, routes, and engineering question would be a substantive P1 rerun under a new name. It is rejected. Coupling-D5D and D5E remain blocked because their mechanism prerequisite never passed.

## Evidence that remains valid

- D4B established steady-state OFF equivalence for the locked project AST transformation.
- D5A and D5B established the offline active contract and project-owned static path.
- D5C established a real route-isolation failure for per-call temporary method binding.
- The transaction patch passed synthetic lifecycle acceptance, but P1 produced no real patched-route evidence.
- The explicit reporter adapter repair passed 21 remote tests, 13 report checks, nine frozen acceptance categories, and the complete 12-call pure-Python fixture.

None of these facts establishes a real 2.9B active Self effect.

## Rejected continuations

1. Reuse the D5C or P1 authorization, claim, output path, or execution schedule.
2. Create a nominal P2 that asks whether the same temporary-binding patch works after fixing the reporter.
3. Open D5D or D5E based only on fake acceptance or reporter correctness.
4. Treat synthetic projection differences as Self semantics.
5. Switch methods or callback attributes on the model object between scored calls.

## Recommended architecture pivot: Coupling-D6

Coupling-D6 uses a persistent instrumented instance rather than per-forward temporary binding:

- compile and install the locked instrumented methods once before the instance's first forward;
- bind one fixed project-owned dispatcher once and never replace or delete it during the instance lifetime;
- carry OFF, zero, or active coupling as an explicit context-local request consumed by that dispatcher;
- keep method descriptors, resolved methods, and the model instance dictionary stable across every scored call;
- reject nested or concurrent requests unless request isolation is proven;
- make OFF and zero return the residual unchanged without constructing a projection;
- retain source Self State and recurrent-state immutability and fail before output commit on invalid requests or non-finite projection;
- never compare a post-active raw-original route on the same instance, because that is the failed temporary-binding design.

This is a materially different lifecycle architecture and a new engineering question. It does not retroactively pass D5C or P1.

## Proposed gate ladder

- Coupling-D6A: pure-Python persistent-dispatch contract and fake lifecycle acceptance; no model.
- Coupling-D6B: project-owned persistent AST integration and static/no-model verification; no model.
- Coupling-D6C: separately authorized real 2.9B non-Core mechanism gate for the persistent architecture; no Self-effect claim.
- Coupling-D6D: non-Core Self semantic pilot only if D6C passes and after a new design and authorization.
- Coupling-D6E: formal preregistered Self-effect experiment only after all earlier gates pass.

Every gate requires a separate owner decision. No existing authorization carries forward.

## Exact next confirmation

The next implementation round is limited to Coupling-D6A:

> 确认进入 Self Model v0.1 Coupling-D6A persistent-instrumented dispatcher纯离线contract与fake lifecycle实现；只使用合成Python fixture验证一次安装、固定dispatcher、context-local OFF/zero/active请求、无模型对象属性切换、嵌套/并发失败关闭及输入不变性；不授权D6B/D6C/D6D/D6E、RWKV/Torch导入、权重访问、模型加载或执行、D5C/P1/P2重跑、真实层选择、真实Self projection、Self效果实验、Self Updater或自动重跑。
