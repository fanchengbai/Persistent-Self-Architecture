# Coupling-D6C real execution authorization and pre-execution hold

On 2026-08-25, the project owner supplied the exact authorization frozen in the D6C configuration:

> 授权执行 Self Model v0.1 Coupling-D6C 真实2.9B persistent-instrumented非Core机制验证一次（冻结两个非Core形状、每形状1次OFF预条件和4轮OFF/zero/active拉丁调度，共26次调用；固定synthetic probe与层访问计数），并授权观察本次机制结果；不授权重跑D5C/P1/P2或D6C、自动重跑、D6D/D6E、raw-original路线、正式测试集、真实层选择、真实Self projection、Self效果结论或Self Updater。

The authorization was received after the D6C no-model gate on commit `5b94a57bb4376aa7621033da992619c8862ec4dd`. Before generating any execution command, a read-only lifecycle audit found that the runner would create the machine authorization and then reconstruct the bound static report using a live check that required that same authorization to be absent. The run would therefore fail before installed-source inspection and claim creation.

No machine authorization or claim was created. The installed source was not inspected, no weights were accessed, and no model was loaded or executed.

The no-model fix preserves the protocol and keeps initial creation strict: authorization and claim must both be absent. During later validation of the newly created machine authorization, the code now reconstructs the same frozen pre-authorization report digest without misclassifying the authorization file itself as a violation. A new regression test exercises the create-then-validate lifecycle.

Because the entry source and locked test digest changed after the owner's authorization, the authorization is recorded but held rather than consumed. The fixed commit must pass a new remote no-model verification, after which the owner must repeat the exact authorization before any D6C execution lock, machine authorization, claim, installed-source probe, weight access, model load, or forward call is allowed.
