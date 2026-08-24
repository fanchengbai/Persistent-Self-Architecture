# Self Model v0.1 D5C-P1 real execution authorization

On 2026-08-24 the project owner supplied the exact frozen authorization:

> 授权执行 Self Model v0.1 D5C-P1 补丁后真实2.9B非Core工程验证一次（固定两个夹具、每夹具6次、共12次调用），并授权观察本次工程结果；不授权重跑原D5C、不改变历史失败结论，也不授权D5D/D5E、正式测试集、Self效果结论、真实Self projection、Self Updater或自动重跑。

This authorizes one D5C-P1 engineering attempt and observation of that attempt. The future runner must create a new machine authorization and consume a new single-use P1 claim on a clean `main` before model configuration verification, weights access, or model loading. Success or failure consumes the opportunity and stops.

This authorization does not reuse the historical D5C authorization or claim, does not rerun D5C, and cannot change its historical failure conclusion. It does not authorize D5D/D5E, a formal test set, a Self-effect conclusion, a real Self projection, a Self Updater, or an automatic rerun.
