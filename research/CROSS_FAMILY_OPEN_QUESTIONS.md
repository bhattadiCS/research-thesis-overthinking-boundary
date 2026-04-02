# Cross-Family Open Questions

| Question | Status | Joint answer |
| --- | --- | --- |
| Is the boundary robust across model families? | partially answered | A late boundary is only supported in one capable family so far, so cross-family robustness is still unproven. |
| Does boundary location appear capability-linked? | answered | The weak Qwen control stays early while the higher-capability Qwen run moves later, which supports a capability-linked boundary location. |
| Does detector ranking change with capability? | answered | Detector ranking changes across runs, so ranking is not invariant across capability regimes. |
| Is answer revision or entropy more cross-family stable? | partially answered | Signal leadership is not stable yet: DeepSeek emphasizes answer revision, while Qwen-family evidence includes entropy or verbosity proxies. |
| Do the data support a family effect or mostly a capability effect? | partially answered | The matched benchmark now includes multiple families, but the evidence still cannot cleanly isolate family effects from capability effects. |
| What cannot yet be claimed without the stronger second-family run? | answered | Even with the Qwen 7B run, the repo still cannot claim benchmark-invariant behavior, clean family-versus-capability separation, or universal observable stability without additional families or benchmarks. |
