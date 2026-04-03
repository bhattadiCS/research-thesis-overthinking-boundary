# Cross-Family Open Questions

| Question | Status | Joint answer |
| --- | --- | --- |
| Is the boundary robust across model families? | partially answered | A clearly late boundary is still supported in only one capable family, but Mistral adds weaker second-family support with a corrected step-3 crossing and a large never-stop penalty. Cross-family robustness is stronger than before, but not yet settled. |
| Does boundary location appear capability-linked? | answered | The weak Qwen control stays early while the higher-capability Qwen run moves later, which supports a capability-linked boundary location. |
| Does detector ranking change with capability? | answered | Detector ranking changes across runs, so ranking is not invariant across capability regimes. |
| Is answer revision or entropy more cross-family stable? | partially answered | Signal leadership is not stable yet: DeepSeek emphasizes answer revision, while Qwen-family evidence includes entropy or verbosity proxies. |
| Do the data support a family effect or mostly a capability effect? | partially answered | The completed Mistral run weakens the case for a single-family artifact, but its earlier step-3 boundary still leaves family-versus-capability attribution incomplete and below a strong cross-family late-boundary claim. |
| What cannot yet be claimed without the stronger second-family run? | answered | Even with the completed Mistral follow-up, the repo still cannot claim benchmark-invariant behavior, clean family-versus-capability separation, or universal observable stability without additional families or benchmarks. |
