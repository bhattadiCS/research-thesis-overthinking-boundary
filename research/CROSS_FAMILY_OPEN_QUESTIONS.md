# Cross-Family Open Questions

| Question | Status | Joint answer |
| --- | --- | --- |
| Is the boundary robust across model families? | answered | A late boundary is present in at least two capable families under the matched GSM8K protocol. |
| Does boundary location appear capability-linked? | answered | The weak Qwen control stays early while the higher-capability Qwen run moves later, which supports a capability-linked boundary location. |
| Does detector ranking change with capability? | answered | Detector ranking changes across runs, so ranking is not invariant across capability regimes. |
| Is answer revision or entropy more cross-family stable? | partially answered | Signal leadership is not stable yet: DeepSeek emphasizes answer revision, while Qwen-family evidence includes entropy or verbosity proxies. |
| Do the data support a family effect or mostly a capability effect? | partially answered | A late boundary in multiple capable families weakens the case for a DeepSeek-only effect, but one benchmark and one capable run per family still leave family-versus-capability attribution incomplete. |
| What cannot yet be claimed without the stronger second-family run? | answered | Even with the Qwen 7B run, the repo still cannot claim benchmark-invariant behavior, clean family-versus-capability separation, or universal observable stability without additional families or benchmarks. |
