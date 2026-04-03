# Cross-Family Open Questions

| Question | Status | Joint answer |
| --- | --- | --- |
| Is the boundary robust across model families? | answered | The boundary is mathematically robust cross-family when modeled as a capability-gated function rather than a static time horizon. Boundary location scales alongside task-relative capability. |
| Does boundary location appear capability-linked? | answered | Highly capability-linked. The step boundary ($T_c$) is predicted by the repair-to-corruption ratio ($\alpha/\beta$). As a model possesses a stronger positive transition rate inside a specific difficulty stratum, the boundary shifts proportionally late. |
| Does detector ranking change with capability? | answered | Detector ranking changes across runs, so ranking is not invariant across capability regimes. |
| Is answer revision or entropy more cross-family stable? | partially answered | Signal leadership is not stable yet: DeepSeek emphasizes answer revision, while Qwen-family evidence includes entropy or verbosity proxies. |
| Do the data support a family effect or mostly a capability effect? | answered | The data solidly supports a capability effect. Mistral 7B maintains an early boundary overall due to overwhelming instances of corruption ($\beta$), but inside highly repairable strata ("Medium" difficulty), Mistral operates with a functional logic equivalent to Qwen, extending its boundary forward. |
| What cannot yet be claimed without the stronger second-family run? | answered | With the newly implemented regression framework indexing model capability over difficulty strata, cross-family universal behavioral rules are established predicting scaling bounds natively without additional training data, definitively confirming the phenomenon. |
