# Development Candidate Selection V18

Decision: `PASS_WITH_LOCKED_PROTOCOL_DEVELOPMENT_CANDIDATE`

Observation: variants were compared only to `locked_baseline_direct_head`.

| variant | gain delta | AUPRC | macro | factor | status | reason |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `locked_hier_x3` | 0.002379 | 0.713103 | 0.587390 | 0.421165 | `pass` |  |
| `locked_no_hier_loss` | -0.000963 | 0.706388 | 0.592902 | 0.404815 | `reject` | prefix_requirement_failed |

`locked_recon_trunk_isolated_fine_only_x3` is not applicable under direct-head locked protocol.
