# Locked Variant Contract V18

Legacy names are aliases only; the locked protocol is direct-head.

| legacy | locked name |
| --- | --- |
| `baseline_shared_trunk` | `locked_baseline_direct_head` |
| `hier_x3` | `locked_hier_x3` |
| `recon_trunk_isolated_fine_only_x3` | `locked_recon_trunk_isolated_fine_only_x3` |
| `no_hier_loss` | `locked_no_hier_loss` |

| variant | parameterization | train status | role | params |
| --- | --- | --- | --- | ---: |
| `locked_baseline_direct_head` | `direct_head` | `reuse_v17_locked_replay` | historical reproducibility anchor | 17160 |
| `locked_hier_x3` | `direct_head` | `continue_from_v9_250` | test stronger hierarchy reconstruction under locked direct-head protocol | 17160 |
| `locked_recon_trunk_isolated_fine_only_x3` | `direct_head` | `not_applicable_direct_head_no_shared_trunk` | legacy V14 routing candidate | 17160 |
| `locked_no_hier_loss` | `direct_head` | `continue_from_v9_250` | hierarchy reconstruction objective control | 17160 |
