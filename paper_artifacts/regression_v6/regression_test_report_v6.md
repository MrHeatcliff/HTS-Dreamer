# Regression Test Report V6

Status: `pass`

| id | name | status |
| --- | --- | --- |
| RT-01 | dreamer_anchor unchanged | PASS |
| RT-02 | disabling all HTS scales recovers anchor loss path | PASS |
| RT-03 | hts_no_temp differs only by temporal loss | PASS |
| RT-04 | hts_no_vc differs only by VC loss | PASS |
| RT-05 | hts_no_hier differs only by hierarchy reconstruction loss | PASS |
| RT-06 | hts_no_sdyn differs only by sparse-dynamics loss | PASS |
| RT-07 | dense_multistride_no_sparse differs only by TopK/L1 | PASS |
| RT-08 | flat_partition_dim_matched has active flat reconstruction gradient | PASS |
| RT-09 | larger_flat_param matches flat_mh objective except width | PASS |
