PASS: 34 | XFAIL: 1 | FAIL: 0

| test_id | test_name | status | failure_reason |
| --- | --- | --- | --- |
| UT-01 | six HTS head shapes | PASS |  |
| UT-02 | TopK per level active budget | PASS |  |
| UT-03 | nested prefix input contract | PASS |  |
| UT-04 | decoder lower-prefix stop-gradient | PASS |  |
| UT-05 | coarse-to-fine stride mapping | PASS |  |
| UT-06 | action-window indexing | PASS |  |
| UT-07 | terminal/reset masking | PASS |  |
| UT-08 | temporal positive sampler validity | PASS |  |
| UT-09 | far-negative modes | PASS |  |
| UT-10 | VICReg anti-collapse behavior | PASS |  |
| UT-11 | weighted objective equality | PASS |  |
| UT-12 | training-regime parameter deltas | PASS |  |
| UT-13A | decoder prefix stop-gradient trace | PASS |  |
| UT-13B | predictor prefix stop-gradient trace | PASS |  |
| UT-13C | dynamics target stop-gradient trace | PASS |  |
| UT-13D | detached synthetic linear probe path | PASS |  |
| UT-14 | synthetic evaluation labels excluded from training | PASS |  |
| UT-15-MATRIX | component matrix V7 typed parity | PASS |  |
| UT-15-P0 | all P0 one-step smoke rows | PASS |  |
| IT-01 | tiny synthetic shard overfit | PASS |  |
| IT-02 | synthetic checkpoint evaluator | PASS |  |
| IT-03 | short Atari artifact plumbing smoke | PASS |  |
| IT-04 | periodic eval state isolation | PASS |  |
| IT-05 | checkpoint resume plumbing | PASS |  |
| IT-06 | real replay ratio convergence | PASS |  |
| UT-15-P1 | P1 optional controls | XFAIL | larger_flat_flops remains P1 |
| RT-01 | dreamer_anchor unchanged | PASS |  |
| RT-02 | disabling all HTS scales recovers anchor loss path | PASS |  |
| RT-03 | hts_no_temp differs only by temporal loss | PASS |  |
| RT-04 | hts_no_vc differs only by VC loss | PASS |  |
| RT-05 | hts_no_hier differs only by hierarchy reconstruction loss | PASS |  |
| RT-06 | hts_no_sdyn differs only by sparse-dynamics loss | PASS |  |
| RT-07 | dense_multistride_no_sparse differs only by TopK/L1 | PASS |  |
| RT-08 | flat_partition_dim_matched has active flat reconstruction gradient | PASS |  |
| RT-09 | larger_flat_param matches flat_mh objective except searched width | PASS |  |
