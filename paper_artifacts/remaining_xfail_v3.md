# Remaining XFAIL v3

- `UT-15-P1` optional P1 status report: larger_flat_flops remains P1 pending and does not block P0
- `UT-12` regime-specific parameter deltas: requires automated one-step optimizer delta by module and regime
- `UT-13` diagnostic decoder detach: detached diagnostic decoder/probe optimizer not implemented
- `IT-01` tiny synthetic shard overfit: official HTS synthetic trainer not wired
- `IT-02` synthetic checkpoint evaluator smoke: current evaluator sample is structural placeholder
- `IT-03` short Atari smoke with complete artifacts: manual smoke exists; automated assertion pending
- `IT-04` periodic evaluation does not mutate training state: periodic eval not integrated
- `IT-05` checkpoint resume preserves optimizer and config: resume smoke not automated
- `IT-06` run-end replay-ratio consistency: writer exists; update-producing smoke pending
- `RT-01` dreamer anchor unchanged: baseline-vs-HTS regression pending
- `RT-02` disabling all HTS scales recovers anchor loss: zero-scale regression pending
- `RT-03` hts_no_temp differs only by temporal loss: ablation regression pending
- `RT-04` hts_no_vc differs only by VC loss: ablation regression pending
- `RT-05` hts_no_hier differs only by hierarchy loss: ablation regression pending
- `RT-06` hts_no_sdyn differs only by sparse-dynamics loss: ablation regression pending
- `RT-07` dense_multistride_no_sparse differs only by TopK/L1: official variant/regression pending
