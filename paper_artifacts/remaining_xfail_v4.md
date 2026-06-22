# Remaining XFAIL V4

| Item | Status | Reason |
| --- | --- | --- |
| Gate A1 | blocked | real replay update_event_trace convergence and UT-12/13B not complete |
| Gate A2 | blocked | actual size12m HTS/larger-flat counts and full P0 optimizer/checkpoint reload smoke pending |
| Gate B | blocked | real synthetic trainer/evaluator and checkpoint-derived metrics pending |
| larger_flat_flops | P1 | FLOPs estimator/search not implemented |
| DMC/DMC-GB2 | P1 | not wired in official port |
| MiniHack/KeyCorridor | P1 | THICK-compatible wrapper not implemented |
