# Protocol Diff Matrix V17

Observation: the largest verified mismatch is model parameterization/checkpoint lineage, not evaluator formula.

| item | same | importance | reason |
| --- | --- | --- | --- |
| script_path | `false` | `high` | historical replay/eval was V12; V15 retraining/eval used causal audit harness |
| script_hash | `false` | `high` | different entrypoint and model parameterization |
| entrypoint | `false` | `high` | checkpoint lineage and training loop differ |
| config_hash | `false` | `high` | V15 route flags describe shared-trunk reconstruction; historical has no trunk |
| resolved_config | `false` | `high` | same coefficients but different trainable graph |
| sampler_seed | `false` | `medium` | ordering differs because historical starts from V9 checkpoint lineage |
| sampler_order | `false` | `medium` | could affect nonlinear optimization but does not explain architecture mismatch alone |
| stop_gradient_flags | `false` | `medium` | shared trunk introduced route-specific gradient flags |
| routing_flags | `false` | `high` | candidate minimal drift factor |
| checkpoint_selection | `false` | `high` | historical high checkpoint is not same protocol as V15 exact |
